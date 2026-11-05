# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory-lean drop-in for Megatron's native fused vocab-parallel cross entropy.

The native version casts the whole [s, b, vocab/tp] logits to fp32 and keeps
that fp32 softmax alive until backward (4 GiB per microbatch for Qwen3.5 at
seq 8192 x mbs 2 x 248320/4, plus the bf16 logits during the cast). This
version does the same fp32 math in row chunks: forward keeps only the bf16
logits plus per-token max / sum-exp, and backward recomputes the softmax per
chunk. Collectives match the native version (one MAX all-reduce, one SUM
all-reduce of [predicted_logits, sum_exp] per call), so every TP rank issues
the same sequence.

Enable with ``FLAGSCALE_CHUNKED_CE=1``; chunk size in rows (tokens) via
``FLAGSCALE_CHUNKED_CE_ROWS`` (default 2048, ~0.5 GiB fp32 per chunk at
vocab/tp = 62080). Only used for ``cross_entropy_fusion_impl=native``.
"""

import os

import torch
import torch.distributed as dist

_PATCHED = False


def _vocab_range(partition_vocab_size, tp_group):
    start = tp_group.rank() * partition_vocab_size
    return start, start + partition_vocab_size


class _ChunkedVocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, tp_group):
        rows = int(os.environ.get("FLAGSCALE_CHUNKED_CE_ROWS", "2048"))
        v = vocab_parallel_logits.size(-1)
        logits_2d = vocab_parallel_logits.reshape(-1, v)
        target_1d = target.reshape(-1)
        n = logits_2d.size(0)
        vocab_start, vocab_end = _vocab_range(v, tp_group)

        target_mask = (target_1d < vocab_start) | (target_1d >= vocab_end)
        masked_target = (target_1d - vocab_start).masked_fill(target_mask, 0)

        # Pass 1: local max over the vocab shard (exact in bf16, same value as
        # the max of the fp32 cast), then the global max across TP ranks.
        logits_max = torch.empty(n, dtype=torch.float32, device=logits_2d.device)
        for i in range(0, n, rows):
            logits_max[i : i + rows] = logits_2d[i : i + rows].amax(dim=-1).float()
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

        # Pass 2: predicted logit and sum(exp(logit - max)) per token, in fp32.
        pred_sum = torch.empty(2, n, dtype=torch.float32, device=logits_2d.device)
        arange = torch.arange(n, device=logits_2d.device)
        for i in range(0, n, rows):
            chunk = logits_2d[i : i + rows].float()
            chunk -= logits_max[i : i + rows].unsqueeze(-1)
            pred_sum[0, i : i + rows] = chunk[arange[: chunk.size(0)], masked_target[i : i + rows]]
            pred_sum[1, i : i + rows] = chunk.exp_().sum(dim=-1)
            del chunk
        pred_sum[0].masked_fill_(target_mask, 0.0)
        dist.all_reduce(pred_sum, op=dist.ReduceOp.SUM, group=tp_group)

        predicted_logits, sum_exp_logits = pred_sum[0], pred_sum[1]
        loss = torch.log(sum_exp_logits) - predicted_logits

        ctx.rows = rows
        ctx.shape = vocab_parallel_logits.shape
        ctx.save_for_backward(vocab_parallel_logits, logits_max, sum_exp_logits, target_mask, masked_target)
        return loss.view_as(target)

    @staticmethod
    def backward(ctx, grad_output):
        logits, logits_max, sum_exp_logits, target_mask, masked_target = ctx.saved_tensors
        rows = ctx.rows
        v = logits.size(-1)
        logits_2d = logits.reshape(-1, v)
        n = logits_2d.size(0)
        grad_1d = grad_output.reshape(-1).float()
        softmax_update = 1.0 - target_mask.float()
        arange = torch.arange(n, device=logits_2d.device)

        grad_input = torch.empty_like(logits_2d, dtype=torch.bfloat16)
        for i in range(0, n, rows):
            # softmax = exp(logit - max) / sum_exp, as in the native fp32 path.
            chunk = logits_2d[i : i + rows].float()
            chunk -= logits_max[i : i + rows].unsqueeze(-1)
            chunk.exp_()
            chunk.div_(sum_exp_logits[i : i + rows].unsqueeze(-1))
            chunk[arange[: chunk.size(0)], masked_target[i : i + rows]] -= softmax_update[i : i + rows]
            chunk.mul_(grad_1d[i : i + rows].unsqueeze(-1))
            grad_input[i : i + rows] = chunk
            del chunk
        return grad_input.view(ctx.shape), None, None


def chunked_vocab_parallel_cross_entropy(vocab_parallel_logits, target, tp_group):
    """Same contract as megatron's fused_vocab_parallel_cross_entropy."""
    return _ChunkedVocabParallelCrossEntropy.apply(vocab_parallel_logits, target, tp_group)


def apply_chunked_cross_entropy_patch():
    global _PATCHED
    if _PATCHED or os.environ.get("FLAGSCALE_CHUNKED_CE", "0") != "1":
        return
    from megatron.core.models.common.language_module import language_module

    # compute_language_model_loss looks the function up as a module global
    # (MTP losses go through the same method).
    language_module.fused_vocab_parallel_cross_entropy = chunked_vocab_parallel_cross_entropy
    _PATCHED = True
