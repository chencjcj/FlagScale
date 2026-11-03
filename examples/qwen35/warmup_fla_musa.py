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

"""Compile/autotune fla GDN triton kernels for Qwen3.5 / Qwen3.6 per-rank shapes on MUSA.

The triton cache (~/.triton) is per node and keyed by shape, so the first training
step otherwise spends minutes compiling, and with pipeline parallelism that can
trip "MUSA launch timed out" on the stage waiting for it. Run once per node (one
GPU is enough) before launching a new TP / micro-batch configuration:

    MUSA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/fla_pkg:/workspace/Megatron-LM-FL \
        python examples/qwen35/warmup_fla_musa.py <tp_size> [micro_batch_size] \
        [--model qwen36-35b-a3b] [--varlen-docs 5]

--varlen-docs N also compiles the packed-sequence (cu_seqlens) variants used with
reset_attention_mask, with N documents per sample.

If a node hangs in autotune (seen in causal_conv1d_fwd do_bench), kill it and copy
a warmed ~/.triton/cache from another node instead.
"""

import argparse
import os
import sys
import time

import torch
import torch_musa  # noqa: F401

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../flagscale/train/megatron")
)
from patches.musa_gdn import _drop_fla_autotune_configs  # noqa: E402
from fla.modules.convolution import causal_conv1d  # noqa: E402
from fla.modules.l2norm import l2norm  # noqa: E402
from fla.ops.common import chunk_o  # noqa: E402
from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: E402

# GDN key / value heads (head dim 128, conv width 4).
MODELS = {"qwen35-27b": (16, 48), "qwen36-35b-a3b": (16, 32)}

p = argparse.ArgumentParser()
p.add_argument("tp", type=int)
p.add_argument("micro_batch_size", type=int, nargs="?", default=2)
p.add_argument("--model", choices=sorted(MODELS), default="qwen35-27b")
p.add_argument("--seq-length", type=int, default=8192)
p.add_argument("--varlen-docs", type=int, default=0)
args = p.parse_args()

# Same filter as the training patch: num_warps=2 aborts the MUSA triton compiler.
_drop_fla_autotune_configs(chunk_o.chunk_bwd_kernel_dqkwg, lambda c: c.num_warps < 4)

D = 128
hk, hv = (n // args.tp for n in MODELS[args.model])
dev = torch.device("musa")


def run(B, S, cu_seqlens):
    x = torch.randn(B, S, 2 * hk * D + hv * D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(x.shape[-1], 4, device=dev, dtype=torch.bfloat16, requires_grad=True)
    y, _ = causal_conv1d(x=x, weight=w, bias=None, activation="silu", cu_seqlens=cu_seqlens)
    y.float().sum().backward()
    qk = torch.randn(B, S, 2 * hk, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    l2norm(qk).float().sum().backward()
    q, k = [l2norm(torch.randn(B, S, hv, D, device=dev)).bfloat16().requires_grad_() for _ in range(2)]
    v = torch.randn(B, S, hv, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    g = (-torch.nn.functional.softplus(torch.randn(B, S, hv, device=dev)) * 0.5).requires_grad_()
    beta = torch.rand(B, S, hv, device=dev, dtype=torch.bfloat16, requires_grad=True)
    o, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, cu_seqlens=cu_seqlens)
    o.float().sum().backward()
    torch.musa.synchronize()
    return bool(torch.isfinite(o).all())


t0 = time.time()
ok = run(args.micro_batch_size, args.seq_length, None)
if args.varlen_docs:
    # Packed sample: one row, uneven document boundaries.
    S, n = args.seq_length, args.varlen_docs
    cuts = sorted(torch.randperm(S - 1)[: n - 1].add(1).tolist())
    cu = torch.tensor([0] + cuts + [S], device=dev, dtype=torch.int32)
    ok = run(1, S, cu) and ok
print(
    f"warmup model={args.model} tp={args.tp} B={args.micro_batch_size} S={args.seq_length} "
    f"varlen_docs={args.varlen_docs} done in {time.time() - t0:.0f}s, finite={ok}",
    flush=True,
)
