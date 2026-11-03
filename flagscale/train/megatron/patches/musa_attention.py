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

"""MUSA fallback for attention with head_dim > 192 (Qwen3.5 uses 256).

The MUSA TE build hardcodes ``use_flash_attention = True`` in
``DotProductAttention.forward`` (ignoring ``NVTE_FLASH_ATTN`` and Megatron's
``attention_backend``). The MUSA ``flash_attn`` package is an SDPA wrapper that
only enables the flash kernel and asserts ``64 <= head_dim <= 192``. Plain
``torch.nn.functional.scaled_dot_product_attention`` handles larger head dims on
MUSA, so route those calls there and keep the original path otherwise.

Its ``flash_attn_varlen_func`` is also wrong for more than one packed segment
(NaN for non-causal, assertion for causal), so multi-segment varlen calls always
go through per-segment SDPA.
"""

import torch
import torch.nn.functional as F

_MAX_MUSA_FLASH_HEAD_DIM = 192
_PATCHED = False
_CAUSAL_MASKS = {}


def _additive_causal_mask(seq_len, dtype, device):
    """Cached [s, s] additive causal mask (0 on/below the diagonal, -inf above).

    The MUSA math SDPA builds a bool tril mask on every ``is_causal=True`` call;
    at s=8192 that alone is ~4.5 ms of the 12.7 ms forward. Passing the same mask
    precomputed gives bit-identical outputs and gradients.
    """
    key = (seq_len, dtype, device)
    mask = _CAUSAL_MASKS.get(key)
    if mask is None:
        mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
        mask.triu_(1)
        _CAUSAL_MASKS[key] = mask
    return mask


def _sdpa_bshd(q, k, v, dropout_p, softmax_scale, causal):
    # q: [b, sq, hq, d], k/v: [b, sk, hk, d] -> [b, sq, hq, d]
    hq, hk = q.shape[2], k.shape[2]
    if hq != hk:
        k = k.repeat_interleave(hq // hk, dim=2)
        v = v.repeat_interleave(hq // hk, dim=2)
    assert not causal or q.shape[1] == k.shape[1], (
        "MUSA SDPA fallback only supports causal attention with seqlen_q == seqlen_kv"
    )
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=_additive_causal_mask(q.shape[1], q.dtype, q.device) if causal else None,
        dropout_p=dropout_p,
        scale=softmax_scale,
    )
    # TE calls .view() on the result, so it must be contiguous.
    return out.transpose(1, 2).contiguous()


def _check_unsupported(window_size, alibi_slopes, return_attn_probs, kwargs):
    # TE normalizes causal masks to (-1, 0); both mean "no sliding window".
    assert window_size in (None, (-1, -1), (-1, 0)), (
        "sliding window unsupported in MUSA SDPA fallback"
    )
    assert alibi_slopes is None, "alibi unsupported in MUSA SDPA fallback"
    assert not return_attn_probs, "return_attn_probs unsupported in MUSA SDPA fallback"
    assert kwargs.get("block_table") is None, "block_table unsupported in MUSA SDPA fallback"
    assert not kwargs.get("softcap"), "softcap unsupported in MUSA SDPA fallback"


def _wrap_flash_attn_func(orig):
    def flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        **kwargs,
    ):
        if q.shape[-1] <= _MAX_MUSA_FLASH_HEAD_DIM:
            return orig(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                **kwargs,
            )
        _check_unsupported(window_size, alibi_slopes, return_attn_probs, kwargs)
        return _sdpa_bshd(q, k, v, dropout_p, softmax_scale, causal)

    return flash_attn_func


def _wrap_flash_attn_varlen_func(orig):
    def flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        **kwargs,
    ):
        # MUSA's native varlen kernel is only correct for a single segment: with
        # more than one (ViT with several images, padded LM batch > 1) it returns
        # NaN (non-causal) or asserts (causal). Use per-segment SDPA instead.
        if q.shape[-1] <= _MAX_MUSA_FLASH_HEAD_DIM and cu_seqlens_q.numel() <= 2:
            return orig(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                **kwargs,
            )
        _check_unsupported(window_size, alibi_slopes, return_attn_probs, kwargs)
        # q: [total_q, hq, d]; run each packed segment separately. TE may pass
        # more rows than cu_seqlens covers (unpadded batch > 1), so zero-init:
        # leftover rows from empty_like can hold NaN and poison the gradients.
        out = torch.zeros_like(q)
        bq, bk = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
        for qs, qe, ks, ke in zip(bq[:-1], bq[1:], bk[:-1], bk[1:]):
            if qe <= qs:
                continue
            out[qs:qe] = _sdpa_bshd(
                q[qs:qe].unsqueeze(0),
                k[ks:ke].unsqueeze(0),
                v[ks:ke].unsqueeze(0),
                dropout_p,
                softmax_scale,
                causal,
            ).squeeze(0)
        return out

    return flash_attn_varlen_func


def apply_musa_large_head_dim_attn_patch():
    """Wrap the flash_attn functions TE's FlashAttention calls. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    import transformer_engine.pytorch.attention as te_attn

    if te_attn.flash_attn_func is not None:
        te_attn.flash_attn_func = _wrap_flash_attn_func(te_attn.flash_attn_func)
    if te_attn.flash_attn_varlen_func is not None:
        te_attn.flash_attn_varlen_func = _wrap_flash_attn_varlen_func(
            te_attn.flash_attn_varlen_func
        )
    _PATCHED = True
    print(
        "[FlagScale][MUSA] Patched TE flash_attn_func/flash_attn_varlen_func to use "
        f"SDPA for head_dim > {_MAX_MUSA_FLASH_HEAD_DIM} and multi-segment varlen."
    )
