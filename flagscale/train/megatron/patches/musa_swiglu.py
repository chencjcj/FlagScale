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

"""Fused triton SwiGLU forward/backward for MUSA.

Megatron's ``jit_fuser`` is a no-op off CUDA, so ``swiglu`` / ``swiglu_back`` run
as eager chunk + silu + mul (+ sigmoid, cat in backward), several full passes
over the [tokens, 2 * ffn] activation. The triton build in the MUSA image has a
MUSA driver, so replace both with one read/write pass each. Math is fp32 like
the eager ops' internal compute; outputs keep the input dtype.
"""

import torch
import triton
import triton.language as tl

_PATCHED = False


@triton.jit
def _swiglu_fwd_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    a = tl.load(x_ptr + row * 2 * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + row * 2 * n_cols + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    out = a * tl.sigmoid(a) * b
    tl.store(out_ptr + row * n_cols + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_bwd_kernel(g_ptr, x_ptr, dx_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    g = tl.load(g_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(x_ptr + row * 2 * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + row * 2 * n_cols + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.sigmoid(a)
    da = g * s * (1 + a * (1 - s)) * b
    db = g * a * s
    dt = dx_ptr.dtype.element_ty
    tl.store(dx_ptr + row * 2 * n_cols + cols, da.to(dt), mask=mask)
    tl.store(dx_ptr + row * 2 * n_cols + n_cols + cols, db.to(dt), mask=mask)


_BLOCK = 1024


def triton_swiglu(y):
    """``silu(y[..., :n]) * y[..., n:]`` with n = y.shape[-1] // 2."""
    y = y.contiguous()
    n = y.shape[-1] // 2
    rows = y.numel() // y.shape[-1]
    out = torch.empty(*y.shape[:-1], n, dtype=y.dtype, device=y.device)
    if rows:
        _swiglu_fwd_kernel[(rows, triton.cdiv(n, _BLOCK))](y, out, n, BLOCK=_BLOCK)
    return out


def triton_swiglu_back(g, y):
    """Gradient of ``triton_swiglu`` w.r.t. y, same layout as Megatron's swiglu_back."""
    g, y = g.contiguous(), y.contiguous()
    n = y.shape[-1] // 2
    rows = y.numel() // y.shape[-1]
    dx = torch.empty_like(y)
    if rows:
        _swiglu_bwd_kernel[(rows, triton.cdiv(n, _BLOCK))](g, y, dx, n, BLOCK=_BLOCK)
    return dx


def apply_musa_triton_swiglu_patch():
    """Route Megatron's unclamped, unbiased SwiGLU through the triton kernels.

    ``SwiGLUFunction`` looks up ``swiglu`` / ``swiglu_back`` as module globals at
    call time, so swapping them covers the dense MLP path. Idempotent.
    """
    global _PATCHED
    if _PATCHED:
        return
    import megatron.core.fusions.fused_bias_swiglu as fbs

    fbs.swiglu = triton_swiglu
    fbs.swiglu_back = triton_swiglu_back
    _PATCHED = True
    print("[FlagScale][MUSA] Patched Megatron swiglu/swiglu_back with fused triton kernels.")
