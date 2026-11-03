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

"""MUSA fix for Transformer-Engine's LayerNorm op.

The MUSA TE backend (``transformer_engine/musa/pytorch/csrc/extensions/
normalization.cpp``) ships ``layernorm_fwd``/``layernorm_bwd`` as *stubs*: the
real ``nvte_layernorm_fwd``/``nvte_layernorm_bwd`` kernel calls are commented
out, so ``workspace`` stays default-constructed (``ndim == 0``) and the very
first ``allocateSpace(workspace.shape(), ...)`` trips

    RuntimeError: .../common.cpp:208 in function allocateSpace:
    Assertion failed: false. Should never reach here! func: allocateSpace

RMSNorm has a real MUSA implementation (``at::musa::FusedRMSNormForwardOut``),
so the main model norms (``normalization: RMSNorm``) work fine. The failing
path is the GLM5/DeepSeek-V3.2 DSA indexer ``k_norm``, which Megatron hardcodes
to ``LayerNorm`` regardless of ``config.normalization`` (see
``megatron/core/transformer/experimental_attention_variant/dsa.py``).

``torch.nn.functional.layer_norm`` is available and correct on MUSA, so we
replace the two broken bindings with native-torch equivalents that match the
exact calling contract of TE's ops-based ``LayerNorm`` (see
``transformer_engine/pytorch/ops/basic/layer_norm.py``):

    y, means, rstdevs = layernorm_fwd(x, w, b, eps, ln_out, quantizer,
                                      out_dtype, sm_margin, zero_centered_gamma)
    dx, dw, db        = layernorm_bwd(dy, x, means, rstdevs, w, sm_margin,
                                      zero_centered_gamma)

Only the no-FP8 (quantizer is None) case is supported, which is what GLM5
training uses. FP8 LayerNorm on MUSA would need a real kernel.
"""

import torch

_PATCHED = False


def _native_layernorm_fwd(
    x,
    weight,
    bias,
    eps,
    ln_out,
    quantizer,
    out_dtype,
    sm_margin,  # noqa: ARG001 - unused, kept for signature parity
    zero_centered_gamma,
):
    """Native-torch replacement for TE's ``layernorm_fwd``.

    Returns ``(y, means, rstdevs)`` where ``means``/``rstdevs`` are the per-row
    statistics saved for the backward pass (fp32, shape ``[N]``).
    """
    if quantizer is not None:
        raise NotImplementedError(
            "MUSA native LayerNorm patch does not support FP8 output "
            "(quantizer must be None)."
        )

    inp_dtype = x.dtype
    x_fp32 = x.float()
    inner = x_fp32.shape[-1]

    # Per-row mean / variance over the last (normalized) dimension.
    means = x_fp32.mean(dim=-1)
    var = x_fp32.var(dim=-1, unbiased=False)
    rstdevs = torch.rsqrt(var + eps)

    x_hat = (x_fp32 - means.unsqueeze(-1)) * rstdevs.unsqueeze(-1)

    w = weight.float()
    if zero_centered_gamma:
        w = 1.0 + w
    y = x_hat * w.unsqueeze(0) + bias.float().unsqueeze(0)

    y = y.to(inp_dtype)
    if ln_out is not None:
        ln_out.copy_(y)
        y = ln_out

    return y, means, rstdevs


def _native_layernorm_bwd(
    dy,
    x,
    means,
    rstdevs,
    weight,
    sm_margin,  # noqa: ARG001 - unused, kept for signature parity
    zero_centered_gamma,
):
    """Native-torch replacement for TE's ``layernorm_bwd``.

    Returns ``(dx, dw, db)`` matching the forward's input/weight/bias shapes.
    """
    inp_dtype = x.dtype
    x_fp32 = x.float()
    dy_fp32 = dy.float()
    inner = x_fp32.shape[-1]

    means = means.unsqueeze(-1)
    rstdevs = rstdevs.unsqueeze(-1)
    x_hat = (x_fp32 - means) * rstdevs

    w = weight.float()
    if zero_centered_gamma:
        w = 1.0 + w

    # Gradients wrt affine params.
    dw = (dy_fp32 * x_hat).sum(dim=0)
    db = dy_fp32.sum(dim=0)

    # Gradient wrt input (standard LayerNorm backward).
    dy_w = dy_fp32 * w.unsqueeze(0)
    mean_dy_w = dy_w.mean(dim=-1, keepdim=True)
    mean_dy_w_xhat = (dy_w * x_hat).mean(dim=-1, keepdim=True)
    dx = rstdevs * (dy_w - mean_dy_w - x_hat * mean_dy_w_xhat)

    return dx.to(inp_dtype), dw.to(weight.dtype), db.to(weight.dtype)


def _native_hadamard_transform(x, scale=1.0):
    """Native-torch fast Walsh-Hadamard transform along the last dim.

    Drop-in replacement for ``fast_hadamard_transform.hadamard_transform`` (the
    CUDA-only Dao-AILab kernel), which is unavailable on MUSA. It returns
    ``scale * (H @ x)`` using the unnormalized Sylvester Hadamard matrix ``H``,
    matching the kernel's convention. The last dim must be a power of two
    (DSA uses ``dsa_indexer_head_dim`` = 128).

    Computation is done in fp32 for stability and cast back to the input dtype
    (DSA calls this on bf16 activations).
    """
    orig_dtype = x.dtype
    n = x.shape[-1]
    assert (n & (n - 1)) == 0, (
        f"hadamard transform requires a power-of-two last dim, got {n}"
    )
    y = x.float().contiguous()
    h = 1
    while h < n:
        y = y.view(*y.shape[:-1], n // (2 * h), 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = torch.stack([a + b, a - b], dim=-2)
        y = y.reshape(*y.shape[:-3], n)
        h *= 2
    y = y * scale
    return y.to(orig_dtype)


def apply_musa_hadamard_patch():
    """Provide a native Hadamard transform for the DSA indexer on MUSA.

    The DSA indexer (``experimental_attention_variant/dsa.py``) imports
    ``hadamard_transform`` from the CUDA-only ``fast_hadamard_transform``
    package; on MUSA the import fails and the symbol is ``None``, so
    ``rotate_activation`` asserts. Install a native-torch implementation.
    """
    try:
        import megatron.core.transformer.experimental_attention_variant.dsa as dsa
    except Exception:
        # DSA module not present in this build; nothing to patch.
        return

    if getattr(dsa, "hadamard_transform", None) is None:
        dsa.hadamard_transform = _native_hadamard_transform
        print(
            "[FlagScale][MUSA] Patched DSA indexer hadamard_transform with a "
            "native-torch fast Walsh-Hadamard transform."
        )


def apply_musa_te_layernorm_patch():
    """Replace TE's broken MUSA ``layernorm_fwd``/``layernorm_bwd`` bindings.

    Idempotent and safe to call unconditionally; only patches the ops module
    that TE's ``LayerNorm`` (both the ops class and the module subclass) calls
    at runtime.
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        import transformer_engine.pytorch.ops.basic.layer_norm as te_ln
    except Exception as exc:  # pragma: no cover - TE always present on MUSA
        raise RuntimeError(
            "Failed to import transformer_engine ops LayerNorm module for "
            "MUSA patching."
        ) from exc

    te_ln.layernorm_fwd = _native_layernorm_fwd
    te_ln.layernorm_bwd = _native_layernorm_bwd

    # The fused modules (LayerNormLinear / LayerNormMLP, e.g. the Qwen3-VL ViT)
    # look the bindings up at call time on ``transformer_engine_torch`` (as
    # ``tex``) and on ``cpp_extensions`` (via ``_get_normalization_func``).
    import transformer_engine.pytorch.cpp_extensions as te_cpp
    import transformer_engine_torch as te_torch

    for mod in (te_torch, te_cpp):
        mod.layernorm_fwd = _native_layernorm_fwd
        mod.layernorm_bwd = _native_layernorm_bwd
    _PATCHED = True
    print(
        "[FlagScale][MUSA] Patched transformer_engine LayerNorm "
        "(layernorm_fwd/layernorm_bwd) with native-torch implementations."
    )


def _shift_gamma(weight):
    # zero-centered gamma: effective weight is (1 + w), computed in fp32.
    return (weight.float() + 1.0).to(weight.dtype)


def apply_musa_te_rmsnorm_zero_centered_patch():
    """Make MUSA TE RMSNorm honor ``zero_centered_gamma``.

    The MUSA ``rmsnorm_fwd``/``rmsnorm_bwd`` kernels ignore the
    ``zero_centered_gamma`` flag and scale by ``w`` instead of ``1 + w``.
    Qwen3.5 sets ``layernorm_zero_centered_gamma: True``, so every norm output
    collapses (initial loss ~ln(vocab)). Call the kernels with ``1 + w`` and the
    flag off instead; ``dgamma`` is unchanged because d(1 + w)/dw = 1.
    Idempotent.
    """
    import transformer_engine.pytorch.cpp_extensions as te_cpp
    import transformer_engine.pytorch.ops.basic.rmsnorm as te_rms
    import transformer_engine_torch as te_torch

    if getattr(te_torch.rmsnorm_fwd, "_flagscale_musa_patched", False):
        return

    orig_fwd, orig_bwd = te_torch.rmsnorm_fwd, te_torch.rmsnorm_bwd

    def rmsnorm_fwd(x, weight, eps, ln_out, quantizer, out_dtype, sm_margin, zero_centered_gamma):
        if zero_centered_gamma:
            weight = _shift_gamma(weight)
        return orig_fwd(x, weight, eps, ln_out, quantizer, out_dtype, sm_margin, False)

    def rmsnorm_bwd(dy, x, rsigma, weight, sm_margin, zero_centered_gamma):
        if zero_centered_gamma:
            weight = _shift_gamma(weight)
        return orig_bwd(dy, x, rsigma, weight, sm_margin, False)

    rmsnorm_fwd._flagscale_musa_patched = True
    for mod in (te_torch, te_cpp, te_rms):
        mod.rmsnorm_fwd = rmsnorm_fwd
        mod.rmsnorm_bwd = rmsnorm_bwd
    print("[FlagScale][MUSA] Patched TE RMSNorm to honor zero_centered_gamma.")
