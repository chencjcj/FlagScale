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

"""MUSA fallback for the Gated DeltaNet (GDN) layer used by Qwen3.5.

``megatron/core/ssm/gated_delta_net.py`` imports ``causal_conv1d``, ``l2norm``
and ``chunk_gated_delta_rule`` from ``fla`` (flash-linear-attention). Those are
triton kernels, and the triton shipped in the MUSA image has no MUSA driver
(``RuntimeError: 0 active drivers``), so ``fla`` cannot run on Moore Threads.
Without ``fla`` the GDN constructor raises ``ImportError``.

Megatron already ships a torch-native ``torch_chunk_gated_delta_rule`` (used in
deterministic mode). This module wires it, plus native-torch ``causal_conv1d``
and ``l2norm``, into the GDN module globals so the layer runs on MUSA. Packed
sequences (``cu_seqlens``) are handled by running each segment separately.
"""

import os

import torch
import torch.nn.functional as F

_PATCHED = False


def _native_l2norm(x, dim=-1, eps=1e-6):
    """Match ``fla.modules.l2norm.l2norm``: x * rsqrt(sum(x^2) + eps), fp32 math."""
    x_dtype = x.dtype
    x = x.float()
    y = x * torch.rsqrt(x.pow(2).sum(dim=dim, keepdim=True) + eps)
    return y.to(x_dtype)


def _causal_conv1d_dense(x, weight, bias, activation):
    # x: [b, s, d], weight: [d, w] -> depthwise causal conv over s.
    seq_len, width = x.shape[1], weight.shape[-1]
    out = F.conv1d(
        x.transpose(1, 2),
        weight.unsqueeze(1),
        bias=bias,
        padding=width - 1,
        groups=weight.shape[0],
    )[..., :seq_len]
    if activation in ("silu", "swish"):
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"Unsupported activation for MUSA causal_conv1d: {activation}")
    return out.transpose(1, 2)


def _native_causal_conv1d(
    x,
    weight,
    bias=None,
    residual=None,
    initial_state=None,
    output_final_state=False,
    activation=None,
    cu_seqlens=None,
    **kwargs,
):
    """Match the training-path contract of ``fla.modules.convolution.causal_conv1d``.

    x: [b, s, d] (b == 1 when ``cu_seqlens`` is given), weight: [d, w].
    Returns ``(y, None)``; initial/final states are inference-only and unsupported.
    """
    assert initial_state is None and not output_final_state, (
        "MUSA causal_conv1d fallback does not support conv states"
    )
    if cu_seqlens is None:
        y = _causal_conv1d_dense(x, weight, bias, activation)
    else:
        bounds = cu_seqlens.tolist()
        y = torch.cat(
            [
                _causal_conv1d_dense(x[:, s:e], weight, bias, activation)
                for s, e in zip(bounds[:-1], bounds[1:])
                if e > s
            ],
            dim=1,
        )
    if residual is not None:
        y = y + residual
    return y, None


def _unit_lower_triangular_inverse(m):
    """Inverse of unit lower-triangular ``m`` ([..., c, c], c a power of 2).

    Blocked forward substitution: start from the 1x1 diagonal blocks and merge
    neighbours with inv([[A, 0], [C, D]]) = [[A^-1, 0], [-D^-1 C A^-1, D^-1]].
    Numerically equivalent to row-by-row substitution (unlike the power series
    (I+N)(I+N^2)..., whose intermediates overflow for near-identical keys).
    """
    c = m.shape[-1]
    assert c & (c - 1) == 0, "chunk size must be a power of 2"
    lead = m.shape[:-2]
    inv = torch.ones(*lead, c, 1, 1, dtype=m.dtype, device=m.device)
    s = 1
    while s < c:
        nb = c // (2 * s)
        # Diagonal 2s x 2s blocks of m: [..., nb, 2s, 2s].
        blk = torch.diagonal(m.reshape(*lead, nb, 2 * s, nb, 2 * s), dim1=-4, dim2=-2)
        cmat = blk.movedim(-1, -3)[..., s:, :s]
        pair = inv.reshape(*lead, nb, 2, s, s)
        a_inv, d_inv = pair[..., 0, :, :], pair[..., 1, :, :]
        lower = -(d_inv @ cmat @ a_inv)
        inv = torch.cat(
            [torch.cat([a_inv, torch.zeros_like(a_inv)], -1), torch.cat([lower, d_inv], -1)],
            -2,
        )
        s *= 2
    return inv.reshape(*lead, c, c)


def _fast_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    # 128 halves the sequential inter-chunk loop vs fla's 64: fwd+bwd per layer
    # 91 -> 50 ms on MUSA (mbs2 seq8192 TP4), +0.4 GB peak; 256 gains ~3% more
    # for +0.9 GB. Output matches chunk 64 to ~5e-4 relative.
    chunk_size=128,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Vectorized torch chunked gated delta rule (same math as Megatron's
    ``torch_chunk_gated_delta_rule``).

    The reference solves the per-chunk unit lower-triangular system with a
    63-step Python loop, which launches thousands of tiny kernels per layer.
    Here ``T = (I - N)^-1`` uses a blocked forward substitution in log2(C)
    vectorized steps (see ``_unit_lower_triangular_inverse``). Only the
    inherently sequential inter-chunk state recurrence stays a loop.

    q/k: [b, t, h, dk], v: [b, t, h, dv], g/beta: [b, t, h].
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _native_l2norm(query, dim=-1, eps=1e-6)
        key = _native_l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    b, h, seq_len, dk = key.shape
    dv = value.shape[-1]
    c = chunk_size
    pad = (c - seq_len % c) % c
    if pad:
        query, key, value = [F.pad(x, (0, 0, 0, pad)) for x in (query, key, value)]
        beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    n = (seq_len + pad) // c
    query = query * dk**-0.5

    # [b, h, n, c, d]
    query, key, value = [x.reshape(b, h, n, c, x.shape[-1]) for x in (query, key, value)]
    beta = beta.reshape(b, h, n, c, 1)
    g = g.reshape(b, h, n, c).cumsum(dim=-1)

    k_beta = key * beta
    v_beta = value * beta
    strict_upper = torch.ones(c, c, dtype=torch.bool, device=key.device).triu(1)
    # decay[i, j] = exp(g_i - g_j) for j <= i, else 0 (masked before exp).
    decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).masked_fill(strict_upper, float("-inf")).exp()

    # N is strictly lower triangular; T = (I - N)^-1.
    diag_and_upper = torch.ones(c, c, dtype=torch.bool, device=key.device).triu(0)
    nmat = (-(k_beta @ key.transpose(-1, -2)) * decay).masked_fill(diag_and_upper, 0)
    tmat = _unit_lower_triangular_inverse(
        torch.eye(c, dtype=nmat.dtype, device=nmat.device) - nmat
    )

    w_v = tmat @ v_beta  # [b, h, n, c, dv]
    k_cumdecay = tmat @ (k_beta * g.exp().unsqueeze(-1))  # [b, h, n, c, dk]
    attn_intra = (query @ key.transpose(-1, -2)) * decay  # inclusive causal
    q_decay = query * g.exp().unsqueeze(-1)
    g_last = g[..., -1:]  # [b, h, n, 1]
    k_state = key * (g_last - g).exp().unsqueeze(-1)

    # The inter-chunk recurrence stays a loop: a log-depth affine scan was tried
    # and measured slower end to end (8.5 vs 7.5 s/step, +2 GB) on MUSA, since
    # it materializes [dk, dk] maps per chunk per level.
    state = (
        torch.zeros(b, h, dk, dv, dtype=torch.float32, device=key.device)
        if initial_state is None
        else initial_state.to(torch.float32)
    )
    # unbind (not x[:, :, i]) so autograd builds one stacked grad per input
    # instead of a full-size zero grad per chunk step (~6x faster backward).
    w_v_s, k_cumdecay_s = w_v.unbind(2), k_cumdecay.unbind(2)
    k_state_t = k_state.transpose(-1, -2).unbind(2)
    state_decay = g_last.exp().unsqueeze(-1).unbind(2)
    # Only the state recurrence runs per chunk; the output matmuls don't feed
    # it, so they are batched over all chunks after the loop (layer fwd+bwd
    # 58 -> 38 ms at chunk 128, bit-identical result, +0.5 GB transient).
    states, v_news = [], []
    for i in range(n):
        states.append(state)
        v_new = w_v_s[i] - k_cumdecay_s[i] @ state
        v_news.append(v_new)
        state = state * state_decay[i] + k_state_t[i] @ v_new

    out = q_decay @ torch.stack(states, dim=2) + attn_intra @ torch.stack(v_news, dim=2)
    out = out.reshape(b, h, n * c, dv)[:, :, :seq_len]
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    return out, (state if output_final_state else None)


class _ChunkStateScan(torch.autograd.Function):
    """Inter-chunk state recurrence ``S_{i+1} = M_i @ S_i + B_i``.

    m: [n, N, dk, dk], bmat: [n, N, dk, dv], s0: [N, dk, dv] (fp32).
    Returns all states [n + 1, N, dk, dv]. Forward and backward each issue one
    baddbmm per chunk; autograd over the unrolled loop needs ~4 small matmuls
    plus several elementwise kernels per chunk, all latency bound on MUSA.
    """

    @staticmethod
    def forward(ctx, m, bmat, s0):
        n = m.shape[0]
        states = m.new_empty((n + 1,) + tuple(s0.shape))
        states[0].copy_(s0)
        for i in range(n):
            torch.baddbmm(bmat[i], m[i], states[i], out=states[i + 1])
        ctx.save_for_backward(m, states)
        return states

    @staticmethod
    def backward(ctx, dstates):
        m, states = ctx.saved_tensors
        n = m.shape[0]
        dstates = dstates.contiguous()
        # dbmat[i] = dL/dS_{i+1} (total), accumulated from the last chunk back.
        dbmat = torch.empty_like(states[1:])
        dbmat[n - 1].copy_(dstates[n])
        for i in range(n - 1, 0, -1):
            torch.baddbmm(dstates[i], m[i].transpose(1, 2), dbmat[i], out=dbmat[i - 1])
        ds0 = torch.baddbmm(dstates[0], m[0].transpose(1, 2), dbmat[0])
        # dM_i = dS_{i+1} @ S_i^T, batched over all chunks after the loop.
        dm = dbmat @ states[:n].transpose(-1, -2)
        return dm, dbmat, ds0


def _fast_chunk_gated_delta_rule_v2(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=128,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Same math as ``_fast_chunk_gated_delta_rule`` with two changes:

    * Large intra-chunk matmuls take bf16/fp16 operands (fp32 accumulation), the
      precision split fla uses on CUDA. The triangular inverse, decays and the
      state recurrence stay fp32.
    * The recurrence is folded into ``S_{i+1} = M_i S_i + B_i`` with
      ``M_i = exp(g_last_i) I - K_i^T C_i`` and ``B_i = K_i^T W_i`` computed for all
      chunks at once, so the sequential loop is one baddbmm per chunk
      (``_ChunkStateScan``) and ``v_new`` / the output are batched after it.

    q/k: [b, t, h, dk], v: [b, t, h, dv], g/beta: [b, t, h].
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _native_l2norm(query, dim=-1, eps=1e-6)
        key = _native_l2norm(key, dim=-1, eps=1e-6)
    mm = query.dtype if query.dtype in (torch.bfloat16, torch.float16) else torch.float32
    query, key, value, beta = [
        x.transpose(1, 2).contiguous().to(mm) for x in (query, key, value, beta)
    ]
    g = g.transpose(1, 2).contiguous().to(torch.float32)
    b, h, seq_len, dk = key.shape
    dv = value.shape[-1]
    c = chunk_size
    pad = (c - seq_len % c) % c
    if pad:
        query, key, value = [F.pad(x, (0, 0, 0, pad)) for x in (query, key, value)]
        beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    n = (seq_len + pad) // c
    query = query * dk**-0.5

    # [b, h, n, c, d]
    query, key, value = [x.reshape(b, h, n, c, x.shape[-1]) for x in (query, key, value)]
    beta = beta.reshape(b, h, n, c, 1)
    g = g.reshape(b, h, n, c).cumsum(dim=-1)

    k_beta = key * beta
    v_beta = value * beta
    strict_upper = torch.ones(c, c, dtype=torch.bool, device=key.device).triu(1)
    decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).masked_fill(strict_upper, float("-inf")).exp()

    diag_and_upper = torch.ones(c, c, dtype=torch.bool, device=key.device).triu(0)
    nmat = (-(k_beta @ key.transpose(-1, -2)).float() * decay).masked_fill(diag_and_upper, 0)
    tmat = _unit_lower_triangular_inverse(
        torch.eye(c, dtype=nmat.dtype, device=nmat.device) - nmat
    ).to(mm)

    g_exp = g.exp().unsqueeze(-1)
    w_v = tmat @ v_beta  # [b, h, n, c, dv]
    k_cumdecay = tmat @ (k_beta.float() * g_exp).to(mm)  # [b, h, n, c, dk]
    attn_intra = ((query @ key.transpose(-1, -2)).float() * decay).to(mm)  # inclusive causal
    q_decay = (query.float() * g_exp).to(mm)
    g_last = g[..., -1:]  # [b, h, n, 1]
    k_state_t = (key.float() * (g_last - g).exp().unsqueeze(-1)).transpose(-1, -2)

    eye_dk = torch.eye(dk, dtype=torch.float32, device=key.device)
    m = g_last.exp().unsqueeze(-1) * eye_dk - k_state_t @ k_cumdecay.float()
    bmat = k_state_t @ w_v.float()
    s0 = (
        torch.zeros(b * h, dk, dv, dtype=torch.float32, device=key.device)
        if initial_state is None
        else initial_state.to(torch.float32).reshape(b * h, dk, dv)
    )
    states = _ChunkStateScan.apply(
        m.permute(2, 0, 1, 3, 4).reshape(n, b * h, dk, dk),
        bmat.permute(2, 0, 1, 3, 4).reshape(n, b * h, dk, dv),
        s0,
    )
    # State entering each chunk: [b, h, n, dk, dv].
    s_prev = states[:n].reshape(n, b, h, dk, dv).permute(1, 2, 0, 3, 4).to(mm)
    v_new = (w_v.float() - (k_cumdecay @ s_prev).float()).to(mm)
    out = (q_decay @ s_prev).float() + (attn_intra @ v_new).float()
    out = out.reshape(b, h, n * c, dv)[:, :, :seq_len]
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    final_state = states[n].reshape(b, h, dk, dv) if output_final_state else None
    return out, final_state


def _make_chunk_gated_delta_rule(torch_chunk_gated_delta_rule):
    def _native_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
        **kwargs,
    ):
        """Match ``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` for training.

        q/k: [b, t, h, dk], v: [b, t, h, dv], g/beta: [b, t, h]. The torch reference
        always scales by 1/sqrt(dk), which is fla's default when ``scale`` is None.
        """
        assert scale is None or abs(scale - q.shape[-1] ** -0.5) < 1e-8, (
            "MUSA chunk_gated_delta_rule fallback only supports the default scale"
        )

        def run(q_, k_, v_, g_, beta_, state_):
            return torch_chunk_gated_delta_rule(
                q_,
                k_,
                v_,
                g=g_,
                beta=beta_,
                initial_state=state_,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        if cu_seqlens is None:
            return run(q, k, v, g, beta, initial_state)

        assert initial_state is None, "initial_state with cu_seqlens is unsupported"
        bounds = cu_seqlens.tolist()
        outs, states = [], []
        for s, e in zip(bounds[:-1], bounds[1:]):
            if e <= s:
                continue
            o, st = run(q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e], None)
            outs.append(o)
            states.append(st)
        final = torch.cat(states, dim=0) if output_final_state else None
        return torch.cat(outs, dim=1), final

    return _native_chunk_gated_delta_rule


def apply_musa_te_extra_state_patch():
    """Skip empty ``_extra_state`` tensors when loading TE modules.

    The Qwen3.5 converter writes empty (``numel() == 0``) ``_extra_state`` byte
    tensors. TE's ops-based ``set_extra_state`` ignores those, but the MUSA TE
    build's ``TransformerEngineBaseModule.set_extra_state`` unpickles them and
    fails with ``EOFError: Ran out of input``. Add the same empty-tensor guard.
    """
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

    orig = TransformerEngineBaseModule.set_extra_state
    if getattr(orig, "_flagscale_musa_patched", False):
        return

    def set_extra_state(self, state):
        if isinstance(state, torch.Tensor) and state.numel() == 0:
            return
        return orig(self, state)

    set_extra_state._flagscale_musa_patched = True
    TransformerEngineBaseModule.set_extra_state = set_extra_state
    print("[FlagScale][MUSA] Patched TE set_extra_state to skip empty extra states.")


def apply_musa_te_grouped_linear_patch():
    """Provide ``musa_patch.mem_utils.MemMonitor`` for MUSA TE's GroupedLinear.

    The MUSA TE build's ``_GroupedLinear.forward`` unconditionally does
    ``from musa_patch.mem_utils import MemMonitor`` and bumps
    ``MemMonitor.max_token_num``, so every MoE layer fails with
    ``ModuleNotFoundError`` unless Moore Threads' megatron-lm-musa-patch is on
    the path. Importing that package runs its Megatron monkey patches (written
    for a different Megatron fork), so register a stub with just the counter
    instead when the real package is not importable.
    """
    import importlib.util
    import sys
    import types

    if "musa_patch" in sys.modules or importlib.util.find_spec("musa_patch") is not None:
        return

    class MemMonitor:
        max_token_num = 0

    pkg = types.ModuleType("musa_patch")
    pkg.__path__ = []
    mem_utils = types.ModuleType("musa_patch.mem_utils")
    mem_utils.MemMonitor = MemMonitor
    pkg.mem_utils = mem_utils
    sys.modules["musa_patch"] = pkg
    sys.modules["musa_patch.mem_utils"] = mem_utils
    print("[FlagScale][MUSA] Registered stub musa_patch.mem_utils.MemMonitor for TE GroupedLinear.")


def _drop_fla_autotune_configs(kernel, predicate):
    """Remove autotune configs matching ``predicate`` from a (possibly
    heuristics-wrapped) fla triton kernel."""
    tuner = kernel
    while not hasattr(tuner, "configs"):
        tuner = tuner.fn
    kept = [c for c in tuner.configs if not predicate(c)]
    assert kept, "all autotune configs were filtered out"
    tuner.configs = kept


def _apply_fla_gdn(gdn):
    """Use fla's triton GDN kernels (triton in the MUSA image has a MUSA driver).

    The MUSA triton compiler aborts the process (``emitOffsetForSliceLayout``
    assertion) when compiling ``chunk_bwd_kernel_dqkwg`` with ``num_warps=2``;
    4 and 8 warps compile and match the fp32 torch fallback to ~4e-3.
    """
    from fla.modules.convolution import causal_conv1d
    from fla.modules.l2norm import l2norm
    from fla.ops.common import chunk_o
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    _drop_fla_autotune_configs(chunk_o.chunk_bwd_kernel_dqkwg, lambda c: c.num_warps < 4)
    gdn.causal_conv1d = causal_conv1d
    gdn.l2norm = l2norm
    gdn.chunk_gated_delta_rule = chunk_gated_delta_rule
    gdn.HAVE_FLA = True
    print("[FlagScale][MUSA] Using fla triton kernels for GatedDeltaNet ops (impl=fla).")


def apply_musa_gdn_patch():
    """Install native-torch fla replacements into Megatron's GDN module.

    Idempotent. Must run before the model is built, because ``GatedDeltaNet``
    checks ``HAVE_FLA`` and binds ``chunk_gated_delta_rule`` in ``__init__``.
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        import megatron.core.ssm.gated_delta_net as gdn
    except Exception:
        # GDN module not present in this build; nothing to patch.
        return

    # FLAGSCALE_MUSA_GDN_IMPL selects the GDN ops:
    #   "fla"      fla's triton kernels (needs fla-core importable; ~2x faster
    #              chunk rule, ~4x conv1d/l2norm vs "torch" on MTT S5000)
    #   "torch"    native fp32 fallback (default)
    #   "torch_v2" bf16 matmul operands + single-baddbmm state scan, ~4e-3 rel
    #              diff vs fp32, similar to fla on CUDA
    impl = os.environ.get("FLAGSCALE_MUSA_GDN_IMPL", "torch")
    if impl == "fla":
        _apply_fla_gdn(gdn)
        _PATCHED = True
        return
    chunk_impls = {
        "torch": _fast_chunk_gated_delta_rule,
        "torch_v2": _fast_chunk_gated_delta_rule_v2,
    }
    if impl not in chunk_impls:
        raise ValueError(
            f"FLAGSCALE_MUSA_GDN_IMPL={impl!r}; expected 'fla' or one of {sorted(chunk_impls)}"
        )

    gdn.causal_conv1d = _native_causal_conv1d
    gdn.l2norm = _native_l2norm
    gdn.chunk_gated_delta_rule = _make_chunk_gated_delta_rule(chunk_impls[impl])
    gdn.HAVE_FLA = True
    _PATCHED = True
    print(
        "[FlagScale][MUSA] Patched GatedDeltaNet fla ops (causal_conv1d, l2norm, "
        f"chunk_gated_delta_rule) with native-torch implementations (impl={impl})."
    )
