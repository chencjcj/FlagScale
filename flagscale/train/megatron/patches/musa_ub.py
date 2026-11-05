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

"""Skip userbuffer communicators that the enabled TP-overlap flags never use.

TE's ``initialize_ub`` always builds all ten communicators (qkv/proj/fc1/fc2 x
fprop/dgrad/wgrad), each with a [tokens, hidden] device buffer (~160 MiB at
seq 8192 x mbs 2 x hidden 5120 bf16), even when e.g. RS fprop overlap or bulk
dgrad is disabled. On MUSA, deleting a communicator after init corrupts the
remaining ones (copy_into_buffer: invalid argument), so the unused ones must
not be created at all.

``FLAGSCALE_UB_SKIP`` is a comma-separated list of communicator names, e.g.
``proj_fprop,fc2_fprop,qkv_dgrad,fc1_dgrad`` for
``--disable-tp-comm-overlap-rs --disable-tp-comm-bulk-dgrad``. Every rank must
use the same list (communicator creation is collective). A skipped
communicator is replaced by a placeholder that raises on use, so a wrong list
fails loudly instead of silently.
"""

import os

import torch

_PATCHED = False

# initialize_ub forwards the cfg's num_sm as the constructor's num_comm_sm; this
# value marks a communicator to skip.
_SKIP_SENTINEL = -0x5B1


class _SkippedUserbuffer:
    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        raise RuntimeError(
            f"userbuffer communicator '{self._name}' was skipped via FLAGSCALE_UB_SKIP "
            f"but is used ({attr}); remove it from FLAGSCALE_UB_SKIP"
        )


class _TexProxy:
    """Forwards to transformer_engine_torch, except skipped CommOverlap objects."""

    def __init__(self, tex):
        self._tex = tex

    def __getattr__(self, attr):
        return getattr(self._tex, attr)

    def _make(self, cls_name):
        real = getattr(self._tex, cls_name)

        def ctor(*args, **kwargs):
            if kwargs.get("num_comm_sm") == _SKIP_SENTINEL:
                return _SkippedUserbuffer(cls_name)
            return real(*args, **kwargs)

        return ctor

    @property
    def CommOverlap(self):  # noqa: N802 - mirrors the tex class name
        return self._make("CommOverlap")

    @property
    def CommOverlapP2P(self):  # noqa: N802
        return self._make("CommOverlapP2P")


_SAVED_LN_OUT_FIXED = False


def _ub_buffer_ranges(te_base):
    """[start, end) device address ranges of all live userbuffer buffers."""
    ranges = []
    for ub in (te_base._ub_communicators or {}).values():
        if isinstance(ub, _SkippedUserbuffer):
            continue
        buf = ub.get_buffer(None)
        start = buf.data_ptr()
        ranges.append((start, start + buf.numel() * buf.element_size()))
    return ranges


def apply_musa_ub_saved_ln_out_fix():
    """Stop LayerNormLinear from saving a userbuffer view for backward.

    With ub_overlap_ag (AG fprop overlap), MT-TE's LayerNormLinear writes the
    norm output straight into the shared ``<ub_name>_fprop`` buffer and saves
    that view as ``ln_out`` for the wgrad GEMM. The next layer using the same
    buffer name (every qkv / fc1, and GDN in_proj which also uses "fc1")
    overwrites it before backward, so wgrad is computed from the wrong input
    (2-layer test: dw rel err 1.5; full model: grad norm inf at iter 1). Plain
    Linear copies its input into the buffer and saves the original, so it is
    unaffected. Clone any saved tensor that lives inside a userbuffer.
    """
    global _SAVED_LN_OUT_FIXED
    if _SAVED_LN_OUT_FIXED:
        return
    from transformer_engine.pytorch.module import base as te_base
    from transformer_engine.pytorch.module import layernorm_linear

    orig_prepare = layernorm_linear.prepare_for_saving
    cache = {"comms": None, "ranges": []}

    def prepare_for_saving(*tensors):
        comms = te_base._ub_communicators
        if comms:
            if cache["comms"] is not comms:
                cache["comms"], cache["ranges"] = comms, _ub_buffer_ranges(te_base)
            fixed = []
            for t in tensors:
                if isinstance(t, torch.Tensor) and t.numel() > 0:
                    ptr = t.data_ptr()
                    if any(lo <= ptr < hi for lo, hi in cache["ranges"]):
                        t = t.clone()
                fixed.append(t)
            tensors = fixed
        return orig_prepare(*tensors)

    layernorm_linear.prepare_for_saving = prepare_for_saving

    # Same aliasing on the way back: with ub_bulk_wgrad the column-parallel
    # backward returns dgrad as a view of the shared "<ub_name>_wgrad" buffer.
    # Autograd may accumulate the residual gradient into that view and carry it
    # through the rest of the block, and the next layer's backward with the same
    # ub_name overwrites it (2-block test: every grad of the earlier block wrong,
    # rel err ~1). Return a private copy instead.
    from transformer_engine.pytorch.module import linear

    def wrap_backward(fn_cls):
        orig_backward = fn_cls.backward

        def backward(ctx, *grad_outputs):
            grads = orig_backward(ctx, *grad_outputs)
            comms = te_base._ub_communicators
            if not comms:
                return grads
            if cache["comms"] is not comms:
                cache["comms"], cache["ranges"] = comms, _ub_buffer_ranges(te_base)
            out = []
            for g in grads:
                if isinstance(g, torch.Tensor) and g.numel() > 0:
                    ptr = g.data_ptr()
                    if any(lo <= ptr < hi for lo, hi in cache["ranges"]):
                        g = g.clone()
                out.append(g)
            return tuple(out)

        fn_cls.backward = staticmethod(backward)

    wrap_backward(layernorm_linear._LayerNormLinear)
    wrap_backward(linear._Linear)
    _SAVED_LN_OUT_FIXED = True


def apply_musa_ub_skip_patch():
    global _PATCHED
    skip = [n.strip() for n in os.environ.get("FLAGSCALE_UB_SKIP", "").split(",") if n.strip()]
    if _PATCHED or not skip:
        return
    from transformer_engine.pytorch.module import base as te_base

    orig_initialize_ub = te_base.initialize_ub

    def initialize_ub(*args, ub_cfgs=None, **kwargs):
        cfgs = {k: dict(v) for k, v in (ub_cfgs or {}).items()}
        for name in skip:
            cfgs.setdefault(name, {})["num_sm"] = _SKIP_SENTINEL
        real_tex = te_base.tex
        te_base.tex = _TexProxy(real_tex)
        try:
            orig_initialize_ub(*args, ub_cfgs=cfgs, **kwargs)
        finally:
            te_base.tex = real_tex
        for name in skip:
            ub = te_base._ub_communicators.get(name)
            if isinstance(ub, _SkippedUserbuffer):
                ub._name = name
            else:
                raise RuntimeError(f"FLAGSCALE_UB_SKIP: unknown userbuffer name '{name}'")
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[FlagScale] skipped userbuffer communicators: {skip}", flush=True)

    te_base.initialize_ub = initialize_ub
    _PATCHED = True
