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

"""Deduplicate FP8 (delayed scaling) amax reductions under activation recompute.

With ``fp8_recipe=delayed`` Megatron wraps a microbatch's decoder forward in one
``fp8_autocast``. TE reduces amaxes (concat of the global buffer + all-reduce +
fused history/scale update over *all* FP8 modules) when the outermost autocast
exits (forward) and in the backward of the first FP8 module (backward). With
``te_checkpoint`` (Megatron block recompute) this TE build instead reduces once
per recompute region and more:

* Forward: ``_CheckpointFunction.backward`` re-enters ``fp8_autocast`` at depth 0,
  so every recompute exit reduces the whole forward buffer again. Stock TE
  depends on this: ``get_old_fp8_meta_tensors_for_recompute`` keeps the
  "updated" history/scale as references to the same tensors it then overwrites
  with the pre-forward stash, so ``restore_fp8_meta_tensors`` is a no-op and
  only the recompute-exit reduction moves the scale forward.
* Backward: the original forward of a checkpoint region runs under
  ``torch.no_grad`` so no module consumes ``IS_FIRST_FP8_MODULE``; every region
  stashes True, and in the recompute phase each module re-arms the flag, so
  every FP8 module of every region triggers a full backward reduction.

Qwen3.5-27B TP4 DP8 (46 recomputed layers, 16 microbatches): ~2350 tiny
cross-node all-reduces and ~250k concat kernels per step (~7 s on MUSA).

This opt-in patch (``FLAGSCALE_FP8_DEDUP_RECOMPUTE_AMAX=1``):

* snapshots the post-reduction history/scale as copies before loading the
  stash for the recompute, restores them afterwards, and skips the
  recompute-exit forward reduction. Forward scales then follow the regular
  one-reduction-per-microbatch schedule (as without recompute); the
  recompute's duplicate amaxes are dropped.
* stashes ``IS_FIRST_FP8_MODULE = True`` only for the first checkpoint region of
  each outer autocast and keeps the restored value on the recompute's autocast
  entry, so backward reductions drop to the FP8 modules of one region per
  microbatch. Grad amaxes of layers whose backward runs after it are folded
  into the next microbatch's reduction (one microbatch of extra delay).
"""

import os

_PATCHED = False


def apply_fp8_recompute_amax_dedup_patch():
    """Install the dedup patch on TE's FP8 state manager. Idempotent."""
    global _PATCHED
    if _PATCHED or os.environ.get("FLAGSCALE_FP8_DEDUP_RECOMPUTE_AMAX", "0") != "1":
        return

    import transformer_engine.pytorch.distributed as te_dist
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager as M

    orig_enter = M.fp8_autocast_enter.__func__
    orig_exit = M.fp8_autocast_exit.__func__

    def fp8_autocast_enter(cls, *args, **kwargs):
        in_recompute = te_dist.in_fp8_activation_recompute_phase()
        restored_first = cls.IS_FIRST_FP8_MODULE
        orig_enter(cls, *args, **kwargs)
        if in_recompute and cls.FP8_AUTOCAST_DEPTH == 1:
            # activation_recompute_forward (entered just before this autocast
            # in _CheckpointFunction.backward) restored the stashed flag.
            cls.IS_FIRST_FP8_MODULE = restored_first

    def fp8_autocast_exit(cls, enabled, _graph):
        if enabled and cls.FP8_AUTOCAST_DEPTH == 1 and te_dist.in_fp8_activation_recompute_phase():
            # Recompute of a checkpoint region: skip the duplicate reduction.
            cls.FP8_AUTOCAST_DEPTH -= 1
            return
        orig_exit(cls, enabled, _graph)

    def get_old_fp8_meta_tensors_for_recompute(cls, fp8_meta):
        if fp8_meta["recipe"].mxfp8():
            return
        scaling = fp8_meta["scaling_fwd"]
        # Copies, not references: the stash is loaded into these same tensors.
        fp8_meta["updated_amax_history_fwd"] = scaling.amax_history.clone()
        fp8_meta["updated_scale_fwd"] = scaling.scale.clone()
        stashed = cls.fp8_tensors_recompute_buffer[
            fp8_meta["global_fp8_buffer_pos_fwd_recompute"]
        ].popleft()
        scaling.amax_history.copy_(stashed[0])
        scaling.scale.copy_(stashed[1])

    M.fp8_autocast_enter = classmethod(fp8_autocast_enter)
    M.fp8_autocast_exit = classmethod(fp8_autocast_exit)
    M.get_old_fp8_meta_tensors_for_recompute = classmethod(get_old_fp8_meta_tensors_for_recompute)
    # restore_fp8_meta_tensors already copies updated_* back; with real copies
    # above it now restores the post-reduction state.

    rf = te_dist.activation_recompute_forward
    orig_rf_enter = rf.__enter__

    def rf_enter(self):
        orig_rf_enter(self)
        if self.activation_recompute and not self.recompute_phase and M.is_fp8_enabled():
            # Only the first region per outer autocast keeps True on the stash.
            M.IS_FIRST_FP8_MODULE = False

    rf.__enter__ = rf_enter
    _PATCHED = True
    print("[FlagScale] FP8: deduplicated amax reductions under activation recompute.")
