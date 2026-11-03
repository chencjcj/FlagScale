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

"""Uneven reduce-scatter for the flagcx backend on MUSA.

With expert tensor parallelism > 1, the MoE alltoall dispatcher combines tokens
with ``reduce_scatter_to_sequence_parallel_region(..., input_split_sizes=...)``
(and the uneven all-gather in dispatch has the same op in its backward).
Megatron implements that with the list form of ``torch.distributed.reduce_scatter``,
which flagcx rejects: "flagcx only support same size reducescatter operation".

Pad every split to the largest one, run the equal-size reduce-scatter, and trim.
The zero padding adds nothing to the sums, so the result is identical.
"""

import torch

_PATCHED = False


def _make_uneven_reduce_scatter(orig):
    def _reduce_scatter_along_first_dim(
        input_, group, input_split_sizes=None, use_global_buffer=False
    ):
        if input_split_sizes is None or group is None or group.size() == 1:
            return orig(input_, group, input_split_sizes, use_global_buffer)
        world_size, rank = group.size(), group.rank()
        max_split = max(input_split_sizes)
        tail = input_.shape[1:]
        padded = input_.new_zeros((world_size * max_split,) + tuple(tail))
        offset = 0
        for r, size in enumerate(input_split_sizes):
            if size:
                padded[r * max_split : r * max_split + size].copy_(input_[offset : offset + size])
            offset += size
        output = input_.new_empty((max_split,) + tuple(tail))
        torch.distributed.reduce_scatter_tensor(output, padded, group=group)
        return output[: input_split_sizes[rank]]

    return _reduce_scatter_along_first_dim


def apply_musa_uneven_reduce_scatter_patch():
    """Replace Megatron's uneven-split reduce-scatter with a padded equal-size one.

    ``_ReduceScatterToSequenceParallelRegion`` and the gather's backward look up
    ``_reduce_scatter_along_first_dim`` as a module global, so swapping it covers
    both. The equal-size path is untouched. Idempotent.
    """
    global _PATCHED
    if _PATCHED:
        return
    import megatron.core.tensor_parallel.mappings as mappings

    mappings._reduce_scatter_along_first_dim = _make_uneven_reduce_scatter(
        mappings._reduce_scatter_along_first_dim
    )
    _PATCHED = True
    print("[FlagScale][MUSA] Patched uneven reduce-scatter (pad to equal size) for flagcx.")
