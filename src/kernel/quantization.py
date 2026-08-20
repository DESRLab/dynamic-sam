from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ['sparse_quantize']


@triton.jit
def _ravel_voxel_key_kernel(
    discrete_ptr, key_ptr,
    N,
    min_x, min_y, min_z,
    range_y, range_z,
    stride_n, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    ix = tl.load(discrete_ptr + offs * stride_n + 0 * stride_c, mask=mask, other=0).to(tl.int64)
    iy = tl.load(discrete_ptr + offs * stride_n + 1 * stride_c, mask=mask, other=0).to(tl.int64)
    iz = tl.load(discrete_ptr + offs * stride_n + 2 * stride_c, mask=mask, other=0).to(tl.int64)

    # collision-free ravel index over the discrete coordinates' own bounding box,
    # equivalent to MinkowskiEngine.utils.quantization.ravel_hash_vec
    key = (ix - min_x) * range_y * range_z + (iy - min_y) * range_z + (iz - min_z)
    tl.store(key_ptr + offs, key, mask=mask)


def sparse_quantize(coordinates: torch.Tensor, quantization_size: float) -> torch.Tensor:
    """GPU voxel-grid deduplication, equivalent to calling

        MinkowskiEngine.utils.sparse_quantize(
            coordinates=coordinates, quantization_size=quantization_size,
            return_index=True, return_inverse=False)

    and keeping only the returned index tensor (the second element of that
    call's return tuple), which is how this project's only call site
    (`voxelize_pcd` in src/datasets/utils/preprocess.py) uses it.

    Per-point discretization (`floor(coord / quantization_size)`) and the
    ravel-to-int64 voxel key are computed with a Triton kernel. Deduplication
    uses a stable sort: MinkowskiEngine builds its unique map by inserting
    points into a hash map in input order and keeping the first insertion per
    key, and a stable sort preserves the original relative order of points
    that land in the same voxel, so the first element of each equal-key run
    after sorting is exactly that same first-occurrence point.

    Args:
        coordinates: (N, 3) CUDA float tensor of point coordinates.
        quantization_size: voxel side length (scalar).
    Returns:
        1D int64 tensor of indices into `coordinates`, one per occupied voxel.
    """
    assert coordinates.is_cuda and coordinates.dim() == 2 and coordinates.shape[1] == 3
    N = coordinates.shape[0]

    discrete = torch.floor(coordinates / quantization_size).long().contiguous()
    mins = discrete.amin(dim=0)
    ranges = discrete.amax(dim=0) - mins + 1

    keys = torch.empty(N, dtype=torch.int64, device=coordinates.device)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _ravel_voxel_key_kernel[grid](
        discrete, keys,
        N,
        mins[0].item(), mins[1].item(), mins[2].item(),
        ranges[1].item(), ranges[2].item(),
        discrete.stride(0), discrete.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    order = torch.arange(N, device=coordinates.device)
    sorted_keys, sort_idx = torch.sort(keys, stable=True)
    sorted_orig_idx = order[sort_idx]

    is_first = torch.ones(N, dtype=torch.bool, device=coordinates.device)
    is_first[1:] = sorted_keys[1:] != sorted_keys[:-1]

    return sorted_orig_idx[is_first]
