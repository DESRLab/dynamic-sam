from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ['furthest_point_sample']


@triton.jit
def _furthest_point_sample_kernel(
    xyz_ptr, idx_ptr,
    N,
    stride_xb, stride_xn, stride_xc,
    stride_ib, stride_im,
    BLOCK_N: tl.constexpr,
    M: tl.constexpr,
):
    pid_b = tl.program_id(0)
    xyz_ptr += pid_b * stride_xb
    idx_ptr += pid_b * stride_ib

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    x = tl.load(xyz_ptr + offs_n * stride_xn + 0 * stride_xc, mask=mask_n, other=0.0).to(tl.float32)
    y = tl.load(xyz_ptr + offs_n * stride_xn + 1 * stride_xc, mask=mask_n, other=0.0).to(tl.float32)
    z = tl.load(xyz_ptr + offs_n * stride_xn + 2 * stride_xc, mask=mask_n, other=0.0).to(tl.float32)

    # points within 1e-3 squared distance of the origin are treated as padding
    # and can never be selected, matching pointnet2_ops' CUDA kernel.
    valid = mask_n & ((x * x + y * y + z * z) > 1e-3)
    temp = tl.full((BLOCK_N,), 1e10, dtype=tl.float32)

    old = tl.zeros((), dtype=tl.int32)
    tl.store(idx_ptr, old)

    for j in range(1, M):
        x1 = tl.load(xyz_ptr + old * stride_xn + 0 * stride_xc).to(tl.float32)
        y1 = tl.load(xyz_ptr + old * stride_xn + 1 * stride_xc).to(tl.float32)
        z1 = tl.load(xyz_ptr + old * stride_xn + 2 * stride_xc).to(tl.float32)

        d = (x - x1) * (x - x1) + (y - y1) * (y - y1) + (z - z1) * (z - z1)
        d2 = tl.minimum(d, temp)
        temp = tl.where(valid, d2, temp)

        cand = tl.where(valid, d2, -1.0)
        old = tl.argmax(cand, axis=0).to(tl.int32)
        tl.store(idx_ptr + j * stride_im, old)


def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Iterative furthest point sampling, Triton kernel (no CUDA-toolkit build step).

    Semantics match `pointnet2_ops.pointnet2_utils.furthest_point_sample`: sampling
    always starts from point index 0, and points within 1e-3 squared distance of the
    origin are treated as padding and excluded from selection.

    Args:
        xyz: (B, N, 3) CUDA float tensor of point coordinates.
        npoint: number of points to sample.
    Returns:
        (B, npoint) int64 tensor of sampled point indices.
    """
    assert xyz.is_cuda and xyz.dim() == 3 and xyz.shape[-1] == 3
    xyz = xyz.contiguous()
    B, N, _ = xyz.shape
    idx = torch.empty((B, npoint), dtype=torch.int32, device=xyz.device)
    BLOCK_N = triton.next_power_of_2(N)
    _furthest_point_sample_kernel[(B,)](
        xyz, idx,
        N,
        xyz.stride(0), xyz.stride(1), xyz.stride(2),
        idx.stride(0), idx.stride(1),
        BLOCK_N=BLOCK_N,
        M=npoint,
    )
    return idx.long()
