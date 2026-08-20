from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import triton
import triton.language as tl

__all__ = ['KNN']


@triton.jit
def _knn_kernel(
    ref_ptr, query_ptr, idx_ptr, dist_ptr,
    N,
    stride_rb, stride_rn, stride_rc,
    stride_qb, stride_qg, stride_qc,
    stride_ib, stride_ig, stride_ik,
    stride_db, stride_dg, stride_dk,
    BLOCK_N: tl.constexpr,
    K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    ref_ptr += pid_b * stride_rb
    query_ptr += pid_b * stride_qb + pid_g * stride_qg
    idx_ptr += pid_b * stride_ib + pid_g * stride_ig
    dist_ptr += pid_b * stride_db + pid_g * stride_dg

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    x = tl.load(ref_ptr + offs_n * stride_rn + 0 * stride_rc, mask=mask_n, other=0.0).to(tl.float32)
    y = tl.load(ref_ptr + offs_n * stride_rn + 1 * stride_rc, mask=mask_n, other=0.0).to(tl.float32)
    z = tl.load(ref_ptr + offs_n * stride_rn + 2 * stride_rc, mask=mask_n, other=0.0).to(tl.float32)

    qx = tl.load(query_ptr + 0 * stride_qc).to(tl.float32)
    qy = tl.load(query_ptr + 1 * stride_qc).to(tl.float32)
    qz = tl.load(query_ptr + 2 * stride_qc).to(tl.float32)

    sqdist = (x - qx) * (x - qx) + (y - qy) * (y - qy) + (z - qz) * (z - qz)
    sqdist = tl.where(mask_n, sqdist, float('inf'))

    for j in range(K):
        best_val = tl.min(sqdist, axis=0)
        best_idx = tl.argmin(sqdist, axis=0).to(tl.int32)
        tl.store(idx_ptr + j * stride_ik, best_idx)
        tl.store(dist_ptr + j * stride_dk, tl.sqrt(best_val))
        sqdist = tl.where(offs_n == best_idx, float('inf'), sqdist)


def _knn_forward(ref: torch.Tensor, query: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Brute-force k-nearest-neighbor search (transpose_mode=True layout).

    For each query point, finds the `k` nearest points in `ref` by Euclidean
    distance, ascending. Equivalent to the only mode this project uses from
    `knn_cuda.KNN(k, transpose_mode=True)`: `dist, idx = knn(ref, query)`.

    Args:
        ref: (B, N, 3) reference/support points.
        query: (B, G, 3) query points.
        k: number of neighbors per query point.
    Returns:
        dist: (B, G, k) Euclidean distances, ascending (matches
            `torch.cdist(query, ref, p=2)` gathered at `idx`).
        idx: (B, G, k) int64 indices into `ref`'s point dimension.
    """
    assert ref.is_cuda and query.is_cuda
    assert ref.dim() == 3 and ref.shape[-1] == 3
    assert query.dim() == 3 and query.shape[-1] == 3
    assert ref.shape[0] == query.shape[0]

    ref = ref.contiguous()
    query = query.contiguous()
    B, N, _ = ref.shape
    _, G, _ = query.shape

    idx = torch.empty((B, G, k), dtype=torch.int32, device=ref.device)
    dist = torch.empty((B, G, k), dtype=torch.float32, device=ref.device)
    BLOCK_N = triton.next_power_of_2(N)
    grid = (B, G)
    _knn_kernel[grid](
        ref, query, idx, dist,
        N,
        ref.stride(0), ref.stride(1), ref.stride(2),
        query.stride(0), query.stride(1), query.stride(2),
        idx.stride(0), idx.stride(1), idx.stride(2),
        dist.stride(0), dist.stride(1), dist.stride(2),
        BLOCK_N=BLOCK_N,
        K=k,
    )
    return dist, idx.long()


class KNN(nn.Module):
    """Drop-in replacement for `knn_cuda.KNN` (transpose_mode=True only, the
    only mode this project uses), backed by a Triton kernel instead of a
    precompiled CUDA extension.
    """

    def __init__(self, k: int, transpose_mode: bool = True):
        super().__init__()
        assert transpose_mode, "src.kernel.knn.KNN only implements transpose_mode=True"
        self.k = k

    def forward(self, ref: torch.Tensor, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _knn_forward(ref, query, self.k)
