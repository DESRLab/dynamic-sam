"""Parity tests for src/kernel/quantization.py against the reference semantics
of MinkowskiEngine.utils.sparse_quantize(..., return_index=True, return_inverse=False)
(MinkowskiEngine/MinkowskiEngine/utils/quantization.py): discretize by
floor(coord / quantization_size), then keep, for each occupied voxel, the
first point (in original input order) that lands in it.

The reference below is a plain Python port of that semantic, not an import
of MinkowskiEngine (which needs a CUDA-toolkit build step we are trying to
avoid), so it acts as an independent correctness oracle for the Triton path.
"""
import pytest
import torch

from src.kernel.quantization import sparse_quantize

CUDA = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not CUDA, reason="Triton kernel requires a CUDA device")


def reference_sparse_quantize(coordinates: torch.Tensor, quantization_size: float) -> torch.Tensor:
    discrete = torch.floor(coordinates / quantization_size).long()
    first_seen = {}
    for i in range(discrete.shape[0]):
        key = tuple(discrete[i].tolist())
        if key not in first_seen:
            first_seen[key] = i
    return torch.tensor(sorted(first_seen.values()), dtype=torch.long)


def assert_same_index_set(got: torch.Tensor, ref: torch.Tensor):
    got_sorted = torch.sort(got.cpu())[0]
    ref_sorted = torch.sort(ref)[0]
    assert torch.equal(got_sorted, ref_sorted), f"got={got_sorted.tolist()}\nref={ref_sorted.tolist()}"


@skip_no_cuda
@pytest.mark.parametrize("N,qs,seed", [(50, 0.1, 0), (500, 0.05, 1), (2000, 0.2, 2)])
def test_sparse_quantize_matches_reference(N, qs, seed):
    g = torch.Generator().manual_seed(seed)
    coords = (torch.rand(N, 3, generator=g) * 4 - 2).cuda()  # includes negative coords
    got = sparse_quantize(coords, qs)
    ref = reference_sparse_quantize(coords.cpu(), qs)
    assert_same_index_set(got, ref)


@skip_no_cuda
def test_sparse_quantize_exact_duplicates_keep_first():
    coords = torch.tensor([
        [0.01, 0.01, 0.01],
        [0.02, 0.02, 0.02],  # same voxel as row 0 at qs=0.1
        [5.0, 5.0, 5.0],
        [0.03, 0.03, 0.03],  # same voxel as row 0 again
    ]).cuda()
    idx = sparse_quantize(coords, 0.1)
    idx_list = sorted(idx.cpu().tolist())
    assert idx_list == [0, 2]


@skip_no_cuda
def test_sparse_quantize_one_index_per_occupied_voxel():
    coords = (torch.rand(1000, 3) * 10 - 5).cuda()
    qs = 0.3
    idx = sparse_quantize(coords, qs)
    discrete = torch.floor(coords / qs).long()
    n_unique_voxels = torch.unique(discrete, dim=0).shape[0]
    assert idx.numel() == n_unique_voxels
    assert idx.numel() == torch.unique(idx).numel()  # no duplicate indices returned


@skip_no_cuda
def test_sparse_quantize_downstream_indexing_shape():
    N = 300
    coords = (torch.rand(N, 3) * 3).cuda()
    idx = sparse_quantize(coords, 0.1)
    downsampled = coords[idx]
    assert downsampled.shape[0] == idx.numel()
    assert downsampled.shape[0] <= N
