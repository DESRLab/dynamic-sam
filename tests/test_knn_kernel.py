"""Parity tests for src/kernel/knn.py against native PyTorch brute-force KNN
(torch.cdist + torch.topk).

The original dependency, knn_cuda (https://github.com/unlimblue/KNN_CUDA), is
no longer obtainable: its GitHub repo has been deleted, it was never
published to PyPI, and no cached copy exists on this machine. Since the only
call site (src/models/pointcloud_encoder.py) discards the `dist` output and
only uses `idx`, the correctness target is unambiguous regardless: for each
query point, the indices of the k nearest reference points by Euclidean
distance, ascending. torch.cdist + torch.topk(largest=False) is PyTorch's own
native, actively-maintained implementation of exactly that, so it serves as
the reference oracle here.
"""
import pytest
import torch

from src.kernel.knn import KNN, _knn_forward

CUDA = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not CUDA, reason="Triton kernel requires a CUDA device")


def reference_knn(ref: torch.Tensor, query: torch.Tensor, k: int):
    d = torch.cdist(query, ref, p=2)  # (B, G, N)
    dist, idx = torch.topk(d, k, dim=-1, largest=False, sorted=True)
    return dist, idx


@skip_no_cuda
@pytest.mark.parametrize(
    "B,N,G,k,seed",
    [
        (1, 20, 5, 3, 0),
        (2, 100, 16, 8, 1),
        (3, 256, 32, 10, 2),
        (2, 4096, 128, 32, 3),  # matches the ScanNet training config scale
        (1, 10, 1, 10, 4),      # k == N edge case
        (4, 50, 1, 5, 5),       # single query point
    ],
)
def test_knn_matches_native_pytorch(B, N, G, k, seed):
    g = torch.Generator().manual_seed(seed)
    ref = torch.randn(B, N, 3, generator=g).cuda()
    query = torch.randn(B, G, 3, generator=g).cuda()

    got_dist, got_idx = _knn_forward(ref, query, k)
    ref_dist, ref_idx = reference_knn(ref, query, k)

    assert got_idx.shape == (B, G, k)
    assert got_dist.shape == (B, G, k)
    assert torch.equal(got_idx.cpu(), ref_idx.cpu())
    assert torch.allclose(got_dist.cpu(), ref_dist.cpu(), atol=1e-4)


@skip_no_cuda
def test_knn_module_matches_native_pytorch():
    B, N, G, k = 2, 200, 20, 12
    ref = torch.randn(B, N, 3).cuda()
    query = torch.randn(B, G, 3).cuda()

    knn = KNN(k=k, transpose_mode=True)
    dist, idx = knn(ref, query)
    ref_dist, ref_idx = reference_knn(ref, query, k)

    assert torch.equal(idx.cpu(), ref_idx.cpu())
    assert torch.allclose(dist.cpu(), ref_dist.cpu(), atol=1e-4)


@skip_no_cuda
def test_knn_distances_are_ascending():
    ref = torch.randn(3, 500, 3).cuda()
    query = torch.randn(3, 30, 3).cuda()
    dist, _ = _knn_forward(ref, query, 40)
    assert torch.all(dist[:, :, 1:] >= dist[:, :, :-1] - 1e-5)


@skip_no_cuda
def test_knn_indices_are_distinct_per_query():
    ref = torch.randn(2, 300, 3).cuda()
    query = torch.randn(2, 15, 3).cuda()
    _, idx = _knn_forward(ref, query, 50)
    for b in range(idx.shape[0]):
        for g in range(idx.shape[1]):
            row = idx[b, g].tolist()
            assert len(set(row)) == len(row)


@skip_no_cuda
def test_knn_indices_in_range():
    B, N, G, k = 2, 77, 9, 20
    ref = torch.randn(B, N, 3).cuda()
    query = torch.randn(B, G, 3).cuda()
    _, idx = _knn_forward(ref, query, k)
    assert idx.min().item() >= 0
    assert idx.max().item() < N


@skip_no_cuda
def test_knn_exact_query_match_returns_itself_first():
    # query points that are exact copies of some ref points: the nearest
    # neighbor of such a query must be that exact ref point, at distance 0.
    ref = torch.randn(1, 100, 3).cuda()
    query = ref[:, [3, 17, 42], :].clone()
    dist, idx = _knn_forward(ref, query, 5)
    assert torch.equal(idx[0, :, 0].cpu(), torch.tensor([3, 17, 42]))
    assert torch.allclose(dist[0, :, 0].cpu(), torch.zeros(3), atol=1e-5)


@skip_no_cuda
def test_knn_duplicate_points_set_matches_reference():
    # exact ties: order is implementation-defined (true of the original
    # native kernel too), so compare the *set* of selected indices, not order.
    ref = torch.zeros(1, 10, 3).cuda()
    ref[0, :5] = torch.tensor([1.0, 0.0, 0.0])  # 5 duplicate points, all at distance 1
    ref[0, 5:] = torch.tensor([10.0, 0.0, 0.0])  # 5 duplicate points, far away
    query = torch.zeros(1, 1, 3).cuda()

    _, got_idx = _knn_forward(ref, query, 5)
    _, ref_idx = reference_knn(ref, query, 5)

    assert set(got_idx[0, 0].tolist()) == set(ref_idx[0, 0].tolist()) == {0, 1, 2, 3, 4}


@skip_no_cuda
def test_knn_group_forward_pipeline_shapes():
    # mirrors how src/models/pointcloud_encoder.py's Group.forward uses KNN
    B, num_points, num_group, group_size = 2, 512, 32, 16
    xyz = torch.randn(B, num_points, 3).cuda()
    center = torch.randn(B, num_group, 3).cuda()

    knn = KNN(k=group_size, transpose_mode=True)
    _, idx = knn(xyz, center)

    assert idx.size(1) == num_group
    assert idx.size(2) == group_size
