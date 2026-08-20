"""Concurrency/isolation/capacity tests for src/serve's session cache.

The original bug: the API stored the encoded point cloud on the single shared
DynamicSAM instance (predictor.set_pcd(...) -> predictor._pcd_embeddings etc.),
so one user's /encode_pcd could silently overwrite another user's in-flight
session before their /predict_mask ran. These tests build the real FastAPI app
(create_fast_api) around a small randomly-initialized DynamicSAM and drive it
through FastAPI's TestClient, including genuine multithreaded interleaving
(FastAPI runs these sync def handlers in a thread pool, mirroring real
concurrent users).

Cache model (UserFrameCache in src/serve/__init__.py): up to `max_users`
distinct users, each holding up to `max_frames_per_user` distinct encoded
frames (pcd_id). /encode_pcd upserts a (user_id, pcd_id) entry; once either
capacity is exceeded, the least-recently-*encoded* entry at that level is
evicted (stack/FIFO by encode order, not by read access). /predict_mask on an
evicted-or-never-encoded (user_id, pcd_id) 404s.
"""
import concurrent.futures

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.serve import create_fast_api
from src.models import DynamicSAM, MaskDecoderParams, PcdEncoderParams, PromptEncoderParams

CUDA = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not CUDA, reason="model runs on CUDA")

NUM_GROUP = 64


def make_client(max_users: int = 10, max_frames_per_user: int = 10) -> TestClient:
    pcd_p = PcdEncoderParams(trans_dim=384, depth=2, num_heads=6, group_size=32, num_group=NUM_GROUP)
    model = DynamicSAM(pcd_p, PromptEncoderParams(embedding_dim=384), MaskDecoderParams(trans_dim=384)).cuda()
    model.eval()
    app = create_fast_api(model, device="cuda:0", max_users=max_users, max_frames_per_user=max_frames_per_user)
    return TestClient(app)


def encode(client: TestClient, user_id: str, pcd_id: str, pcd: np.ndarray) -> None:
    resp = client.post(
        "/encode_pcd",
        content=pcd.astype(np.float32).tobytes(),
        headers={"X-Num-Points": str(pcd.shape[0]), "X-User-Id": user_id, "X-Pcd-Id": pcd_id},
    )
    assert resp.status_code == 200, resp.text


def predict(client: TestClient, user_id: str, pcd_id: str, points: np.ndarray, labels: list, expect_status: int = 200) -> np.ndarray | None:
    resp = client.request(
        "PUT",
        "/predict_mask",
        content=points.astype(np.float32).tobytes(),
        headers={
            "X-User-Id": user_id,
            "X-Pcd-Id": pcd_id,
            "X-Num-Points": str(points.shape[0]),
            "X-Labels": ",".join(str(l) for l in labels),
        },
    )
    assert resp.status_code == expect_status, resp.text
    if expect_status != 200:
        return None
    return np.frombuffer(resp.content, dtype=np.float32)


@skip_no_cuda
def test_two_users_do_not_cross_contaminate():
    client = make_client()

    pcd_a = np.random.randn(500, 3).astype(np.float32)
    pcd_b = np.random.randn(500, 3).astype(np.float32) * 5 + 10  # clearly different point cloud

    encode(client, "user-a", "scene-1", pcd_a)
    encode(client, "user-b", "scene-1", pcd_b)

    points = np.random.randn(3, 3).astype(np.float32)
    labels = [1, 1, 0]

    out_a = predict(client, "user-a", "scene-1", points, labels)
    out_b = predict(client, "user-b", "scene-1", points, labels)

    # different encoded point clouds -> different predictions for the same
    # query: proves /predict_mask used each user's own encoding, not
    # whichever point cloud happened to be encoded most recently.
    assert not np.allclose(out_a, out_b)

    # re-predicting on user-a must reproduce the same result: user-a's state
    # must not have been perturbed by encoding/predicting for user-b.
    out_a_again = predict(client, "user-a", "scene-1", points, labels)
    assert np.allclose(out_a, out_a_again)


@skip_no_cuda
def test_predict_mask_requires_encoded_user():
    client = make_client()
    points = np.random.randn(3, 3).astype(np.float32)
    predict(client, "nobody", "scene-1", points, [1, 1, 0], expect_status=404)


@skip_no_cuda
def test_predict_mask_still_works_for_an_older_still_cached_frame():
    """Switching scenes no longer invalidates the previous frame outright --
    it just becomes one entry among up to max_frames_per_user cached ones."""
    client = make_client(max_frames_per_user=5)
    pcd_1 = np.random.randn(300, 3).astype(np.float32)
    pcd_2 = np.random.randn(300, 3).astype(np.float32) * 4

    encode(client, "user-a", "scene-1", pcd_1)
    points = np.random.randn(2, 3).astype(np.float32)
    out_scene_1 = predict(client, "user-a", "scene-1", points, [1, 0])

    # same session (same user), switches scenes: encode under the same
    # user_id with a new pcd_id. Well within max_frames_per_user=5, so
    # scene-1 is NOT evicted.
    encode(client, "user-a", "scene-2", pcd_2)

    out_scene_1_again = predict(client, "user-a", "scene-1", points, [1, 0])
    assert np.allclose(out_scene_1, out_scene_1_again)

    out_scene_2 = predict(client, "user-a", "scene-2", points, [1, 0])
    assert not np.allclose(out_scene_1, out_scene_2)


@skip_no_cuda
def test_delete_session_invalidates_all_frames_for_that_user():
    client = make_client()
    encode(client, "user-a", "scene-1", np.random.randn(200, 3).astype(np.float32))
    encode(client, "user-a", "scene-2", np.random.randn(200, 3).astype(np.float32))

    resp = client.delete("/session", headers={"X-User-Id": "user-a"})
    assert resp.status_code == 200

    points = np.random.randn(1, 3).astype(np.float32)
    predict(client, "user-a", "scene-1", points, [1], expect_status=404)
    predict(client, "user-a", "scene-2", points, [1], expect_status=404)


@skip_no_cuda
def test_multiple_decode_requests_reuse_one_encode():
    """The core workflow: encode a scene once, then fire many /predict_mask
    calls (e.g. iterative click refinement) against that same encoding."""
    client = make_client()
    pcd = np.random.randn(350, 3).astype(np.float32)
    encode(client, "user-a", "scene-1", pcd)

    for _ in range(10):
        points = np.random.randn(np.random.randint(1, 5), 3).astype(np.float32)
        labels = list(np.random.randint(0, 2, size=points.shape[0]))
        out = predict(client, "user-a", "scene-1", points, labels)
        assert out.shape[0] == pcd.shape[0]


@skip_no_cuda
def test_max_frames_per_user_evicts_oldest_frame_first():
    client = make_client(max_users=10, max_frames_per_user=3)
    pcds = {f"scene-{i}": np.random.randn(100, 3).astype(np.float32) * (i + 1) for i in range(4)}

    for pcd_id, pcd in pcds.items():
        encode(client, "user-a", pcd_id, pcd)

    points = np.random.randn(2, 3).astype(np.float32)
    # scene-0 was encoded first and capacity is 3, so it's the one evicted
    # when scene-3 (the 4th distinct frame) is encoded.
    predict(client, "user-a", "scene-0", points, [1, 0], expect_status=404)
    # the 3 most recently encoded frames all remain.
    for pcd_id in ("scene-1", "scene-2", "scene-3"):
        predict(client, "user-a", pcd_id, points, [1, 0], expect_status=200)


@skip_no_cuda
def test_max_users_evicts_oldest_user_first():
    client = make_client(max_users=2, max_frames_per_user=5)
    for i in range(3):
        encode(client, f"user-{i}", "scene-1", np.random.randn(100, 3).astype(np.float32) * (i + 1))

    points = np.random.randn(2, 3).astype(np.float32)
    # user-0 was the first encoded and capacity is 2 users, evicted entirely
    # once user-2 shows up.
    predict(client, "user-0", "scene-1", points, [1, 0], expect_status=404)
    predict(client, "user-1", "scene-1", points, [1, 0], expect_status=200)
    predict(client, "user-2", "scene-1", points, [1, 0], expect_status=200)


@skip_no_cuda
def test_default_capacity_is_one_user_five_frames():
    """create_fast_api's defaults (no max_users/max_frames_per_user passed)
    should match the documented default: 1 user, 5 frames."""
    pcd_p = PcdEncoderParams(trans_dim=384, depth=2, num_heads=6, group_size=32, num_group=NUM_GROUP)
    model = DynamicSAM(pcd_p, PromptEncoderParams(embedding_dim=384), MaskDecoderParams(trans_dim=384)).cuda()
    model.eval()
    client = TestClient(create_fast_api(model, device="cuda:0"))

    for i in range(5):
        encode(client, "user-a", f"scene-{i}", np.random.randn(100, 3).astype(np.float32) * (i + 1))
    points = np.random.randn(2, 3).astype(np.float32)
    for i in range(5):
        predict(client, "user-a", f"scene-{i}", points, [1, 0], expect_status=200)

    # a 6th distinct frame for the same user evicts the oldest (scene-0).
    encode(client, "user-a", "scene-5", np.random.randn(100, 3).astype(np.float32))
    predict(client, "user-a", "scene-0", points, [1, 0], expect_status=404)
    predict(client, "user-a", "scene-5", points, [1, 0], expect_status=200)

    # a second distinct user evicts user-a entirely (max_users=1 default).
    encode(client, "user-b", "scene-0", np.random.randn(100, 3).astype(np.float32))
    predict(client, "user-a", "scene-1", points, [1, 0], expect_status=404)
    predict(client, "user-b", "scene-0", points, [1, 0], expect_status=200)


@skip_no_cuda
def test_reencoding_existing_frame_refreshes_its_recency():
    client = make_client(max_users=10, max_frames_per_user=2)
    pcd_a1 = np.random.randn(100, 3).astype(np.float32)
    pcd_a2 = np.random.randn(100, 3).astype(np.float32) * 2

    encode(client, "user-a", "scene-A", pcd_a1)
    encode(client, "user-a", "scene-B", np.random.randn(100, 3).astype(np.float32))
    # re-encoding scene-A should count as fresh, protecting it from eviction
    # ahead of scene-B (which is now the older of the two).
    encode(client, "user-a", "scene-A", pcd_a2)
    encode(client, "user-a", "scene-C", np.random.randn(100, 3).astype(np.float32))

    points = np.random.randn(2, 3).astype(np.float32)
    # scene-B, not scene-A, should have been evicted to make room for scene-C.
    predict(client, "user-a", "scene-B", points, [1, 0], expect_status=404)
    predict(client, "user-a", "scene-A", points, [1, 0], expect_status=200)
    predict(client, "user-a", "scene-C", points, [1, 0], expect_status=200)


@skip_no_cuda
def test_concurrent_interleaved_users_stay_isolated():
    """Interleave many /predict_mask calls for two distinct users across
    threads and check every prediction stays attributable to its own user.
    """
    client = make_client()

    pcds = {
        "user-a": np.random.randn(400, 3).astype(np.float32),
        "user-b": np.random.randn(400, 3).astype(np.float32) * 3 - 7,
    }
    points = np.random.randn(4, 3).astype(np.float32)
    labels = [1, 1, 0, 0]

    baseline = {}
    for user_id, pcd in pcds.items():
        encode(client, user_id, "scene-1", pcd)
        baseline[user_id] = predict(client, user_id, "scene-1", points, labels)

    def worker(user_id):
        got = predict(client, user_id, "scene-1", points, labels)
        return user_id, np.allclose(got, baseline[user_id])

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(worker, user_id) for user_id in list(pcds.keys()) * 20]
        results = [f.result() for f in futures]

    for user_id, matched in results:
        assert matched, f"user {user_id} prediction drifted under concurrent access"


@skip_no_cuda
def test_concurrent_encode_with_different_point_counts_does_not_crash_or_leak_group_size():
    """Concurrent /encode_pcd calls (different users) with different point
    counts imply different group_size overrides (see PointCloudEncoder.forward).
    If the override leaked between concurrent calls, Group.forward's internal
    shape assertions would fail under the race.
    """
    point_counts = [300, 5000] * 10
    client = make_client(max_users=len(point_counts), max_frames_per_user=10)

    def worker(i, n):
        user_id = f"user-{i}"
        pcd = np.random.randn(n, 3).astype(np.float32)
        encode(client, user_id, "scene-1", pcd)
        return user_id, n

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        users = [f.result() for f in [ex.submit(worker, i, n) for i, n in enumerate(point_counts)]]

    # the model outputs one mask value per point in the ORIGINAL (encoded)
    # point cloud, not per query point -- so each user's prediction shape
    # must match that user's own point count, never another user's.
    points = np.random.randn(2, 3).astype(np.float32)
    for user_id, n in users:
        out = predict(client, user_id, "scene-1", points, [1, 0])
        assert out.shape[0] == n, f"user encoded with {n} points returned {out.shape[0]} -- group_size override leaked across users"
