from __future__ import annotations

import threading
import torch

from collections import OrderedDict

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response

import numpy as np
from pydantic import BaseModel

from src.models import DynamicSAM, PcdSession

__all__ = ['create_fast_api', 'UserFrameCache']

class EncoderInput(BaseModel):
    input_pcd : bytes
    num_points: int

class DecoderInputs(BaseModel):
    points: bytes
    labels: list[int]
    num_points: int

class PredResults(BaseModel):
    mask: bytes
    iou : bytes

class UserFrameCache:
    """Bounded, thread-safe cache of encoded point clouds: up to `max_users`
    distinct users, each holding up to `max_frames_per_user` distinct
    encoded frames (keyed by pcd_id).

    Eviction is stack/FIFO-style based on encode order, at both levels: once
    a user's frame count would exceed `max_frames_per_user`, that user's
    least-recently-*encoded* frame is dropped to make room for the new one;
    once the number of distinct users would exceed `max_users`, the
    least-recently-*encoded* user's entire frame cache is dropped. Re-encoding
    an existing (user_id, pcd_id) counts as fresh and moves it back to the
    front, so actively-used entries aren't evicted just because other
    users/frames were touched in between.

    A lock is needed here (unlike a plain dict) because eviction is a
    check-then-act sequence over nested OrderedDicts, not a single atomic
    dict operation.
    """

    def __init__(self, max_users: int, max_frames_per_user: int):
        assert max_users > 0 and max_frames_per_user > 0
        self.max_users = max_users
        self.max_frames_per_user = max_frames_per_user
        self._users: OrderedDict[str, OrderedDict[str, PcdSession]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, user_id: str, pcd_id: str, session: PcdSession) -> None:
        with self._lock:
            if user_id in self._users:
                self._users.move_to_end(user_id)
            else:
                if len(self._users) >= self.max_users:
                    evicted_user, _ = self._users.popitem(last=False)
                    print(f"[session cache] max_users={self.max_users} reached, evicted user '{evicted_user}'")
                self._users[user_id] = OrderedDict()

            user_frames = self._users[user_id]
            if pcd_id in user_frames:
                del user_frames[pcd_id]
            elif len(user_frames) >= self.max_frames_per_user:
                evicted_pcd_id, _ = user_frames.popitem(last=False)
                print(f"[session cache] max_frames_per_user={self.max_frames_per_user} reached for user "
                      f"'{user_id}', evicted frame '{evicted_pcd_id}'")
            user_frames[pcd_id] = session

    def get(self, user_id: str, pcd_id: str) -> PcdSession | None:
        with self._lock:
            user_frames = self._users.get(user_id)
            if user_frames is None:
                return None
            return user_frames.get(pcd_id)

    def delete_user(self, user_id: str) -> bool:
        with self._lock:
            return self._users.pop(user_id, None) is not None

async def parse_pcd_encoder_body(request: Request) -> EncoderInput:
    data: bytes = await request.body()
    return EncoderInput(input_pcd=data, num_points=int(request.headers.get('X-Num-Points', 0)))

async def parse_mask_decoder_body(request: Request) -> DecoderInputs:
    points: bytes = await request.body()
    num_points = int(request.headers.get('X-Num-Points', 0))
    labels_string = request.headers.get('X-Labels', ',')
    labels = [int(num) for num in labels_string.split(',')]

    return DecoderInputs(
        points=points,
        labels=labels,
        num_points=num_points,
    )

def create_fast_api(predictor: DynamicSAM, device, max_users: int = 1, max_frames_per_user: int = 5) -> FastAPI:
    app = FastAPI()
    # Kept off the shared `predictor` instance on purpose: multiple users hit
    # the same predictor concurrently (FastAPI runs these sync handlers in a
    # thread pool), so storing encoded point clouds on `predictor` itself
    # would let one user's /predict_mask read another user's point cloud.
    cache = UserFrameCache(max_users=max_users, max_frames_per_user=max_frames_per_user)

    @app.get('/')
    def read_root():
        return {"hello": "world"}

    @app.post("/encode_pcd")
    def encode_pcd(
        encoder_input: EncoderInput = Depends(parse_pcd_encoder_body), # type: ignore
        x_user_id: str = Header(...),
        x_pcd_id: str = Header(...),
    ):
        pcd_arr = np.frombuffer(encoder_input.input_pcd, dtype=np.float32).reshape(encoder_input.num_points, 3)[None,:]
        if isinstance(pcd_arr, np.ndarray):
            try:
                # Adapt group_size to this point cloud's size, same as before,
                # but as a per-call override (see DynamicSAM.encode_pcd) rather
                # than mutating predictor.pcd_encoder.group_size --- that
                # mutation would itself race across concurrent users.
                num_group = predictor.pcd_encoder.num_group
                group_size = int(pcd_arr.shape[1] / num_group) + 1

                pcd_tensor = torch.from_numpy(pcd_arr.astype(np.float32)).to(device)
                pcd_session = predictor.encode_pcd(pcd_tensor, group_size=group_size, num_group=num_group)

                cache.put(x_user_id, x_pcd_id, pcd_session)

                return {"status": "success"}
            except Exception as e:
                print(e)
                raise HTTPException(status_code=400, detail=str(e))
        else:
           raise HTTPException(status_code=400, detail=str('Provided wrong input data.'))

    @app.put('/predict_mask')
    def predict_mask(
        decoder_inputs: DecoderInputs = Depends(parse_mask_decoder_body),
        x_user_id: str = Header(...),
        x_pcd_id: str = Header(...),
    ):
        pcd_session = cache.get(x_user_id, x_pcd_id)
        if pcd_session is None:
            raise HTTPException(
                status_code=404,
                detail=f'No cached encoding for user {x_user_id}, pcd {x_pcd_id} '
                       '(never encoded, or evicted to make room for more recent frames/users)',
            )

        points_arr = np.frombuffer(decoder_inputs.points, dtype=np.float32).reshape(decoder_inputs.num_points, 3)
        labels_arr = np.array(decoder_inputs.labels)
        points = torch.from_numpy(points_arr.copy()).to(device)
        labels = torch.from_numpy(labels_arr).to(device)
        try:
            logits = predictor.decode_inference(pcd_session, points[None,:], labels[None,:])
            masks_logits = logits.cpu().numpy().astype(np.float32)

            return Response(content=masks_logits.tobytes(), media_type='application/octet-stream')

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete('/session')
    def end_session(x_user_id: str = Header(...)):
        if not cache.delete_user(x_user_id):
            raise HTTPException(status_code=404, detail=f'No active session for user: {x_user_id}')
        return {"status": "success"}

    return app
