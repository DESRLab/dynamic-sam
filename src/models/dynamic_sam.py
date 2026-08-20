from __future__ import annotations

import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional, Tuple, TypedDict
from typing_extensions import TypeAlias

from .pointcloud_encoder import PointCloudEncoder, PcdEncoderParams
from .prompt_encoder import PromptEncoder, PromptEncoderParams
from .mask_decoder import MaskDecoder, MaskDecoderParams


Tensor: TypeAlias = torch.Tensor

__all__ = ['DynamicSAM', 'ModelOutPut', 'PcdSession']


class ModelOutPut(TypedDict):
    pred_logits: Tensor
    pred_mask: Tensor
    binarized_mask: Tensor


@dataclass
class PcdSession:
    """Everything decode() needs for one encoded point cloud.

    Returned by `encode_pcd` instead of being stored on `self`, so that
    concurrent callers (e.g. one per user session in src/serve/__init__.py) can
    each hold their own encoded point cloud without racing on shared model
    state -- see `encode_pcd`/`decode`.
    """
    min_vals: Tensor
    max_vals: Tensor
    pcd_embeddings: Tensor
    pcd_pe: Tensor
    normalized_pcd: Tensor
    center: Tensor


class DynamicSAM(nn.Module):
    def __init__(self, pcd_encoder_params: PcdEncoderParams, prompt_encoder_params: PromptEncoderParams, mask_decoder_params: MaskDecoderParams):
        super().__init__()

        self.pcd_encoder = PointCloudEncoder(pcd_encoder_params)
        self.prompt_encoder = PromptEncoder(prompt_encoder_params, self.pcd_encoder.pos_embed)
        self.mask_decoder = MaskDecoder(mask_decoder_params)
        self._min_vals = None
        self._max_vals = None
        self._pcd_embeddings = None
        self._pcd_pe = None

    @property
    def min_vals(self) -> Optional[Tensor]:
        return self._min_vals

    @property
    def max_vals(self) -> Optional[Tensor]:
        return self._max_vals

    @property
    def pcd_embeddings(self) -> Optional[Tensor]:
        return self._pcd_embeddings

    @property
    def pcd_pe(self) -> Optional[Tensor]:
        return self._pcd_pe

    def load_modules_state_dict(self, pcd_encoder: dict, prompt_encoder: dict, mask_decoder: dict):
        self.pcd_encoder.load_state_dict(pcd_encoder)
        self.prompt_encoder.load_state_dict(prompt_encoder)
        self.mask_decoder.load_state_dict(mask_decoder)

    @staticmethod
    def _normalize_pcd(pcd: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        min_vals, _ = torch.min(input=pcd, dim=1, keepdim=True)
        max_vals, _ = torch.max(input=pcd, dim=1, keepdim=True)
        normalized_point_cloud = (pcd - min_vals) / (max_vals - min_vals)
        return normalized_point_cloud, min_vals, max_vals

    def preprocess_pcd(self, pcd: Tensor) -> Tensor:
        normalized_point_cloud, min_vals, max_vals = self._normalize_pcd(pcd)
        self._min_vals = min_vals
        self._max_vals = max_vals
        return normalized_point_cloud

    def normalize_point(self, point: Tensor) -> Tensor:
        assert self.min_vals is not None and self.max_vals is not None

        return (point - self.min_vals) / (self.max_vals - self.min_vals)

    def encode_pcd(self, pcd: Tensor, group_size: Optional[int] = None, num_group: Optional[int] = None) -> PcdSession:
        """Stateless point-cloud encoding.

        Returns a `PcdSession` instead of mutating `self`, so it is safe to
        call concurrently for different point clouds (e.g. different user
        sessions sharing one model instance). `group_size`/`num_group`, when
        passed, override the encoder's configured grouping for this call only
        (see `PointCloudEncoder.forward`) -- also without mutating shared state.
        """
        normalized_pcd, min_vals, max_vals = self._normalize_pcd(pcd.float())
        pcd_embeddings, pos_embedding, center = self.pcd_encoder(normalized_pcd, group_size=group_size, num_group=num_group)
        return PcdSession(
            min_vals=min_vals,
            max_vals=max_vals,
            pcd_embeddings=pcd_embeddings,
            pcd_pe=pos_embedding,
            normalized_pcd=normalized_pcd,
            center=center,
        )

    def set_pcd(self, pcd: Tensor):
        session = self.encode_pcd(pcd)
        self._min_vals = session.min_vals
        self._max_vals = session.max_vals
        self.mask_decoder.set_input_pcd(session.normalized_pcd, session.center)
        self._pcd_embeddings = session.pcd_embeddings
        self._pcd_pe = session.pcd_pe

    def decode(self, session: PcdSession, point_prompts: Tensor, prompts_labels: Tensor) -> Tensor:
        """Stateless mask decoding: takes the `PcdSession` from `encode_pcd`
        explicitly instead of reading `self.pcd_embeddings`/`self.pcd_pe`, so
        it never races with another session's `encode_pcd`/`decode` call.
        """
        normalized_points = (point_prompts - session.min_vals) / (session.max_vals - session.min_vals)
        sparse_embedding = self.prompt_encoder(points=(normalized_points, prompts_labels))

        predicted_masks = self.mask_decoder(
            session.pcd_embeddings, session.pcd_pe, sparse_embedding, (sparse_embedding.shape[1] == 1),
            input_pcd=session.normalized_pcd, pcd_center=session.center,
        )
        return predicted_masks

    def decode_with_post_process(self, session: PcdSession, point_prompts: Tensor, prompts_labels: Tensor, threshold: float = 0.49) -> ModelOutPut:
        predicted_masks = self.decode(session, point_prompts, prompts_labels)
        return self.post_process_model_output(predicted_masks, threshold)

    @torch.no_grad()
    def decode_inference(self, session: PcdSession, point_prompts: Tensor, prompts_labels: Tensor) -> Tensor:
        mask_logits = self.decode(session, point_prompts, prompts_labels)
        sig_logits = torch.sigmoid(mask_logits)
        return sig_logits.squeeze(0).transpose(1, 0).squeeze(1)

    def forward(self, point_prompts: Tensor, prompts_labels: Tensor) -> Tensor:
        assert self.pcd_embeddings is not None and self.pcd_pe is not None
        session = PcdSession(
            min_vals=self.min_vals, max_vals=self.max_vals,
            pcd_embeddings=self.pcd_embeddings, pcd_pe=self.pcd_pe,
            normalized_pcd=self.mask_decoder.input_pcd, center=self.mask_decoder.pcd_center,
        )
        return self.decode(session, point_prompts, prompts_labels)

    def forward_with_post_process(self, point_prompts: Tensor, prompts_labels: Tensor, threshold: float = 0.49) -> ModelOutPut:
        predicted_masks = self.forward(point_prompts, prompts_labels)
        return self.post_process_model_output(predicted_masks, threshold)

    def post_process_model_output(self, mask_output: torch.Tensor, threshold: float = 0.49) -> ModelOutPut:
        mask_logits = mask_output
        sig_mask = torch.sigmoid(mask_logits)
        binarized_mask = torch.where(sig_mask > threshold, torch.tensor(1.0), torch.tensor(0.0))

        return ModelOutPut(
            pred_logits=mask_logits,
            pred_mask=sig_mask,
            binarized_mask=binarized_mask,
        )


    @torch.no_grad()
    def inference(self, point_prompts: Tensor, prompts_labels: Tensor) -> Tensor:
        mask_logits = self.forward(point_prompts, prompts_labels)
        sig_logits = torch.sigmoid(mask_logits)
        return sig_logits.squeeze(0).transpose(1,0).squeeze(1)
