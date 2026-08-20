from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from typing import Optional, Tuple, Type
from typing_extensions import TypeAlias

Tensor: TypeAlias = torch.Tensor

__all__ = ['PromptEncoder', 'PromptEncoderParams']

@dataclass(frozen=True)
class PromptEncoderParams:
    embedding_dim: int


class PromptEncoder(nn.Module):
    def __init__(self, params: PromptEncoderParams, pos_embedding: nn.Sequential, activation: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        
        self.pos_embedding = pos_embedding
        self._encoder_params = params
        self.num_point_embeddings: int = 2
        self.embedding_dim = params.embedding_dim
        point_embeddings = [nn.Embedding(1, params.embedding_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, params.embedding_dim)
    
    @property
    def encoder_params(self) -> PromptEncoderParams:
        return self._encoder_params

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device # type: ignore

    def _embed_points(
        self,
        coords: torch.Tensor,
        coords_labels: torch.Tensor,
        pad: bool = False,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        if pad:
            padding_point = torch.zeros((coords.shape[0], 1, 3), device=coords.device, dtype=coords.dtype)
            padding_label = -torch.ones((coords_labels.shape[0], 1), device=coords_labels.device, dtype=coords_labels.dtype)
            coords = torch.cat([coords, padding_point], dim=1)
            coords_labels = torch.cat([coords_labels, padding_label], dim=1)

        point_embedding = self.pos_embedding(coords)

        point_embedding = torch.where((coords_labels == -1).unsqueeze(-1).expand_as(point_embedding),
                                      torch.zeros_like(point_embedding) + self.not_a_point_embed.weight,
                                      point_embedding)
        point_embedding = torch.where((coords_labels == 0).unsqueeze(-1).expand_as(point_embedding),
                                      point_embedding + self.point_embeddings[0].weight,
                                      point_embedding)
        point_embedding = torch.where((coords_labels == 1).unsqueeze(-1).expand_as(point_embedding),
                                      point_embedding + self.point_embeddings[1].weight,
                                      point_embedding)

        return point_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[Tensor, Tensor]],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        else:
            return 1

    def forward(
        self,
        points: Tuple[Tensor, Tensor],
    ) -> Tensor:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points.
        """
        coords, labels = points

        sparse_embeddings = self._embed_points(coords, labels, pad=True)

        return sparse_embeddings
