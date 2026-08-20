from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Optional
from typing_extensions import TypeAlias

from .transformer import TwoWayTransformer
from .pointnet2_utils import PointNetFeaturePropagation

Tensor: TypeAlias = torch.Tensor

__all__ = ['MaskDecoder', 'MaskDecoderParams']

@dataclass(frozen=True)
class MaskDecoderParams:
    trans_dim: int
    num_multimask_outputs: int = 3

class MaskDecoder(nn.Module):
    def __init__(self, params: MaskDecoderParams) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
        """
        super().__init__()
        self._decoder_params = params
        self.transformer_dim = params.trans_dim
        self.transformer = TwoWayTransformer(embedding_dim=params.trans_dim)

        self.num_multimask_outputs = params.num_multimask_outputs

        self.num_mask_tokens = params.num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, params.trans_dim)
        self.pointnet_upsample = PointNetFeaturePropagation(in_channel=params.trans_dim+3,
                                                        mlp=[params.trans_dim // 4, params.trans_dim // 8])

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(params.trans_dim, params.trans_dim, params.trans_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )

        self._input_pcd = None
        self._pcd_center = None

    @property
    def decoder_params(self) -> MaskDecoderParams:
        return self._decoder_params
    @property
    def input_pcd(self) -> Optional[Tensor]:
        return self._input_pcd
    
    @property
    def pcd_center(self) -> Optional[Tensor]:
        return self._pcd_center

    def set_input_pcd(self, pcd_input: Tensor, center: Tensor):
        self._input_pcd = pcd_input
        self._pcd_center = center

    def forward(
        self,
        pcd_embeddings: Tensor,
        pcd_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        multimask_output: bool,
        input_pcd: Optional[Tensor] = None,
        pcd_center: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          pcd_embeddings (Tensor): the embeddings from the image encoder
          pcd_pe (Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.
          input_pcd (Tensor, optional): the point cloud to upsample onto. Defaults
            to the value set by `set_input_pcd` when not provided, for callers
            that rely on that stateful path (e.g. training).
          pcd_center (Tensor, optional): the group centers matching `input_pcd`.
            Same default-to-`set_input_pcd` behavior as `input_pcd`.

        Returns:
          Tensor: batched predicted masks
        """
        masks = self.predict_masks(
            pcd_embeddings=pcd_embeddings,
            pcd_pe=pcd_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            input_pcd=input_pcd,
            pcd_center=pcd_center,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :]

        # Prepare output
        return masks

    def predict_masks(
        self,
        pcd_embeddings: Tensor,
        pcd_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        input_pcd: Optional[Tensor] = None,
        pcd_center: Optional[Tensor] = None,
    ) -> Tensor:
        """Predicts masks. See 'forward' for more details."""
        if input_pcd is None:
            input_pcd = self.input_pcd
        if pcd_center is None:
            pcd_center = self.pcd_center
        assert input_pcd is not None and pcd_center is not None

        # Concatenate output tokens
        output_tokens = self.mask_tokens.weight
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = pcd_embeddings
        pos_src = pcd_pe

        # Expand per-pcd data in batch direction to be per-mask
        if pcd_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(pcd_embeddings, tokens.shape[0], dim=0)
            pos_src = torch.repeat_interleave(pcd_pe, tokens.shape[0], dim=0)

        # Run the transformer
        mask_tokens_out, src = self.transformer(src, pos_src, tokens)

        upscaled_embedding = self.pointnet_upsample(input_pcd.transpose(-1,-2), pcd_center.transpose(-1,-2), input_pcd.transpose(-1,-2), src.transpose(-1,-2))

        hyper_in_list: List[Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        masks = hyper_in @ upscaled_embedding

        return masks

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
