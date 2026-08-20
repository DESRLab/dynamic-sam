from __future__ import annotations

from dataclasses import dataclass
from src.kernel.fps import furthest_point_sample
from src.kernel.gather import gather_operation
from src.kernel.knn import KNN

from typing import Union, List, Tuple, Optional
from typing_extensions import TypeAlias

import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_

from .utils import get_missing_parameters_message, get_unexpected_parameters_message

Tensor: TypeAlias = torch.Tensor

__all__ = ['PointCloudEncoder', 'PcdEncoderParams']

def fps(data: Tensor, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = furthest_point_sample(data, number)
    fps_data = gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous() # type: ignore
    return fps_data


class Group(nn.Module):
    def __init__(self, num_group: int, group_size: int):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz: Tensor, group_size: Optional[int] = None, num_group: Optional[int] = None) -> Tuple[Tensor,...]:
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3

            group_size/num_group, when passed, override self.group_size/self.num_group
            for this call only (no mutation of shared state), e.g. to adapt to a
            differently-sized point cloud at inference time without a data race
            across concurrent callers sharing this module.
        '''
        group_size = group_size if group_size is not None else self.group_size
        num_group = num_group if num_group is not None else self.num_group

        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz.cuda(), num_group)  # B G 3
        # knn to get the neighborhood. Built fresh from the effective group_size
        # on every call (rather than cached at __init__) so that group_size
        # actually affects the result, whether from self.group_size or an
        # explicit override above.
        knn = KNN(k=group_size, transpose_mode=True)
        _, idx = knn(xyz.cuda(), center.cuda())  # B G M
        assert idx.size(1) == num_group
        assert idx.size(2) == group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx.to(idx_base.device) + idx_base
        idx = idx.view(-1)
        center = center.to(idx.device)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.to(center.device)
        neighborhood = neighborhood.view(batch_size, num_group, group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Encoder(nn.Module):
    def __init__(self, encoder_channel: int):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups: Tensor) -> Tensor:
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int]=None, out_features: Optional[int]=None, act_layer=nn.GELU, drop:float=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int=8, qkv_bias=False, qk_scale=None, attn_drop: float=0., proj_drop: float=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path: float=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim:int=768, depth:int=4, num_heads:int=12, mlp_ratio:float=4., qkv_bias=False, qk_scale=None,
                 drop_rate:float=0., attn_drop_rate:float=0., drop_path_rate: Union[List[float],float]=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        for i, block in enumerate(self.blocks):
            x = block(x + pos)

        return x

@dataclass(frozen=True)
class PcdEncoderParams:
    trans_dim:int=384
    depth:int=12
    drop_path_rate:float=0.1
    num_heads:int=6
    group_size: int=64
    num_group: int=128

class PointCloudEncoder(nn.Module):
    def __init__(self, params: PcdEncoderParams):
        super().__init__()

        self._encoder_params = params
        self.trans_dim = params.trans_dim
        self.depth = params.depth
        self.drop_path_rate = params.drop_path_rate
        self.num_heads = params.num_heads

        # grouper. group_size/num_group below are properties that delegate to
        # this Group instance, so it's the single source of truth --- reading
        # or writing pcd_encoder.group_size actually changes Group's behavior.
        self.group_divider = Group(num_group=params.num_group, group_size=params.group_size)
        # define the encoder
        self.encoder = Encoder(encoder_channel=self.trans_dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )
        
        self.norm = nn.LayerNorm(self.trans_dim)
        
        self.apply(self._init_weights)

    @property
    def group_size(self) -> int:
        return self.group_divider.group_size

    @group_size.setter
    def group_size(self, value: int):
        self.group_divider.group_size = value

    @property
    def num_group(self) -> int:
        return self.group_divider.num_group

    @num_group.setter
    def num_group(self, value: int):
        self.group_divider.num_group = value

    @property
    def encoder_params(self) -> PcdEncoderParams:
        return self._encoder_params

    def forward(self, pcd: Tensor, group_size: Optional[int] = None, num_group: Optional[int] = None) -> Tuple[Tensor, ...]:
        neighborhood, center = self.group_divider(pcd, group_size=group_size, num_group=num_group)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        
        x = group_input_tokens
        pos = self.pos_embed(center)

        pcd_features = self.blocks(x, pos)
        pcd_embeddings = self.norm(pcd_features).transpose(-1, -2).contiguous()
        
        return pcd_embeddings, pos, center
        
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def load_model_from_conv_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            model_ckpt = {}
            for k, v in base_ckpt.items():
                if k.startswith('model.encoder.blocks.local_blocks.'):
                    new_k = k.replace('model.encoder.blocks.local_blocks.', 'blocks.blocks.')
                    model_ckpt[new_k] = v
                elif k.startswith('model.pos_embed'):
                    model_ckpt[k.replace('model.', '')] = v
                elif k.startswith('model.norm'):
                    model_ckpt[k.replace('model.', '')] = v
                elif k.startswith('model.embed.'):
                    new_k = k.replace('model.embed.', 'encoder.')
                    model_ckpt[new_k] = v

            incompatible = self.load_state_dict(model_ckpt, strict=False)

            if incompatible.missing_keys:
                print('missing_keys')
                print(
                    get_missing_parameters_message(incompatible.missing_keys)
                )
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(
                    get_unexpected_parameters_message(incompatible.unexpected_keys)

                )

            print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def load_model_from_ckpt(self, bert_ckpt_path: str | None, model_key='ACT_encoder'):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith(model_key):
                    base_ckpt[k[len(model_key + '.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print('missing_keys')
                print(
                        get_missing_parameters_message(incompatible.missing_keys)
                    )
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(
                        get_unexpected_parameters_message(incompatible.unexpected_keys)

                    )

            print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def load_model_from_ckpt_withrename(self, bert_ckpt_path:str):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)['model_state_dict']
            model_dict = self.state_dict()
            for k in list(model_dict.keys()):
                if k in ckpt:
                    model_dict[k] = ckpt[k]
                else:
                    old_k = k.replace("_cls", "")
                    print(old_k, k)
                    model_dict[k] = ckpt[old_k]

            incompatible = self.load_state_dict(model_dict, strict=False)

            if incompatible.missing_keys:
                print('missing_keys')
                print(
                        get_missing_parameters_message(incompatible.missing_keys)
                    )
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(
                        get_unexpected_parameters_message(incompatible.unexpected_keys)

                    )

            print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')