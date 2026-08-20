from __future__ import annotations

import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler


__all__ = ['LayerNorm1d']

class LayerNorm1d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1) * x + self.bias.unsqueeze(-1)
        return x

def normalize_point_cloud(point_cloud: torch.Tensor) -> torch.Tensor:
    # Create a scaler object
    scaler = MinMaxScaler(feature_range=(0, 1), clip=True, copy=False)
    b,n,c = point_cloud.shape

    # Reshape the point cloud to (-1, channels) if needed
    # if point_cloud.dim() == 3:
    point_cloud = point_cloud.view(-1, c)

    # Convert to NumPy array
    point_cloud_np = point_cloud.detach().cpu().numpy()

    # Normalize the point cloud
    normalized_point_cloud_np = scaler.fit_transform(point_cloud_np)
    
    if np.isnan(normalized_point_cloud_np).any():
        raise ValueError("Normalized point cloud contains NaN values.")

    # Convert back to tensor
    normalized_point_cloud = torch.from_numpy(normalized_point_cloud_np).to(point_cloud.device)

    # Reshape back to the original shape
    # if point_cloud.dim() == 3:
    normalized_point_cloud = normalized_point_cloud.view(b, n, c)

    return normalized_point_cloud