from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from src.kernel.fps import furthest_point_sample
from src.kernel.quantization import sparse_quantize
import torch
from typing import Tuple


def fps(data: NDArray[np.float32], num_samples: int) -> NDArray[np.int64]:
    '''
        data B N 3
        number int
    '''
    fps_idx: torch.Tensor = furthest_point_sample(torch.from_numpy(data[None, :, :]).contiguous().cuda(), num_samples)

    assert isinstance(fps_idx, torch.Tensor)

    return fps_idx.cpu().numpy()

def voxelize_pcd(pcd: NDArray[np.float32], quantization: float=0.05) -> NDArray[np.int64]:
    pcd_tensor = torch.from_numpy(pcd).contiguous().cuda()

    indices = sparse_quantize(pcd_tensor, quantization)

    assert isinstance(indices, torch.Tensor)

    return indices.cpu().numpy()

def randomly_sample_pcd(pcd: NDArray[np.float32], num_samples: int) -> NDArray[np.int64]:
    n = pcd.shape[0]
    if n < num_samples:
        indices = np.random.choice(n, num_samples - n, replace=True)
        indices = np.array(list(range(n)) + list(indices), dtype=np.int64)
    else:
        indices = np.random.choice(n, num_samples, replace=False)

    return indices

def voxelize_only(pcd: NDArray[np.float32], instances: NDArray[np.int64], quantization: float=0.1) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
    voxelized_indices = voxelize_pcd(pcd, quantization)
    pcd = pcd[voxelized_indices]
    instances = instances[voxelized_indices]
    return pcd, instances

def voxelize_fps(pcd: NDArray[np.float32], instances: NDArray[np.int64], num_points: int=2500, quantization: float=0.1) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
    voxelized_indices = voxelize_pcd(pcd, quantization)    
    pcd = pcd[voxelized_indices]
    instances = instances[voxelized_indices]

    fps_indices = fps(pcd, num_points)
    
    return pcd[fps_indices].transpose(1,2,0).squeeze(2), instances[fps_indices].transpose(1,0).squeeze(1)

def fps_only(pcd: NDArray[np.float32], instances: NDArray[np.int64], num_points: int=2500) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
    fps_indices = fps(pcd, num_points)
    
    return pcd[fps_indices].transpose(1,2,0).squeeze(2), instances[fps_indices].transpose(1,0).squeeze(1)

def voxelize_downsample(pcd: NDArray[np.float32], instances: NDArray[np.int64], num_points: int=2500, quantization: float=0.1) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
    voxelized_indices = voxelize_pcd(pcd, quantization)
    pcd = pcd[voxelized_indices]
    instances = instances[voxelized_indices]

    random_indices = randomly_sample_pcd(pcd, num_points)
    
    return pcd[random_indices], instances[random_indices]


def rotz(t: int):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def augment(point_cloud: NDArray[np.float32]) -> NDArray[np.float32]:
    if np.random.random() > 0.5:
        # Flipping along the YZ plane
        point_cloud[:, 0] = -1 * point_cloud[:, 0]

    if np.random.random() > 0.5:
        # Flipping along the XZ plane
        point_cloud[:, 1] = -1 * point_cloud[:, 1]

    # Rotation along up-axis/Z-axis
    rot_angle_pre = np.random.choice([0, np.pi/2, np.pi, np.pi/2*3])
    rot_mat_pre = rotz(rot_angle_pre)
    point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat_pre))

    rot_angle = (np.random.random() * 2* np.pi) - np.pi  # -180 ~ +180 degree
    rot_mat = rotz(rot_angle)
    point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))


    return point_cloud

