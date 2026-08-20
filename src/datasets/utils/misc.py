from __future__ import annotations

import numpy as np
from  numpy.typing import NDArray

import open3d as o3d

from open3d.visualization.tensorboard_plugin import summary  # noqa: F401
from torch.utils.tensorboard.writer import SummaryWriter

def vis_ptcloud_with_instances(writer: SummaryWriter, step: int, ptcloud:NDArray[np.float32], instance_labels:NDArray[np.int64]):

    # Find unique instance labels
    unique_labels = np.unique(instance_labels)
    aggregated_colors = np.zeros((ptcloud.shape[0], 3))

    # Loop through the unique labels
    for label in unique_labels:
        # Index the point cloud to find points belonging to the instance object
        mask = instance_labels == label
        instance_points = ptcloud[mask]

        # Assign a random unique color to the points
        color = np.random.uniform(0, 1, size=(1, 3))

        # Create an Open3D color map for the instance points
        colors = np.tile(color, (instance_points.shape[0], 1))
        colors = o3d.utility.Vector3dVector(colors)

        # Assign the colors to the instance points in the point cloud
        aggregated_colors[mask] = color

    pcd_dict = {
            'vertex_positions': ptcloud,
            'vertex_colors': aggregated_colors,
        }
    writer.add_3d('pcd', pcd_dict, step=step) # type: ignore
    
def vis_ptcloud_with_masks(writer: SummaryWriter, step: int, ptcloud:NDArray[np.float32], masks:NDArray[np.int8]):

    # Find unique instance labels
    # unique_labels = np.unique(masks)
    aggregated_colors = np.zeros((ptcloud.shape[0], 3))

    # Loop through the unique labels
    for midx in range(masks.shape[0]):
        # Index the point cloud to find points belonging to the instance object
        indices = masks[midx,:] == 1
        instance_points = ptcloud[indices]

        # Assign a random unique color to the points
        color = np.random.uniform(0, 1, size=(1, 3))

        # Create an Open3D color map for the instance points
        colors = np.tile(color, (instance_points.shape[0], 1))
        colors = o3d.utility.Vector3dVector(colors)

        # Assign the colors to the instance points in the point cloud
        aggregated_colors[indices] = color

    pcd_dict = {
            'vertex_positions': ptcloud,
            'vertex_colors': aggregated_colors,
        }
    writer.add_3d('pcd', pcd_dict, step=step) # type: ignore