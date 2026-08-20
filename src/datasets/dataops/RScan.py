from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import torch
from tqdm import tqdm

from ..utils.preprocess import voxelize_fps, voxelize_only
# from ..utils.misc import vis_ptcloud_with_instances
from .base import PointCloudBaseDataset, Split, DSConfig

# from open3d.visualization.tensorboard_plugin import summary  # noqa: F401
# from torch.utils.tensorboard.writer import SummaryWriter

__all__ = ['RScan', 'RScanConfig']

@dataclass(frozen=True)
class RScanConfig(DSConfig):
    quantization: float
    num_sampling_points: int

class RScan(PointCloudBaseDataset[RScanConfig]):
    IGNORE_CLASS_ID = 0
    NAME = 'RScan'
    
    def __init__(self, split: Split, config: RScanConfig) -> None:
        super().__init__(split, config)
        
    def setup_files(self, split: Split, config: RScanConfig) -> None:
        ds_path = Path(config.root_path)
        if split is Split.TRAIN:
            ds_files: list[str] = np.loadtxt(f'{ds_path}/annotations/splits/train_split_non_overlap.txt', dtype=str).tolist()   
        else:
            ds_files: list[str] = np.loadtxt(f'{ds_path}/annotations/splits/val_split_non_overlap.txt', dtype=str).tolist()
        
        self._ds_files = ds_files
        self._num_points = config.num_sampling_points
        self._quantization = config.quantization
        self._root = config.root_path
        
    @property
    def num_points(self) -> int:
        return self._num_points

    @property
    def quantization(self) -> float:
        return self._quantization
    
    @property
    def ds_files(self) -> list[str]:
        return self._ds_files
    
    @property
    def root(self) -> str:
        return self._root
  
    def get_mask_ds(self) -> list[tuple[NDArray[np.float32],NDArray[np.int8]]]:
        masks_ds: list[tuple[NDArray[np.float32],NDArray[np.int8]]] = []

        for file in tqdm(self.ds_files):
            file_path = f'{self.root}/scan_data/pcd_with_global_alignment/{file}.pth'
            if Path(file_path).exists():
                pcd, _, instances = torch.load(file_path)
                
                assert isinstance(pcd, np.ndarray)
                assert isinstance(instances, np.ndarray)

                if self.voxelize_only:
                    pcd, instances = voxelize_only(pcd.astype(np.float32), instances.astype(np.int64), self.quantization)
                else:
                    pcd, instances = voxelize_fps(pcd.astype(np.float32), instances.astype(np.int64), self.num_points, self.quantization)
    
                labels = np.unique(instances)
                labels = np.delete(labels, np.where(labels==self.IGNORE_CLASS_ID))
                num_points = pcd.shape[0]
                masks = np.zeros((num_points, labels.shape[0]), dtype=np.int8)

                for index, label in enumerate(labels):
                    indices = np.where(instances==label)
                    masks[indices, index] = 1
                    points_in_mask = pcd[indices]
                    if points_in_mask.shape[0] > self.MIN_NUMBER_OF_POINTS_PER_MASK:
                        masks_ds.append((pcd, masks[:,index].transpose()))
        return masks_ds