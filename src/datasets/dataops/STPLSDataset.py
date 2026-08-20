from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

import torch
from tqdm import tqdm

from ..utils.preprocess import voxelize_fps, voxelize_only
from .base import PointCloudBaseDataset, Split, DSConfig

@dataclass(frozen=True)
class STPLSConfig(DSConfig):
    quantization: float
    num_sampling_points: int
    
class STPLS(PointCloudBaseDataset[STPLSConfig]):
    IGNORE_CLASS_ID = -100
    NAME = 'STPLS'
    def __init__(self, split: Split, config: STPLSConfig) -> None:
        super().__init__(split, config)
        
    def setup_files(self, split: Split, config: STPLSConfig) -> None:
        ds_path = Path(config.root_path)
        if split is Split.TRAIN:
            ds_path = ds_path.joinpath('train')
        elif split is Split.VAL:
            ds_path = ds_path.joinpath('val')
        else:
            ds_path = ds_path.joinpath('test')
            
        self._num_points = config.num_sampling_points
        self._quantization = config.quantization
        self._ds_dir=ds_path

    @property
    def ds_dir(self) -> Path:
        return self._ds_dir

    @property
    def num_points(self) -> int:
        return self._num_points

    @property
    def quantization(self) -> float:
        return self._quantization
    
    def get_mask_ds(self) -> list[tuple[NDArray[np.float32],NDArray[np.int8]]]:
        masks_ds: list[tuple[NDArray[np.float32],NDArray[np.int8]]] = []

        print('> Loading path files....')
        for file in tqdm(self.ds_dir.iterdir()):
            if file.suffix == '.pth':
                if self.split is Split.TEST:
                    pcd, _, = torch.load(file)
                    instances = np.zeros(pcd.shape[0])
                else:
                    pcd, _, _, instances = torch.load(file)

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

                for i, label in enumerate(labels):
                    indices = np.where(instances==label)
                    masks[indices, i] = 1
                    points_in_mask = pcd[indices]
                    if points_in_mask.shape[0] > self.MIN_NUMBER_OF_POINTS_PER_MASK:
                        masks_ds.append((pcd, masks[:,i].transpose()))

        return masks_ds
