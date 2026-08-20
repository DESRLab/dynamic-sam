from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np
from numpy.typing import NDArray

import torch
from torch.utils.data import Dataset

from ..utils.preprocess import augment

from .base import DSConfig, Split, DSOutput
from .ScannetDataset import Scannet, ScannetConfig
from .S3DISDataset import S3DIS, S3DISConfig, S3DISAreas
from .Kitti360 import Kitti360, Kitti360Config
from .STPLSDataset import STPLS, STPLSConfig
from .HM3D import HM3D, HM3DConfig
from .RScan import RScan, RScanConfig
from .ARKit import ARKit, ARKitConfig
from .MultiScan import MultiScan, MultiScanConfig

__all__ = ['MegaDataset', 'MegaConfig', 'SupportedDatasets']    

class SupportedDatasets(Enum):
    Kitti360 = Kitti360
    Scannet = Scannet
    S3DIS = S3DIS
    STPLS = STPLS
    HM3D = HM3D
    RScan = RScan
    ARKit= ARKit
    MultiScan = MultiScan

    @classmethod
    def default_val(cls) -> list[SupportedDatasets]:
        return [cls.S3DIS]
    
    @classmethod
    def default_train(cls) -> list[SupportedDatasets]:
        return [ds for ds in cls if ds not in cls.default_val()]


@dataclass(frozen=True)
class MegaConfig(DSConfig):
    quantization: float
    num_sampling_points: int
    train_ds: list[SupportedDatasets]
    val_ds: list[SupportedDatasets]

class MegaDataset(Dataset[DSOutput]):
    def __init__(self, split: Split, config: MegaConfig) -> None:
        super().__init__()
        qt = config.quantization
        num_points = config.num_sampling_points
        root = config.root_path

        configs_dict: dict[str, DSConfig] = {
            'Kitti360':  Kitti360Config(f'{root}/kitti360', False, qt, num_points, False),
            'S3DIS': S3DISConfig(f'{root}/s3dis', False, qt, num_points, S3DISAreas.A5),
            'STPLS': STPLSConfig(f'{root}/Synthetic_v3_InstanceSegmentation', False, qt, num_points),
            'Scannet': ScannetConfig(f'{root}/scannet', False, qt, num_points),
            'HM3D': HM3DConfig(f'{root}/SceneVerse/HM3D', False, qt, num_points),
            'RScan': RScanConfig(f'{root}/SceneVerse/3RScan', False, qt, num_points),
            'ARKit': ARKitConfig(f'{root}/SceneVerse/ARKitScenes', False, qt, num_points),
            'MultiScan': MultiScanConfig(f'{root}/SceneVerse/MultiScan', False, qt, num_points),
        }

        self._masks_ds = []

        # if split is Split.TRAIN:
        #     for cls_ds in config.train_ds:
        #         ds = cls_ds.value(split, configs_dict[cls_ds.value.NAME]) # type: ignore
        #         self._masks_ds.extend(ds._masks_ds)
        # else:
        for cls_ds in config.train_ds:
            ds = cls_ds.value(split, configs_dict[cls_ds.value.NAME]) # type: ignore
            self._masks_ds.extend(ds._masks_ds)       

        self._split = split
        
    @property
    def masks_ds(self) -> list[tuple[NDArray[np.float32], NDArray[np.int8]]]:
        return self._masks_ds

    @property
    def split(self) -> Split:
        return self._split

    def __len__(self) -> int:
        return len(self.masks_ds)

    def __getitem__(self, index:int) -> DSOutput:
        pcd, mask = self.masks_ds[index]
        if self.split is Split.TRAIN:
            pcd = augment(pcd)

        fg_indices = np.nonzero(mask==1)[0]
        fg_random_index = np.random.choice(fg_indices)
        fg_point = pcd[fg_random_index:fg_random_index+1,:]
        fg_label = np.array([1])

        return DSOutput(
            pcd=torch.from_numpy(pcd.astype(np.float32)).float(),
            point_prompt=torch.from_numpy(fg_point).to(torch.bfloat16),
            prompts_labels=torch.from_numpy(fg_label).to(torch.bfloat16),
            mask= torch.from_numpy(mask[None,:]),
            test_class_labels=[],
            test_class_labels_count={},
        )
