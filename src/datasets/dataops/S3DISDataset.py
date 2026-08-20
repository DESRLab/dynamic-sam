from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from plyfile import PlyData
import json

import torch
from tqdm import tqdm

from ..utils.preprocess import voxelize_fps, voxelize_only
from .base import PointCloudBaseDataset, Split, DSConfig, TestLabels

__all__ = ['S3DISAreas', 'S3DISConfig', 'S3DIS', 'S3DISV2']
class S3DISAreas(Enum):
    A1 = 'Area_1'
    A2 = 'Area_2'
    A3 = 'Area_3'
    A4 = 'Area_4'
    A5 = 'Area_5'
    A6 = 'Area_6'

    @classmethod
    def to_list(cls) -> list[str]:
        return [e.value for e in cls]

    @classmethod
    def to_list_except(cls, areas: list[S3DISAreas]) -> list[str]:
        return [e.value for e in cls if e is not areas]

@dataclass(frozen=True)
class S3DISConfig(DSConfig):
    quantization: float
    num_sampling_points: int
    val_area: S3DISAreas

class S3DIS(PointCloudBaseDataset[S3DISConfig]):
    NAME = 'S3DIS'
    def __init__(self, split: Split, config: S3DISConfig) -> None:
        super().__init__(split, config)
    
    def setup_files(self, split: Split, config: S3DISConfig) -> None:
        root = Path(config.root_path)
        ds_files: list[str] = []
        if split is Split.TRAIN:
            areas = S3DISAreas.to_list_except([S3DISAreas.A5, config.val_area])
            for area in areas:
                ds_files.extend([str(scene_path) for scene_path in Path(f'{root}/{area}').iterdir()])
        elif split is Split.VAL:
            ds_files = [str(scene_path) for scene_path in Path(f'{root}/{config.val_area.value}').iterdir()]
        elif split is Split.TEST:
            ds_files = [str(scene_path) for scene_path in Path(f'{root}/{S3DISAreas.A5.value}').iterdir()]
        else:
            areas = S3DISAreas.to_list()
            for area in areas:
                ds_files.extend([str(scene_path) for scene_path in Path(f'{root}/{area}').iterdir()])
        self._ds_files = ds_files
        self._num_points = config.num_sampling_points
        self._quantization = config.quantization

    @property
    def ds_files(self) -> list[str]:
        return self._ds_files

    @property
    def num_points(self) -> int:
        return self._num_points

    @property
    def quantization(self) -> float:
        return self._quantization

    def get_mask_ds(self) -> list[tuple[NDArray[np.float32],NDArray[np.int8]]]:
        masks_ds: list[tuple[NDArray[np.float32],NDArray[np.int8]]] = []

        print('> Loading path files....')

        for file in tqdm(self.ds_files):
            data = torch.load(file)
            pcd = data["coord"]
            assert isinstance(pcd, np.ndarray)
            
            instances = np.ones(pcd.shape[0], dtype=np.int8) * -1
            if "instance_gt" in data.keys():
                instances = data["instance_gt"].reshape([-1])

            assert isinstance(instances, np.ndarray)

            if self.voxelize_only:
                pcd, instances = voxelize_only(pcd.astype(np.float32), instances.astype(np.int64), self.quantization)
            else:
                pcd, instances = voxelize_fps(pcd.astype(np.float32), instances.astype(np.int64), self.num_points, self.quantization)

            labels = np.unique(instances)
            labels = np.delete(labels, np.where(labels==-1))
            num_points = pcd.shape[0]
            masks = np.zeros((num_points, labels.shape[0]), dtype=np.int8)

            for i, label in enumerate(labels):
                indices = np.where(instances==label)
                masks[indices, i] = 1
                points_in_mask = pcd[indices]
                if points_in_mask.shape[0] > self.MIN_NUMBER_OF_POINTS_PER_MASK:
                    masks_ds.append((pcd, masks[:,i].transpose()))

        return masks_ds
    
class S3DISV2(PointCloudBaseDataset[S3DISConfig]):
    IGNORE_CLASS_ID = 0
    def __init__(self, split: Split, config: S3DISConfig) -> None:
        super().__init__(split, config)
        
    def setup_files(self, split: Split, config: S3DISConfig) -> None:
        ds_path = Path(config.root_path)
        ds_files: list[str] = []

        if split is Split.TRAIN:
            with open(ds_path.joinpath('train_files.json')) as file:
                data = json.load(file)
                ds_files = data
        elif split is Split.VAL:
            with open(ds_path.joinpath('val_files.json')) as file:
                data = json.load(file)
                ds_files = data
        else:
            objs = np.load(f'{ds_path.parent}/object_ids.npy')
            classes = np.loadtxt(f'{ds_path.parent}/object_classes.txt', dtype=str)
            sort_indices = np.argsort(classes)
            objs = objs[sort_indices]
            self._label_classes = classes[sort_indices].tolist()
            ds_files = [f'{obj[0]}/{obj[0]}_crop_{obj[1]}.ply' for obj in objs]
        
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
    
    def get_test_class_labels(self) -> TestLabels:
        if self.split is Split.TEST:
            unique_strings, counts = np.unique(self._label_classes, return_counts=True)
            return TestLabels(test_class_labels=self._label_classes,test_class_labels_count=dict(zip(unique_strings, counts)))
        return super().get_test_class_labels()

    def get_mask_ds(self) -> list[tuple[NDArray[np.float32],NDArray[np.int8]]]:
        masks_ds: list[tuple[NDArray[np.float32],NDArray[np.int8]]] = []
        print('loading v2')
        update_label_classes = []

        for i, file in tqdm(enumerate(self.ds_files), total=len(self.ds_files)):
            plydata = PlyData.read(f'{self.root}/{file}')
            
            xyz = plydata['vertex'][['x', 'y', 'z']]
            assert isinstance(xyz, np.ndarray)

            pcd = np.zeros((xyz.shape[0], 3), dtype=np.float32)
            pcd[:,0] = xyz['x']
            pcd[:,1] = xyz['y']
            pcd[:,2] = xyz['z']
            
            instances = np.array(plydata['vertex']['label'], dtype=np.int64)

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
                masks_ds.append((pcd, masks[:,index].transpose()))
                if self.split is Split.TEST:
                    update_label_classes.append(self._label_classes[i])

        self._label_classes = update_label_classes
        return masks_ds