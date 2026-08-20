from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from plyfile import PlyData
import numpy as np
from numpy.typing import NDArray
import json

from tqdm import tqdm

from ..utils.preprocess import voxelize_fps, voxelize_only
from .base import PointCloudBaseDataset, Split, DSConfig, TestLabels

__all__ = ['Kitti360', 'Kitti360V2']

@dataclass(frozen=True)
class Kitti360Config(DSConfig):
    quantization: float
    num_sampling_points: int
    dynamic: bool


class Kitti360(PointCloudBaseDataset[Kitti360Config]):
    NAME = 'Kitti360'
    def __init__(self, split: Split, config: Kitti360Config) -> None:
        super().__init__(split, config)

    def setup_files(self, split: Split, config: Kitti360Config) -> None:
        root = config.root_path
        if split is Split.TRAIN:
            self._ds_files: list[str] = np.loadtxt(f'{root}/2013_05_28_drive_train.txt', dtype=str).tolist()
        elif split is Split.VAL:
            self._ds_files: list[str] = np.loadtxt(f'{root}/2013_05_28_drive_val.txt', dtype=str).tolist()
        else:
            folder_1 = Path(f'{root}/data_3d_semantics/test/2013_05_28_drive_0008_sync/static')
            folder_2 = Path(f'{root}/data_3d_semantics/test/2013_05_28_drive_0018_sync/static')
            self._ds_files =[str(file) for file in folder_1.iterdir() if file.is_file()]
            self._ds_files.extend([str(file) for file in folder_2.iterdir() if file.is_file()])

        self._num_points = config.num_sampling_points
        self._quantization = config.quantization
        self._dynamic = config.dynamic
        self._root = config.root_path
            
    @property
    def ds_files(self) -> list[str]:
        return self._ds_files
    
    @property
    def num_points(self) -> int:
        return self._num_points

    @property
    def quantization(self) -> float:
        return self._quantization

    @property
    def dynamic(self) -> bool:
        return self._dynamic
    
    @property
    def root(self) -> str:
        return self._root

    def get_mask_ds(self) -> list[tuple[NDArray[np.float32],NDArray[np.int8]]]:
        masks_ds: list[tuple[NDArray[np.float32],NDArray[np.int8]]] = []

        for file in tqdm(self.ds_files):
            if self.split is Split.TRAIN or self.split is Split.VAL:
                file = f'{self.root}/{file}'
                if self.dynamic:
                    file = file.replace('static', 'dynamic')

            plydata = PlyData.read(file)
            if self.split is Split.TEST:
                filtered_indices = np.where(plydata['vertex']['visible'] == 1)
            else:
                filtered_indices =  np.where((plydata['vertex']['visible'] == 1) & (plydata['vertex']['confidence'] > 0.9))

            if filtered_indices[0].shape[0] == 0:
                continue

            xyz = plydata['vertex'][filtered_indices][['x', 'y', 'z']]
            assert isinstance(xyz, np.ndarray)

            pcd = np.zeros((xyz.shape[0], 3), dtype=np.float32)
            pcd[:,0] = xyz['x']
            pcd[:,1] = xyz['y']
            pcd[:,2] = xyz['z']

            if self.split is Split.TEST:
                instances = np.zeros(pcd.shape, dtype=np.int64)
            else:
                instances = np.array(plydata['vertex'][filtered_indices]['instance'], dtype=np.int64)
 

            if self.voxelize_only:
                pcd, instances = voxelize_only(pcd.astype(np.float32), instances.astype(np.int64), self.quantization)
            else:
                pcd, instances = voxelize_fps(pcd.astype(np.float32), instances.astype(np.int64), self.num_points, self.quantization)

            labels = np.unique(instances)
            num_points = pcd.shape[0]
            masks = np.zeros((num_points, labels.shape[0]), dtype=np.int8)

            for i, label in enumerate(labels):
                indices = np.where(instances==label)
                masks[indices, i] = 1
                masks_ds.append((pcd, masks[:,i].transpose()))

        return masks_ds

class Kitti360V2(PointCloudBaseDataset[Kitti360Config]):
    IGNORE_CLASS_ID = 0
    def __init__(self, split: Split, config: Kitti360Config) -> None:
        super().__init__(split, config)
        
    def setup_files(self, split: Split, config: Kitti360Config) -> None:
        ds_path = Path(config.root_path)
        ds_files: list[str] = []

        if split is Split.TRAIN:
            with open(ds_path.joinpath('train_files.json')) as file:
                data = json.load(file)
                ds_files = data
        elif split is Split.VAL:
            with open(ds_path.joinpath('val_list.json')) as file:
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

            if pcd.shape[0] < self.num_points:
                continue
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