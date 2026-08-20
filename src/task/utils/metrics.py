from __future__ import annotations

import torch
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Optional
from typing_extensions import TypeAlias
from collections import defaultdict
from torch.utils.tensorboard.writer import SummaryWriter

from .misc import get_dims_with_exclusion

Tensor: TypeAlias = torch.Tensor

__all__ = ['EvalIoU', 'AdaptiveIoU', 'TrainMetric', 'NOCIoU', 'IoUNoC']

def compute_iou(pred_mask: Tensor, gt_mask: Tensor, ignore_mask: Optional[Tensor]=None, keep_ignore: bool=False) -> NDArray[np.float32]:
    gt_mask = gt_mask.to(pred_mask)
    if ignore_mask is not None:
        ignore_mask = ignore_mask.to(pred_mask)
        pred_mask = torch.where(ignore_mask, torch.zeros_like(pred_mask), pred_mask)

    reduction_dims = get_dims_with_exclusion(gt_mask.dim(), 0)
    union: NDArray = torch.mean((pred_mask | gt_mask).float(), dim=reduction_dims).detach().cpu().numpy()
    intersection: NDArray = torch.mean((pred_mask & gt_mask).float(), dim=reduction_dims).detach().cpu().numpy()
    nonzero = union > 0

    iou = (intersection[nonzero] / union[nonzero]).astype(np.float32)

    if not keep_ignore:
        return iou.astype(np.float32)
    else:
        result = np.full_like(intersection, -1).astype(np.float32)
        result[nonzero] = iou.astype(np.float32)
        return result

class TrainMetric:
    def update(self, *args, **kwargs):
        raise NotImplementedError

    def get_epoch_value(self):
        raise NotImplementedError

    def reset_epoch_stats(self):
        raise NotImplementedError

    def log_states(self, sw, tag_prefix, global_step):
        pass

    @property
    def name(self):
        return type(self).__name__


class AdaptiveIoU(TrainMetric):
    def __init__(self, init_thresh=0.45, thresh_step=0.025, thresh_beta=0.99, iou_beta=0.9,
                 ignore_label=-1):
        super().__init__()
        self._ignore_label = ignore_label
        self._iou_thresh = init_thresh
        self._thresh_step = thresh_step
        self._thresh_beta = thresh_beta
        self._iou_beta = iou_beta
        self._ema_iou = 0.0
        self._epoch_iou_sum = 0.0
        self._epoch_batch_count = 0

    @property
    def em_iou(self) -> float:
        return self._ema_iou

    def update(self, pred: Tensor, gt: Tensor):
        gt_mask_area = torch.sum(gt).detach().cpu().numpy()
        if np.all(gt_mask_area == 0):
            return

        ignore_mask = gt == self._ignore_label
        max_iou = compute_iou(pred > self._iou_thresh, gt).mean()
        best_thresh = self._iou_thresh
        for t in [best_thresh - self._thresh_step, best_thresh + self._thresh_step]:
            temp_iou = compute_iou(pred > t, gt, ignore_mask).mean()
            if temp_iou > max_iou:
                max_iou = temp_iou
                best_thresh = t

        self._iou_thresh = self._thresh_beta * self._iou_thresh + (1 - self._thresh_beta) * best_thresh
        self._ema_iou = self._iou_beta * self._ema_iou + (1 - self._iou_beta) * max_iou
        self._epoch_iou_sum += max_iou
        self._epoch_batch_count += 1

    def get_epoch_value(self):
        if self._epoch_batch_count > 0:
            return self._epoch_iou_sum / self._epoch_batch_count
        else:
            return 0.0

    def reset_epoch_stats(self):
        self._epoch_iou_sum = 0.0
        self._epoch_batch_count = 0

    def log_states(self, sw: SummaryWriter, tag_prefix: str, global_step: int):
        sw.add_scalar(tag=tag_prefix + '_ema_iou', scalar_value=self._ema_iou, global_step=global_step)
        sw.add_scalar(tag=tag_prefix + '_iou_thresh', scalar_value=self._iou_thresh, global_step=global_step)

    @property
    def iou_thresh(self):
        return self._iou_thresh

class EvalIoU(TrainMetric):
    def __init__(self, init_thresh=0.45, thresh_step=0.025, thresh_beta=0.99, iou_beta=0.9,
                 ignore_label=-1):
        super().__init__()
        self._ignore_label = ignore_label
        self._iou_thresh = init_thresh
        self._thresh_step = thresh_step
        self._thresh_beta = thresh_beta
        self._iou_beta = iou_beta
        self._epoch_iou_sum = 0.0
        self._epoch_batch_count = 0

    def update(self, pred: Tensor, gt: Tensor, threshold: float):
        self._iou_thresh = threshold
        gt_mask_area = torch.sum(gt).detach().cpu().numpy()
        if np.all(gt_mask_area == 0):
            return

        max_iou = compute_iou(pred > self._iou_thresh, gt).mean()
        self._iou_thresh = threshold
        best_thresh = self._iou_thresh
        for t in [best_thresh - self._thresh_step, best_thresh + self._thresh_step]:
            temp_iou = compute_iou(pred > t, gt).mean()
            if temp_iou > max_iou:
                max_iou = temp_iou
                best_thresh = t

        self._iou_thresh = self._thresh_beta * self._iou_thresh + (1 - self._thresh_beta) * best_thresh
        self._epoch_iou_sum += max_iou
        self._epoch_batch_count += 1

    def get_epoch_value(self):
        if self._epoch_batch_count > 0:
            return self._epoch_iou_sum / self._epoch_batch_count
        else:
            return 0.0

    def reset_epoch_stats(self):
        self._epoch_iou_sum = 0.0
        self._epoch_batch_count = 0

    def log_states(self, sw, tag_prefix, global_step):
        sw.add_scalar(tag=tag_prefix + '_iou_thresh', scalar_value=self._iou_thresh, global_step=global_step)

class IoUNoC:
    def __init__(self, max_nocs: list[int] = [5, 10, 15]) -> None:
        self._max_nocs = max_nocs
        self._iou_obj: list[dict[str, float]] = []
        self._all_iou: list[dict[str, float]] = []

    @property
    def max_nocs(self) -> list[int]:
        return self._max_nocs
    
    def _is_valid_noc(self, value: int) -> Optional[bool]:
        current_nocs = {iou_noc_dict['noc'] for iou_noc_dict in self._iou_obj}
        for noc in self.max_nocs:
            if noc not in current_nocs:
                if value == noc:
                    return True

    def update(self, noc: int, iou: float):
        valid_noc = self._is_valid_noc(noc)
        if valid_noc is not None:
            self._iou_obj.append({'noc': noc, 'iou': iou})

    def reset_obj(self):
        for iou_obj in self._iou_obj:
            self._all_iou.append(iou_obj)

        self._iou_obj = []


    def log_result(self, writer: SummaryWriter, step: int):
        grouped_noc = defaultdict(list)

        # Group the noc values by iou
        for entry in self._all_iou:
            iou_value = entry['iou']
            noc_value = entry['noc']
            grouped_noc[noc_value].append(iou_value)

        # Calculate the average noc value for each iou group
        average_iou_by_noc = {}
        for noc, iou in grouped_noc.items():
            average_iou = sum(iou) / len(iou)
            average_iou_by_noc[f'noc_{noc}'] = average_iou

        df = pd.DataFrame([average_iou_by_noc])
        print(df.head())
        writer.add_text('IoU@k Table', df.to_string(), global_step=step)

    
class NOCIoU:
    def __init__(self, max_ious: list[float] = [0.7, 0.8, 0.9], eps: float=0.1) -> None:
        self._max_ious = max_ious
        self._noc_obj: list[dict[str, float]] = []
        self._other_noc_obj: list[dict[str, float]] = []
        self._all_noc: list[dict[str, float]] = []
        self._eps = eps

    def _get_close_iou_to_value(self, value: float) -> Optional[float]:
        current_ious = {iou_noc_dict['iou'] for iou_noc_dict in self._noc_obj}
        for iou in self.max_ious:
            if iou not in current_ious:
                if value >= iou:
                    return iou

    @property
    def max_ious(self) -> list[float]:
        return self._max_ious

    def update(self, iou: float, noc: int) -> None:
        new_close_iou = self._get_close_iou_to_value(iou)
        if new_close_iou:
            self._noc_obj.append({'iou': new_close_iou, 'noc': noc})
        else:
            self._other_noc_obj.append({'iou': iou, 'noc': noc})

    def reset_obj(self):
        if len(self._noc_obj) < len(self.max_ious):
            self._other_noc_obj.sort(key=lambda x: (-x["iou"], x["noc"]))
            if len(self._noc_obj) == 0:
                current_ious = {}
            else:
                current_ious = {i['iou'] for i in self._noc_obj}
            # top_ranks = [i['noc'] for i in self._other_noc_obj[:len(self.max_ious)]]
            self._max_ious.sort()
            for i, iou_max in enumerate(self.max_ious):
                if iou_max not in current_ious:
                    self._all_noc.append({'iou': iou_max, 'noc': 20})

        if len(self._noc_obj) != 0:
            for noc_obj in self._noc_obj:
                self._all_noc.append(noc_obj)

        self._other_noc_obj = []
        self._noc_obj = []

    def log_result(self, writer: SummaryWriter, step: int):
        grouped_noc = defaultdict(list)

        # Group the noc values by iou
        for entry in self._all_noc:
            iou_value = entry['iou']
            noc_value = entry['noc']
            grouped_noc[iou_value].append(noc_value)

        # Calculate the average noc value for each iou group
        average_noc_by_iou = {}
        for iou, noc_values in grouped_noc.items():
            average_noc = sum(noc_values) / len(noc_values)
            average_noc_by_iou[f'iou_{iou}'] = average_noc

        df = pd.DataFrame([average_noc_by_iou])
        print(df.head())
        writer.add_text('NoC@q', df.to_string(), global_step=step)
        
def compute_bin_iou(pred: Tensor, targets: Tensor, eps: float = 1e-6) -> float:
    """
    Compute the Intersection over Union (IoU) between target masks and predicted masks.
    Args:
        targets: A binary tensor of shape (batch_size, num_point) representing the ground truth masks.
        predictions: A binary tensor of shape (batch_size, num_points) representing the predicted masks.
    Returns:
        A float tensor of shape (batch_size,) containing the IoU score for each sample in the batch.
    """
    pred_mask = pred == 1
    obj_gt_mask = targets == 1

    intersection = torch.logical_and(pred_mask, obj_gt_mask).sum()
    union = torch.logical_or(pred, obj_gt_mask).sum()

    return (intersection /  union ).item()


