from __future__ import annotations

from collections.abc import Callable, Iterable

from dataclasses import dataclass

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from lightning.pytorch.loggers import TensorBoardLogger
# import matplotlib.pyplot as plt

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline
from scipy import ndimage

import torch
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler

from typing import Tuple, Optional, Dict, List
from typing_extensions import TypeAlias


from ..datasets.dataops import DSOutput, InferenceInput
from ..models import DynamicSAM, ModelOutPut


from open3d.visualization.tensorboard_plugin import summary  # noqa: F401
from torch.utils.tensorboard.writer import SummaryWriter

from .utils import (SetCriterion, TrainMetric, AdaptiveIoU, EvalIoU, LossOutput, NOCIoU, IoUNoC,  
                    Clicker, vis_ptcloud_with_masks_prompts_all, NoCIoUEvaluator, # noqa: F401
                    random_vis_ptcloud_with_masks_prompts, compute_iou)

import random

Tensor: TypeAlias = torch.Tensor

__all__ = ['SegmentPcdTask']

@dataclass(frozen=True)
class BatchForwardOutput():
    total_loss: Tensor
    model_output: ModelOutPut
    loss_output: LossOutput
    batch_data: DSOutput

@dataclass(frozen=True)
class TestForwardOutput():
    total_loss: Tensor
    model_output: ModelOutPut
    mIoU: Tensor

class SegmentPcdTask(pl.LightningModule):
    def __init__(
        self,
        network: DynamicSAM,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler],        
        pcd_encoder_pretrain_path: Optional[str] = None,
        pcd_dump_interval=200,
        max_num_next_clicks=8,
        warmup_iter: int = 10,
        init_lr: float = 0.0008,
        iou_threshold: float = 0.49,
        max_ious: list[float] = [0.8, 0.85, 0.9],
        max_nocs: list[int] = [5, 10, 15],
    ) -> None:
        super().__init__()

        self.network = torch.compile(network)
        self.network.pcd_encoder.load_model_from_ckpt(pcd_encoder_pretrain_path)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.iou_threshold = iou_threshold
        metrics: List[TrainMetric] = [AdaptiveIoU(init_thresh=iou_threshold)]
        self.train_metrics = metrics
        self.val_metrics = [EvalIoU(init_thresh=iou_threshold)]
        self.max_ious = max_ious
        self.noc_q_metric = NOCIoU(max_ious)
        self.iou_k_metric = IoUNoC(max_nocs)

        self.pcd_dump_interval = pcd_dump_interval

        self.criterion = SetCriterion()

        self.max_num_next_clicks = max_num_next_clicks
        self.warmup_iter = warmup_iter
        self.init_lr = init_lr
        self.test_class_labels = []
        self.test_class_count = {}

    def foward(self, batch_data: DSOutput):
        return self.network(batch_data)
        
    def logg_eval_at_epoch(self, validation: bool = False):
        metrics = self.val_metrics if validation else self.train_metrics
        prefix = 'Val' if validation else 'Train'

        if isinstance(self.logger, TensorBoardLogger):
            if isinstance(self.logger.experiment, SummaryWriter):
                for metric in metrics:
                    self.log(name=f'{prefix}_Metrics_{metric.name}', value=metric.get_epoch_value(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                    self.logger.experiment.add_scalar(
                        tag=f'{prefix}_Metrics_{metric.name}',
                        scalar_value=metric.get_epoch_value(),
                        global_step=self.global_step)

    def on_train_batch_start(self, batch: DSOutput, batch_idx: int) -> int | None:
        for metric in self.train_metrics:
            metric.reset_epoch_stats()

    def on_validation_batch_start(self, batch: DSOutput, batch_idx: int, dataloader_idx: int = 0) -> None:
        for metric in self.val_metrics:
            metric.reset_epoch_stats()
    
    def update_metric_at_step(self, outputs: ModelOutPut, batch_data: DSOutput, validation: bool = False) -> None:
        metrics = self.val_metrics if validation else self.train_metrics
        prefix = 'Val' if validation else 'Train'
        with torch.no_grad():
            for m in metrics:
                if isinstance(m, EvalIoU):
                    m.update(outputs['pred_mask'], batch_data['mask'], self.iou_threshold)
                else:
                    m.update(outputs['pred_mask'], batch_data['mask'])
                if isinstance(m, AdaptiveIoU):
                    self.iou_threshold = m.iou_thresh
                    if isinstance(self.logger, TensorBoardLogger):
                        if isinstance(self.logger.experiment, SummaryWriter):
                            m.log_states(self.logger.experiment, f'{prefix}_Metrics_{m.name}', self.global_step)

                self.log(name=f'{prefix}_Metrics_{m.name}', value=m.get_epoch_value(), on_epoch=True, on_step=True, prog_bar=True, logger=True, sync_dist=True)

    def logg_pcd_at_step(self, batch_data: DSOutput, model_output: ModelOutPut) -> None:
        if self.pcd_dump_interval > 0 and self.global_step % self.pcd_dump_interval == 0:
                sum_points = self.save_visualization(batch_data, model_output)
                if isinstance(self.logger, TensorBoardLogger):
                    if isinstance(self.logger.experiment, SummaryWriter):
                        for key, value in sum_points.items():
                            self.logger.experiment.add_3d(key, value, step=int(self.global_step/self.pcd_dump_interval), max_outputs=0) # type: ignore

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        if self.global_step < self.warmup_iter:
            lr = self.init_lr * (self.global_step + 1) / self.warmup_iter
            if isinstance(optimizer, AdamW):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        super().on_before_optimizer_step(optimizer)

    def training_step(self, batch_data: DSOutput, batch_idx: int) -> Tensor:
        forward_outputs = self.batch_forward(batch_data, validation=False)
        self.update_metric_at_step(forward_outputs.model_output, forward_outputs.batch_data)
        # self.logg_pcd_at_step(forward_outputs.batch_data, forward_outputs.model_output)
        loss_dicts = {f'Train_{k}': v.item() for k, v in forward_outputs.loss_output.items()} # type: ignore
        self.log_dict(loss_dicts, on_epoch=True, on_step=True, prog_bar=True, logger=True, sync_dist=True) 
        return forward_outputs.total_loss

    def validation_step(self, batch_data: DSOutput, batch_idx: int) -> Tensor:
        forward_outputs = self.batch_forward(batch_data, validation=True)
        self.update_metric_at_step(forward_outputs.model_output, forward_outputs.batch_data, validation=True)
        loss_dicts = {f'Val_{k}': v.item() for k, v in forward_outputs.loss_output.items()} # type: ignore
        self.log_dict(loss_dicts, on_epoch=True, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        return forward_outputs.total_loss

    def _find_optimal_threshold(self, logits: Tensor, gt_mask: Tensor, thresholds: list[float], batch_index:int, label: str, noc: int) -> float:
        ious: list[float] = []
        for threshold in thresholds:
            mIoU = compute_iou(logits > threshold, gt_mask)
            if mIoU.size > 0:
                ious.append(mIoU.mean())
            else:
                thresholds.remove(threshold)
            
        spline = make_interp_spline(thresholds, ious, k=5)
        threshold_grid = np.linspace(min(thresholds), max(thresholds), 100)
        ious_grid = spline(threshold_grid)
        optimal_threshold = threshold_grid[np.argmax(ious_grid)]
        
        # if batch_index == 18:
        #     plt.scatter(thresholds, ious, label="Data points")
        #     plt.plot(threshold_grid, ious_grid, label="Spline interpolation")
        #     plt.axvline(x=optimal_threshold, color='r', linestyle='--', label="Optimal threshold")

        #     plt.xlabel("Threshold")
        #     plt.ylabel("IoU")
        #     plt.title("Spline Interpolation of IoU vs Threshold")
        #     plt.legend()
        #     plt.savefig(f'{self.logger.experiment.log_dir}/{label}_{batch_index}_{noc}.png') # type: ignore
        #     plt.close()
        
        return optimal_threshold
    
    def forward_batch_test(self, batch_data: DSOutput, batch_idx: int, label_class: str) -> TestForwardOutput:
        pcd, gt_mask, points, labels = batch_data['pcd'], batch_data['mask'], batch_data['point_prompt'], batch_data['prompts_labels']


        self.network.eval()
        group_num = self.network.pcd_encoder.num_group
        group_size = int(pcd.shape[1]/group_num) + 1
        self.network.pcd_encoder.group_size = group_size
        self.network.set_pcd(pcd)

        noc_q_metric = self.noc_q_metric
        iou_k_metric = self.iou_k_metric
        noc = points.shape[1]

        clicker = Clicker(gt_mask[0], (points[0], labels[0]), pcd[0], self.device)

        step = 0

        optimal_threshold = self.iou_threshold
        thresholds: list[float] = ((np.arange(10,100, step=2)/100)).tolist()
        thresholds = sorted(list(set(thresholds)))
        with torch.no_grad():
            while noc <= self.max_num_next_clicks:

                model_output = self.network.forward_with_post_process(points, labels, optimal_threshold)
                best_mask = torch.zeros(size=gt_mask.shape)

                if model_output['pred_mask'].shape[1] > 1:
                    best_iou = 0
                    m_index = 0
                    for m in range(model_output['pred_mask'].shape[1]):
                        # temp_out = ModelOutPut(
                        #     pred_logits=model_output['pred_logits'][:,m,:].unsqueeze(1),
                        #     pred_mask=model_output['pred_mask'][:,m,:].unsqueeze(1),
                        #     binarized_mask=model_output['binarized_mask'][:,m,:].unsqueeze(1),
                        # )
                        threshold = self._find_optimal_threshold(model_output['pred_mask'][:,m,:].unsqueeze(1), gt_mask, thresholds, batch_idx, label_class, noc)

                        mIoU = compute_iou(model_output['pred_mask'][:,m,:].unsqueeze(1) > threshold, gt_mask).mean()
                        if mIoU:
                            if mIoU > best_iou:
                                best_iou = mIoU
                                m_index = m
                                optimal_threshold = threshold
                    best_mask = model_output['pred_mask'][:,m_index,:].unsqueeze(1)
                else:
                    best_mask = model_output['pred_mask']
                    optimal_threshold = self._find_optimal_threshold(model_output['pred_mask'], gt_mask, thresholds, batch_idx, label_class, noc)

                if batch_idx == 0 or batch_idx == 10:
                    viz_list = self.viz_test_examples(batch_data, model_output)
                    for viz_dict in viz_list:
                        if isinstance(self.logger, TensorBoardLogger):
                            if isinstance(self.logger.experiment, SummaryWriter):
                                    for key, value in viz_dict.items():
                                        self.logger.experiment.add_3d(key, value, step=step, max_outputs=0) # type: ignore
                step +=1
                mIoU = compute_iou(best_mask > optimal_threshold, gt_mask).mean()

                iou_k_metric.update(noc, mIoU)
                noc_q_metric.update(mIoU, noc)

                line = str(batch_idx) + ' ' + label_class + ' ' + str(noc) + ' ' + str(mIoU) + ' ' + str(optimal_threshold)+ '\n'
                self.test_result_file.write(line)
                clicker.make_next_click(best_mask[0] > optimal_threshold)
                points, labels = clicker.get_clicks()

                noc = points.shape[1]

                batch_data['point_prompt'] = points
                batch_data['prompts_labels'] = labels

                # step +=1
        batch_data['point_prompt'] = points
        batch_data['prompts_labels'] = labels

        outputs  = self.network.forward_with_post_process(batch_data['point_prompt'], batch_data['prompts_labels'], optimal_threshold)
        losses: LossOutput = self.criterion(model_output, batch_data)
        mIoU = compute_iou(model_output['pred_mask'] > optimal_threshold, gt_mask).mean()
        self.noc_q_metric.reset_obj()
        self.iou_k_metric.reset_obj()
        return TestForwardOutput(
            total_loss=losses['total'],
            model_output=outputs,
            mIoU=mIoU,
        )

    def test_step(self, batch_data: DSOutput, batch_idx: int) -> None:
        if batch_idx == 0:
            if isinstance(self.logger, TensorBoardLogger):
                if isinstance(self.logger.experiment, SummaryWriter):
                    self.test_result_file = open(f'{self.logger.experiment.log_dir}/val_res.csv', 'w')
            self.test_class_labels, self.test_class_labels_count = batch_data['test_class_labels'], batch_data['test_class_labels_count']
        
        if self.test_class_labels_count != {}: # type: ignore
            label_class = self.test_class_labels[batch_idx][0]
            self.test_class_labels_count[label_class] -= 1

            if self.test_class_labels_count[label_class] == 0:
                del self.test_class_labels_count[label_class]
        else:
            label_class = ''

        batch_outputs = self.forward_batch_test(batch_data, batch_idx, label_class)

        self.log('total_loss', batch_outputs.total_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('mIoU', batch_outputs.mIoU, on_epoch=True, on_step=True, prog_bar=True, logger=True, sync_dist=True)

        if isinstance(self.logger, TensorBoardLogger):
            if isinstance(self.logger.experiment, SummaryWriter):
                self.noc_q_metric.log_result(self.logger.experiment, step=0)
                self.iou_k_metric.log_result(self.logger.experiment, step=0)

        # if self.test_class_labels_count == {}:
        #     self.test_result_file.close()
        #     if isinstance(self.logger, TensorBoardLogger):
        #         if isinstance(self.logger.experiment, SummaryWriter):
        #             evaluator = NoCIoUEvaluator(f'{self.logger.experiment.log_dir}/val_res.csv', self.test_class_labels, self.max_ious)
        #             results = evaluator.eval_results()
        #             self.logger.experiment.add_text('NoC_IoU_Eval', results, global_step=0)

        # viz_list = self.viz_test_examples(batch_data, batch_outputs.model_output)
        # for viz_dict in viz_list:
        #     if isinstance(self.logger, TensorBoardLogger):
        #         if isinstance(self.logger.experiment, SummaryWriter):
        #             for key, value in viz_dict.items():
        #                 self.logger.experiment.add_3d(key, value, step=batch_idx, max_outputs=0) # type: ignore

    def predict_step(self, batch_data: InferenceInput) -> Tuple[Tensor, Tensor]:
        pcd, points, labels = batch_data['pcd'], batch_data['point_prompt'], batch_data['prompts_labels']
        self.network.set_pcd(pcd)

        return self.network(points, labels)

    def batch_forward(self, batch_data: DSOutput, validation: bool = False) -> BatchForwardOutput:
        with torch.set_grad_enabled(not validation):
            pcd, gt_mask, points, labels = batch_data['pcd'], batch_data['mask'], batch_data['point_prompt'], batch_data['prompts_labels']
            self.network.set_pcd(pcd)

            gt_mask_shp = gt_mask.shape
            with torch.no_grad():
                pairwise_fp_dist =np.zeros(shape=(gt_mask_shp[1], gt_mask_shp[2]), dtype=np.float64)
                pairwise_fn_dist =np.zeros(shape=(gt_mask_shp[1], gt_mask_shp[2]), dtype=np.float64)
                num_clicks = random.randint(0,self.max_num_next_clicks)
                for click_indx in range(num_clicks):
                    if not validation:
                        self.network.eval()

                    eval_model = self.network

                    model_output = eval_model.forward_with_post_process(points, labels, self.iou_threshold)

                    points, labels = self.get_next_points(pcd, model_output['pred_mask'], gt_mask, points, labels, click_indx + 1,  pairwise_fp_dist, pairwise_fn_dist)
 
                    if not validation:
                        self.network.train()

                if num_clicks == 0:
                    points = points[:,0,:].unsqueeze(1)
                    labels = labels[:,0].unsqueeze(1)

            batch_data['point_prompt'] = points
            batch_data['prompts_labels'] = labels
            outputs  = self.network.forward_with_post_process(points, labels, self.iou_threshold)
            losses: LossOutput = self.criterion(outputs, batch_data)

        return BatchForwardOutput(
            total_loss=losses['total'],
            loss_output=losses,
            model_output=outputs,
            batch_data=batch_data
        )

    def get_next_points(self, 
                        pcd: Tensor, pred:Tensor, 
                        gt: Tensor, points: Tensor, 
                        labels: Tensor, click_indx: int,
                        pairwise_fp_dist: NDArray[np.float64],
                        pairwise_fn_dist: NDArray[np.float64],
                        pred_thresh: float=0.49) -> Tuple[Tensor, Tensor]:
        assert click_indx > 0
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()
        b, _, c = pcd.shape
        num_points = points.shape[1]
        fn_mask = np.logical_and(gt, pred < pred_thresh)
        fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

        new_points = torch.zeros((b, num_points+1, c), device=self.device, dtype=torch.bfloat16)
        new_labels = torch.zeros((b, num_points+1), device=self.device, dtype=torch.bfloat16)

        for bindx in range(b):
            ndimage.distance_transform_edt(fn_mask[bindx], distances=pairwise_fn_dist)
            ndimage.distance_transform_edt(fp_mask[bindx], distances=pairwise_fp_dist)

            fn_max_dist = np.max(pairwise_fn_dist)
            fp_max_dist = np.max(pairwise_fp_dist)

            is_positive = fn_max_dist > fp_max_dist
            dt = pairwise_fn_dist if is_positive else pairwise_fp_dist

            inner_mask = dt >= max(fn_max_dist, fp_max_dist) / 2.0
            indices = np.argwhere(inner_mask)

            if len(indices) == 0:
                print('===================indices is 0==============')
                print(len(indices))
                indices = np.argwhere(np.logical_not(inner_mask))
                
            coords = indices[np.random.randint(0, len(indices))]
            new_sample = pcd[bindx, coords[1], :][None,:]

            new_label = torch.zeros(1).to(self.device)
            if is_positive:
                new_label = torch.ones(1).to(self.device)

            new_points[bindx] = torch.cat((points[bindx].to(self.device),new_sample.to(self.device)))
            new_labels[bindx] = torch.cat((labels[bindx].to(self.device), new_label))

        return new_points, new_labels

    def viz_test_examples(self, batch_data, outputs: ModelOutPut) -> List[Dict[str,Dict[str,NDArray]]]:
        pcd = batch_data['pcd']
        points = batch_data['point_prompt']
        instances = batch_data['mask']
        labels = batch_data['prompts_labels']

        gt_instance_masks = instances.cpu().numpy().copy()
        pred_mask = outputs['binarized_mask'].detach().cpu().numpy().copy()
        pred_logits = outputs['pred_mask'].detach().cpu().numpy().copy()
        pcd = pcd.cpu().numpy().copy()
        points = points.float().cpu().numpy().copy()
        labels = labels.int().cpu().numpy().copy()

        return vis_ptcloud_with_masks_prompts_all(pcd, points, labels, gt_instance_masks, pred_mask, pred_logits)

    def save_visualization(self, batch_data: DSOutput, outputs: ModelOutPut) -> Dict[str,Dict[str,NDArray]]:
        pcd = batch_data['pcd']
        points = batch_data['point_prompt']
        instances = batch_data['mask']
        labels = batch_data['prompts_labels']

        gt_instance_masks = instances.cpu().numpy().copy()
        pred_mask = outputs['binarized_mask'].detach().cpu().numpy().copy()
        pred_logits = outputs['pred_logits'].detach().cpu().numpy().copy()
        pcd = pcd.cpu().numpy().copy()
        points = points.float().cpu().numpy().copy()
        labels = labels.int().cpu().numpy().copy()

        return random_vis_ptcloud_with_masks_prompts(pcd, points, labels, gt_instance_masks, pred_mask, pred_logits)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)

        return OptimizerLRSchedulerConfig(optimizer=optimizer, lr_scheduler=scheduler)