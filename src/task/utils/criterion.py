
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import TypedDict, Optional
from typing_extensions import TypeAlias

from ...models import ModelOutPut
from ...datasets.dataops import DSOutput

Tensor: TypeAlias = torch.Tensor

__all__ =['SetCriterion', 'LossOutput']

class LossOutput(TypedDict):
    dice: Tensor
    bce: Tensor
    total: Tensor

def loss_weights(batch: DSOutput, alpha: float=0.8, beta: float=2.0, tita: float=0.3) -> Tensor:
    """Points closer to clicks have bigger weights. Vice versa.
    """
    batch_size = batch['pcd'].shape[0]
    batch_weights: list[Tensor] = []
    for b in range(batch_size):
        pcd_coords = batch['pcd'][b]
        labels = batch['prompts_labels'][b]
        prompts = batch['point_prompt'][b]
        positive_clicks = prompts[labels==1]

        pairwise_distances = torch.cdist(pcd_coords, positive_clicks.float())
        pairwise_distances, _ = torch.min(pairwise_distances, dim=1)

        weights = alpha + (beta-alpha) * (1 - torch.clamp(pairwise_distances, max=tita)/tita)
        batch_weights.append(weights)

    return torch.stack(batch_weights).to(batch['pcd'].device)

def dice_loss(inputs: Tensor, targets: Tensor, weights: Optional[Tensor]=None, eps: float = 1e-6) -> Tensor:
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape of batch_size,num_mask,num_points.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        weights: A float tensor with the same shape as targets.
                represents loss weight for each points with shape of batch_size,num_points.
    """
    inputs = inputs.sigmoid()
    intersect = inputs * targets

    numerator = 2.0 * (intersect).mean(1, keepdim=True)
    denominator = (inputs + targets).mean(1, keepdim=True)
    soft_iou = (numerator + eps) / (denominator + eps)

    return (torch.where(numerator > eps, 1. - soft_iou, soft_iou * 0.) * weights).mean()

def binary_cross_entropy_loss(inputs: Tensor, targets: Tensor, weights:Optional[Tensor]=None, alpha: float = -1, gamma: float = 2, reduction: str = 'None') -> Tensor:
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy(prob, targets.float(), reduction="none")

    if weights is not None:
        loss = ce_loss * weights

    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


class SetCriterion(nn.Module):
    def __init__(self, cost_focal: float = 10, cost_dice: float = 1):
        super().__init__()
        self.cost_focal = cost_focal
        self.cost_dice = cost_dice

    def forward(self, outputs: ModelOutPut, inputs: DSOutput) -> LossOutput:
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        targets = inputs['mask'].clone()
        src_masks = outputs['pred_logits']
        target_masks  = targets.to(src_masks.device).transpose(2,1).squeeze(2)
        ls_weights = loss_weights(inputs)

        total = torch.zeros(size=(1,0))
        if src_masks.shape[1] > 1:    
            lowest_loss = torch.tensor(10)
            for i in range(src_masks.shape[1]):
                bce = binary_cross_entropy_loss(src_masks[:,i,:] , target_masks, weights=ls_weights, reduction='mean')
                dice = dice_loss(src_masks[:,i,:], target_masks, weights=ls_weights)

                loss = (self.cost_focal * bce) + (self.cost_dice * dice)
                if loss < lowest_loss:
                    lowest_loss = loss
            total = lowest_loss
        else:
            src_masks = src_masks.transpose(2,1).squeeze(2)
            bce = binary_cross_entropy_loss(src_masks , target_masks, weights=ls_weights, reduction='mean')
            dice = dice_loss(src_masks, target_masks, weights=ls_weights)

            total = (self.cost_focal * bce) + (self.cost_dice * dice)

        losses = LossOutput(dice=dice, 
                            bce=bce,
                            total= total)
        return losses

