import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou


class FocalLoss(nn.Module):
    """
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t), с использованием BCE with logits.

    Входные параметры
    -----------------
    alpha : float, default=1.0
        Балансировочный параметр.
    gamma : float, default=2.0
        Фокусировочный параметр для регулировки влияния хорошо определяемых примеров.
    reduction : {'mean', 'sum', 'none'}, default='mean'
        Вид возращаемого значения.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * ((1 - pt) ** self.gamma) * BCE_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class DistanceIoULoss(nn.Module):
    """
    Distance-IoU loss for bounding box regression.

    Входные параметры
    -----------------
    reduction : {'mean', 'sum', 'none'}, default='mean'
        Вид возращаемого значения.
    """
    def __init__(self, reduction='mean'):
        super(DistanceIoULoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred_bbox, gt_boxes):
        """
        Параметры
        ---------
        pred_bbox : Tensor
            Предсказанные bounding boxes размерностью [N, 4].
        gt_boxes : Tensor
            Ground truth bounding boxes размерностью [N, 4].

        Returns
        -------
        loss : Tensor
            Distance-IoU loss.
        """
        iou = box_iou(gt_boxes, pred_bbox).diag()
        center_pred_x = (pred_bbox[:, 0] + pred_bbox[:, 2]) / 2
        center_pred_y = (pred_bbox[:, 1] + pred_bbox[:, 3]) / 2
        center_gt_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        center_gt_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        d = (center_pred_x - center_gt_x) ** 2 + (center_pred_y - center_gt_y) ** 2
        x_c_min = torch.min(gt_boxes[:, 0], pred_bbox[:, 0])
        y_c_min = torch.min(gt_boxes[:, 1], pred_bbox[:, 1])
        x_c_max = torch.max(gt_boxes[:, 2], pred_bbox[:, 2])
        y_c_max = torch.max(gt_boxes[:, 3], pred_bbox[:, 3])
        c = (x_c_max - x_c_min) ** 2 + (y_c_max - y_c_min) ** 2 + 1e-7
        diou_loss = 1 - iou + d / c
        if self.reduction == 'mean':
            return torch.mean(diou_loss)
        elif self.reduction == 'sum':
            return torch.sum(diou_loss)
        else:
            return diou_loss


class ComputeLoss:
    """ Базовый расчет лосса.

    Параметры
    ---------
    bbox_loss : nn.Module, optional
        Функция потерь для bounding box regression.
    obj_loss : nn.Module, optional
        Функция потерь для objectness score.
    cls_loss : nn.Module, optional
        Функция потерь для classification.
    weight_bbox : float, default=5
        Вес bounding box loss.
    weight_obj : float, default=1
        Вес objectness loss.
    weight_cls : float, default=1
        Вес classification loss.
    """

    def __init__(self,
                 bbox_loss=None, obj_loss=None, cls_loss=None,
                 weight_bbox=5, weight_obj=1, weight_cls=1 #7.5 1 0.5
                 ):
        self.bbox_loss = nn.SmoothL1Loss() if bbox_loss is None else bbox_loss
        self.obj_loss = nn.BCEWithLogitsLoss() if obj_loss is None else obj_loss
        self.cls_loss = nn.BCEWithLogitsLoss() if cls_loss is None else cls_loss
        self.weight_bbox = weight_bbox
        self.weight_obj = weight_obj
        self.weight_cls = weight_cls

    def __call__(self, predicts, targets):
        """
        Расчет лосса для пары (предсказание, таргет)

        Параметры
        ---------
        predicts : Предсказания модели для одной картинки: Смещения, objectness score и логиты для классов
        targets : Gt значения для расчета лосса, а именно: GT смещения, GT objectness score и GT ohe классы
        """
        pred_bbox, pred_conf_logits, pred_cls_logits = predicts
        gt_bbox, gt_labels, gt_conf = targets
        pos_mask = gt_conf == 1
        # Confidence score считается для предсказаний соотв отрицательным и положительным предсказаниям
        loss_obj = self.obj_loss(pred_conf_logits, gt_conf.unsqueeze(dim=-1))
        # Локализационная и классификационные части считаются только для предсказаинй соотв положительным предсказаниям
        if pos_mask.sum() > 0:
            loss_cls = self.cls_loss(pred_cls_logits[pos_mask], gt_labels[pos_mask])
            loss_bbox = self.bbox_loss(pred_bbox[pos_mask], gt_bbox[pos_mask])
        else:
            loss_cls = torch.tensor(0.0, device=pred_bbox.device)
            loss_bbox = torch.tensor(0.0, device=pred_bbox.device)
        return self.weight_bbox * loss_bbox + self.weight_obj * loss_obj + self.weight_cls * loss_cls
