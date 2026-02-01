import torch
from torchvision.ops import box_iou


def TAL_assigner(pred_bbox, gt_boxes, pred_labels_logit, gt_labels, alpha=6.0, betta=1.0, top_k=13):
    """
    Task-Aligned Assigner (TAL) для детектора.

    Входные параметры
    -----------------
    pred_bbox : Tensor
        Предсказанные bounding boxes размерностью [N, 4] в формате (x_min, y_min, x_max, y_max).
    gt_boxes : Tensor
        Ground truth bounding boxes размерностью [M, 4].
    pred_labels_logit : Tensor
        Предсказанные метки классов (логиты) размерностью [N, C].
    gt_labels : Tensor
        Ground truth метки классов размерностью [M], zero-based.
    alpha : float, default=6.0
        Нормализационная константа для classification score в task-aligned metric.
    betta : float, default=1.0
        Нормализационная константа для IoU в task-aligned metric.
    top_k : int, default=13
        Мкасимальное число самых подходящих предсказаний на GT.

    Returns
    -------
    assigned_gt_bbox : Tensor
        Назначенные GT boxes для каждого предсказания [N, 4].
    assigned_gt_labels : Tensor
        One-hot кодировка назначенных GT меток классов для каждого предсказания [N, C].
    assigned_gt_conf : Tensor
        Маска позитивных предсказаний [N].
    """
    num_pred, num_classes = pred_labels_logit.shape
    num_gt = gt_boxes.shape[0]

    assigned_gt_bbox = torch.zeros((num_pred, 4), device=pred_bbox.device)
    assigned_gt_labels = torch.zeros((num_pred, num_classes), device=pred_bbox.device)
    assigned_gt_conf = torch.zeros(num_pred, device=pred_bbox.device)

    if num_gt == 0:
        # Нет GT, все предсказания - фон
        return [assigned_gt_bbox, assigned_gt_labels, assigned_gt_conf]

    u = box_iou(gt_boxes, pred_bbox)  # [M, N]

    pred_labels_prob = torch.sigmoid(pred_labels_logit)
    # pred_labels_prob = torch.softmax(pred_labels_logit, dim=-1)
    s = pred_labels_prob[:, gt_labels].T  # [M, N]

    pred_center_x = (pred_bbox[:, 0] + pred_bbox[:, 2]) / 2
    pred_center_y = (pred_bbox[:, 1] + pred_bbox[:, 3]) / 2
    gx_min, gy_min, gx_max, gy_max = gt_boxes.T

    center_mask = (
            (gx_min[:, None] < pred_center_x[None, :]) &
            (gx_max[:, None] > pred_center_x[None, :]) &
            (gy_min[:, None] < pred_center_y[None, :]) &
            (gy_max[:, None] > pred_center_y[None, :])
    ).float()

    t = (s ** alpha) * (u ** betta)
    t *= center_mask

    _, top_indices = torch.topk(t, min(top_k, num_pred), largest=True)  # [M, top_k]

    assigned_gt_idx = torch.full((num_pred,), -1, device=pred_bbox.device)  # для каждого предсказания индекс соотв. GT
    assigned_ious = torch.zeros(num_pred, device=pred_bbox.device)          # для каждого предсказания iou с соотв. GT

    for i in range(num_gt):
        not_zero_mask = t[i][top_indices[i]] > 0
        valid_indices = top_indices[i][not_zero_mask]

        for idx in valid_indices:
            if assigned_gt_idx[idx] == -1 or assigned_ious[idx] < u[i, idx]:
                assigned_gt_idx[idx] = i
                assigned_ious[idx] = u[i, idx]
                assigned_gt_labels[idx].zero_()
                assigned_gt_labels[idx][gt_labels[i]] = 1
                assigned_gt_bbox[idx] = gt_boxes[i]
    assigned_gt_conf[assigned_gt_idx != -1] = 1

    return [assigned_gt_bbox, assigned_gt_labels, assigned_gt_conf]


def TAL_assigner_batch(pred_bbox, gt_boxes, pred_labels_logit, gt_labels, alpha=6.0, betta=1.0, top_k=13):
    """
    Task-Aligned Assigner по батчу.

    Входные параметры
    -----------------
    pred_bbox : Tensor
        Предсказанные bounding boxes размерностью [N, 4] в формате (x_min, y_min, x_max, y_max).
    gt_boxes : Tensor
        Ground truth bounding boxes размерностью [M, 4].
    pred_labels_logit : Tensor
        Предсказанные метки классов (логиты) размерностью [N, C].
    gt_labels : Tensor
        Ground truth метки классов размерностью [M], zero-based.
    alpha : float, default=6.0
        Нормализационная константа для classification score в task-aligned metric.
    betta : float, default=1.0
        Нормализационная константа для IoU в task-aligned metric.
    top_k : int, default=13
        Мкасимальное число самых подходящих предсказаний на GT.

    Returns
    -------
    assigned_gt_bboxs : Tensor
        Назначенные GT boxes для каждого предсказания [B, N, 4].
    assigned_gt_labels : Tensor
        One-hot кодировка назначенных GT меток классов для каждого предсказания [B, N, C].
    assigned_gt_confs : Tensor
        Маска позитивных предсказаний [B, N].
    """

    batch_size = pred_labels_logit.shape[0]
    assigned_gt_bboxs = []
    assigned_gt_labels = []
    assigned_gt_confs = []

    for b in range(batch_size):
        assigned_gt = TAL_assigner(pred_bbox[b], gt_boxes[b], pred_labels_logit[b], gt_labels[b], alpha, betta, top_k)
        assigned_gt_bboxs.append(assigned_gt[0])
        assigned_gt_labels.append(assigned_gt[1])
        assigned_gt_confs.append(assigned_gt[2])

    return torch.stack(assigned_gt_bboxs), torch.stack(assigned_gt_labels), torch.stack(assigned_gt_confs)
