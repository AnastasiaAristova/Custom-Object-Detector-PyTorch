import torch
from torchvision.ops import nms


def filter_prediction_nms(predictions, score_threshold=0.3, iou_threshold=0.5, return_type="list"):
    """
    NMS - метод фильтрации предсказаний детектора

    Параметры
    ---------
    predictions : list[Tensors]
        Предсказания модели для батча: bboxes, objectness логиты и логиты для классов
    score_threshold : float, default = 0.3
        Все предсказания, с (confidence score * cls_probs) < score_threshold будут проигнорированны.
    iou_threshold : float, default = 0.5
        Предсказания, имеющие пересечение по IoU >= iou_threshold будут считаться одним предсказанием.
    return_type : str, default = "list"
        Тип возвращаемых данных "list" или "torch".

    Returns
    -------
    final_predictions : List[dict] или Tensor[dict], где
        Содержит:
        - "boxes" : координаты bounding boxes изображении,
        - "labels" : классы объектов,
        - "scores" : confidence scores.
    """
    bbox_preds, conf_logit, cls_logit = predictions
    confidence = torch.sigmoid(conf_logit)
    # cls_probs = torch.softmax(cls_logit, dim=-1)
    cls_probs = torch.sigmoid(cls_logit)
    final_score = confidence * cls_probs  # [B, N, num_cls]
    batch_size, _, num_cls = cls_probs.shape
    final_preds = []
    # цикл по батчу - выбираем предсказания для каждого изображения
    for b in range(batch_size):
        filter_preds = {"boxes": [], "labels": [], "scores": []}
        # цикл по классу - выбираем предсказания для каждого класса на изображении
        for cls in range(num_cls):
            class_score = final_score[b, :, cls]
            mask_score_threshold = class_score > score_threshold
            if mask_score_threshold.sum() == 0:
                continue
            class_score = class_score[mask_score_threshold]
            class_bboxes = bbox_preds[b, mask_score_threshold, :]
            predict_idx = nms(class_bboxes, class_score, iou_threshold)
            for idx in predict_idx:
                filter_preds["boxes"].append(class_bboxes[idx].cpu().tolist())
                filter_preds["labels"].append(cls)
                filter_preds["scores"].append(class_score[idx].item())
        if return_type == "torch":
            for key, item in filter_preds.items():
                filter_preds[key] = torch.tensor(item)
        elif return_type != "list":
            raise ValueError(f"Received unexpected `return_type`. Could be either `torch` or `list`, not {return_type}")
        final_preds.append(filter_preds)
    return final_preds