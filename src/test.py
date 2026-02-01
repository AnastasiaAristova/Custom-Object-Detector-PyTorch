import torch
from tqdm.auto import tqdm
from torchmetrics.detection import MeanAveragePrecision
from filters import filter_prediction_nms


def predict(model, images, device, score_threshold=0.1, iou_threshold=0.5):
    """
    Предсказание моделью для переданного набора изображений после фильтрации по score_threshold
    и применения NMS.

    Параметры
    ---------
    images : torch.tensor
        Изображения для которых нужно сделать предсказание.
    score_threshold : float, default = 0.1
        Все предсказания, с (confidence score * cls_probs) < score_threshold будут проигнорированны.
    iou_threshold : float, default = 0.5
        Предсказания, имеющие пересечение по IoU >= iou_threshold будут считаться одним предсказанием.

    Returns
    -------
    final_predictions : List[dict], где
        Содержит:
        - "boxes" : координаты bounding boxes изображении,
        - "labels" : классы объектов,
        - "scores" : confidence scores.
    """
    model.eval()
    with torch.set_grad_enabled(False):
        images = images.to(device)
        bbox_preds, conf_logit, cls_logit = model(images)
        all_bbox_preds = torch.cat(bbox_preds, dim=1)
        all_conf_logit = torch.cat(conf_logit, dim=1)
        all_cls_logit = torch.cat(cls_logit, dim=1)
        final_predictions = filter_prediction_nms([all_bbox_preds, all_conf_logit, all_cls_logit],
                                                  score_threshold=score_threshold,
                                                  iou_threshold=iou_threshold)
    return final_predictions


def validate(model, dataloader, device="cpu", score_threshold=0.1, iou_threshold=0.5):
    """
    Валидация на тестовой выборке

     Параметры
     ---------
     model : nn.Module
         Модель детектора.
    dataloader : DataLoader, optional
        Даталоадер для валидации.
    device : str, default = "cpu"
    score_threshold : float, default = 0.1
        Порог confidence score для NMS.
    iou_threshold : float, default = 0.5
        IoU threshold для NMS.

     Returns
     -------
     final_predictions : List[dict], где
         Содержит:
         - "boxes" : координаты bounding boxes изображении,
         - "labels" : классы объектов,
         - "scores" : confidence scores.
     """
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    for imgs, targets in tqdm(dataloader, desc="Running validation", leave=False):
        with torch.set_grad_enabled(False):
            imgs = imgs.to(device)
            bbox_preds, conf_logit, cls_logit = model(imgs)
            all_bbox_preds = torch.cat(bbox_preds, dim=1)
            all_conf_logit = torch.cat(conf_logit, dim=1)
            all_cls_logit = torch.cat(cls_logit, dim=1)
            predictions = filter_prediction_nms([all_bbox_preds, all_conf_logit, all_cls_logit],
                                                score_threshold=score_threshold,
                                                iou_threshold=iou_threshold,
                                                return_type="torch")
            metric.update(predictions, targets)
    result_metric = metric.compute()
    return result_metric["map"].item(), result_metric




