import torch
from tqdm.auto import tqdm
import gc
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.detection import MeanAveragePrecision
from assigners import TAL_assigner_batch
from filters import filter_prediction_nms


class Runner:
    """
    Класс обучения и валидации модели детектора.
    Включает в себя:
    - цикл обучения (train loop)
    - расчет лосса
    - запуск валидции
    - логирование метрик и лоссов
    - визуализацию кривых обучения

    Входные параметры
    -----------------
    model : nn.Module
        Модель детектора.
    train_dataloader : DataLoader
        Даталоадер для обучения.
    criterion : callable
        Функция/класс для расчета лосса.
    optimizer : torch.optim.Optimizer
        Оптимизатор.
    device : str, default = "cpu"
    scheduler : torch.optim.lr_scheduler, optional
        Планировщик learning rate.
    val_dataloader : DataLoader, optional
        Даталоадер для валидации.
    score_threshold : float, default = 0.1
        Порог confidence score для NMS.
    iou_threshold : float, default = 0.5
        IoU threshold для NMS.
    val_freq : int, default = 5
        Частота запуска валидации (каждые val_freq эпох).
    """
    def __init__(self, model, train_dataloader, criterion, optimizer,
                 device="cpu", scheduler=None, val_dataloader=None,
                 score_threshold=0.1, iou_threshold=0.5, val_freq=5):
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.val_freq = val_freq

        # Логи обучения
        self.batch_loss = []  # лосс по батчам
        self.epoch_loss = []  # средний лосс по эпохам
        self.val_metric = []  # mAP на валидации

    def _train_epoch(self):
        """Обучение модели на одной эпохе."""
        self.model.train()
        batch_loss = []
        for imgs, targets in (pbar := tqdm(self.train_dataloader, desc=f"Process train epoch", leave=False)):
            self.optimizer.zero_grad()

            imgs = imgs.to(self.device)
            bbox_preds, conf_logit, cls_logit = self.model(imgs)

            batch_size = len(targets)
            num_level = len(bbox_preds)
            level_loss = []
            # Цикл по уровням пирамиды
            for level in range(num_level):
                gt_boxes = [targets[b]['boxes'].to(self.device) for b in range(batch_size)]
                gt_labels = [targets[b]['labels'].to(self.device) for b in range(batch_size)]
                assigned_targets = TAL_assigner_batch(bbox_preds[level], gt_boxes, cls_logit[level], gt_labels)
                level_loss.append(self.criterion([bbox_preds[level], conf_logit[level], cls_logit[level]], assigned_targets))
            loss = torch.stack(level_loss).mean()
            batch_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()
        return batch_loss

    def validate(self, dataloader=None, score_threshold=None, iou_threshold=None):
        """Валидация модели и расчет mAP."""
        self.model.eval()
        dataloader = self.val_dataloader if dataloader is None else dataloader
        score_threshold = self.score_threshold if score_threshold is None else score_threshold
        iou_threshold = self.iou_threshold if iou_threshold is None else iou_threshold
        metric = MeanAveragePrecision(iou_type="bbox")
        for imgs, targets in tqdm(dataloader, desc="Running validation", leave=False):
            with torch.set_grad_enabled(False):
                imgs = imgs.to(self.device)
                bbox_preds, conf_logit, cls_logit = self.model(imgs)
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

    def train(self, num_epochs=10, verbose=True):
        """Основной цикл обучения."""
        val_desc = ""
        for epoch in (epoch_pbar := tqdm(range(1, num_epochs+1), desc="Train epoch", total=num_epochs)):
            loss = self._train_epoch()
            self.batch_loss.extend(loss)
            self.epoch_loss.append(np.mean(self.batch_loss[-len(self.train_dataloader):]))

            if self.val_dataloader is not None and epoch % self.val_freq == 0:
                val_metric, _ = self.validate()
                self.val_metric.append(val_metric)
                val_desc = f" Val metric {val_metric:.4}"

            if verbose:
                epoch_pbar.set_description(f"Last epoch: Train loss {self.epoch_loss[-1]:.4}" + val_desc)

            if self.scheduler is not None:
                self.scheduler.step()

            torch.cuda.empty_cache()
            gc.collect()

    def plot_loss(self, row_figsize=3):
        """Визуализация кривых обучения и mAP."""
        nrows = 2 if self.val_metric else 1
        _, ax = plt.subplots(nrows, 1, figsize=(12, row_figsize*nrows), tight_layout=True)
        ax = np.array([ax]) if not isinstance(ax, np.ndarray) else ax
        ax[0].plot(self.batch_loss, label="Train batch Loss", color="tab:blue")
        ax[0].plot(np.arange(1, len(self.batch_loss)+1, len(self.train_dataloader)), self.epoch_loss,
                   color="tab:orange", label="Train epoch Loss")
        ax[0].grid()
        ax[0].set_title("Train Loss")
        ax[0].set_xlabel("Number of Iterations")
        ax[0].set_ylabel("Loss")
        if self.val_metric:
            ax[1].plot(np.arange(self.val_freq, len(self.epoch_loss)+1, self.val_freq),
                       np.array(self.val_metric) * 100, color="tab:green", label="Validation mAP")
            ax[1].grid()
            ax[1].set_title("Valiation mAP")
            ax[1].set_xlabel("Number of Iterations")
            ax[1].set_ylabel("mAP (%)")
        plt.legend()
        plt.show()


