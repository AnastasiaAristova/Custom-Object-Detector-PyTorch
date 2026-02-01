
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import io
import pandas as pd
from albumentations.pytorch.transforms import ToTensorV2

DATA_MODES = ['train', 'validation', 'test']


class HaloDataset(Dataset):
    """
    PyTorch датасет для задачи детекции

    Входные параметры
    -----------------
    dataframe : pd.DataFrame
        Содержит:
        - image_id: уникальный идентификатор изображения,
        - objects: в json формате:
            - bbox: список bounding boxes в формате [x_min, y_min, w, h],
            - category: список меток объектов (индексация с 1),
        - image: в json формате байты изображения.
    mode : str, default="train"
        Тип датасета: "train", "validation", or "test".
    transform : callable, опционально
        Аугментация для изображений и bounding boxes.
    """
    def __init__(self, dataframe, mode="train", transform=None):
        super().__init__()
        self.data = pd.concat([dataframe[["image_id"]], pd.json_normalize(dataframe['objects'])[["bbox", "category"]],
                               pd.json_normalize(dataframe['image'])[["bytes"]]], axis=1)
        self.transform = transform
        self.mode = mode
        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct. Correct modes: {DATA_MODES}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """

        Параметры
        ---------
        index : int
            индекс экземпляра.

        Returns
        -------
        image : Tensor
            Изображение в формате тензора
        target : dict, опционально
            Возвращается, если mode != "test". Содержит:
            - "boxes": Tensor[N, 4] в формате (x_min, y_min, x_max, y_max),
            - "labels": Tensor[N] метки классов объектов (индексация с 0),
            - "image_id": идентификатор изображения.
        """
        row = self.data.iloc[index]
        image = Image.open(io.BytesIO(row["bytes"]))
        image = np.array(image)

        if self.mode == "test":
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]
            else:
                image = ToTensorV2()(image)
            return image

        target = {"image_id": row["image_id"]}

        labels = row["category"]
        labels = labels - 1

        bboxes = row["bbox"]
        bboxes_xyxy = np.array([[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in bboxes])

        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=bboxes_xyxy, labels=labels)
            image, bboxes_xyxy, labels = transformed["image"], transformed["bboxes"], transformed["labels"]
        else:
            image = ToTensorV2()(image)

        target["boxes"] = torch.tensor(bboxes_xyxy, dtype=torch.float32)
        target["labels"] = torch.tensor(labels, dtype=torch.int64)

        return image, target
