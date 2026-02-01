import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import io
from PIL import Image
from collections import Counter, defaultdict
import pandas as pd


# Определяем константы для цвета ограничивающих рамок и названий классов
class_to_color = {
    1: (89, 161, 197),
    2: (204, 79, 135),
    3: (125, 216, 93),
    4: (175, 203, 33),
}

class_to_name = {
    1: "enemy",
    2: "enemy-head",
    3: "friendly",
    4: "friendly-head"
}


# Вспомогательные функции для отрисовки данных
def add_bbox(image, box, label='', color=(128, 128, 128), txt_color=(0, 0, 0)):
    ''' box в формате [x_min, y_min, x_max, y_max] '''
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # расчет толщины линии рамки
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # толщина линии для отрисовки текста
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3  # координаты прямоугольника-подложки для текста
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(image,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), # Bottom-left corner of the text string in the image
                    fontFace=0,
                    fontScale=lw / 3,
                    color=txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)
    return image


def plot_examples(df, indices=None, num_examples=6, row_figsize=(12, 3)):
    '''
    df - набор данных
    indices - id изображений
    num_examples - число примеров для отрисовки
    '''
    if indices is None:
        indices = np.random.choice(len(df), size=num_examples, replace=False)  # если не задан id, то рандомные 6 примеров
    else:
        num_examples = len(indices)
    ncols = min(num_examples, 3)
    nrows = math.ceil(num_examples / 3)
    _, axes = plt.subplots(nrows, ncols, figsize=(row_figsize[0], row_figsize[1] * nrows), tight_layout=True)
    axes = np.array([axes]) if num_examples==1 else axes
    axes = axes.reshape(-1)
    for ix, ax in zip(indices, axes):
        row = df.iloc[ix]
        image = Image.open(io.BytesIO(row['image']['bytes']))
        bboxes = row["objects"]['bbox']
        bboxes_xyxy = np.array([[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in bboxes])
        classes = row["objects"]['category']
        img = np.array(image)
        for bbox, label in zip(bboxes_xyxy, classes):
            color = class_to_color[label]
            class_name = class_to_name[label]
            img = add_bbox(img, bbox, label=str(class_name), color=color)
        ax.imshow(img)
        ax.set_title(f"Image id: {row['image_id']}")
        ax.set_xticks([])
        ax.set_yticks([])


def plot_predictions(images, predictions, cell_figsize=(4, 4)):
    num_img = images.shape[0]
    ncols = min(num_img, 3)
    nrows = math.ceil(num_img / 3)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    _, axes = plt.subplots(nrows, ncols, figsize=(cell_figsize[0] * ncols, cell_figsize[1] * nrows), tight_layout=True)
    axes = np.array([axes]) if num_img==1 else axes
    axes = axes.reshape(-1)
    for idx, ax in zip(range(num_img), axes):
        img = images[idx].cpu().permute(1, 2, 0).contiguous().numpy()
        img = img * std + mean
        img = (img * 255).astype(np.uint8)
        preds = predictions[idx]
        for bbox, label, score in zip(preds["boxes"], preds["labels"], preds["scores"]):
            color = class_to_color[label+1]
            class_name = class_to_name[label+1]
            img = add_bbox(img, bbox, label=f"Class {class_name}: {score:.2f}", color=color)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    plt.close()


def part_info(df):
    """ Считаем статистики по датасету и рисуем распределение размеров ббоксов по классам. """
    df = pd.json_normalize(df['objects'])[["bbox", "category"]]
    obj_per_img = df['bbox'].apply(lambda obj: len(obj))
    print(f"Min bboxes per image: {obj_per_img.min()}")
    print(f"Max bboxes per image: {obj_per_img.max()}")
    print(f"Mean bboxes per image: {obj_per_img.mean():.3}")

    all_categories = df['category'].apply(lambda obj: obj).tolist()
    counter = Counter(np.concatenate(all_categories))
    msg = [f"{class_to_name[key]} : {value}" for key, value in counter.items()]
    print("\n\nNumber of object per class:\n" + "\n".join(msg))

    bboxes = defaultdict(list)
    for _, row in df.iterrows():
        for bb, cls in zip(row['bbox'], row['category']):
            bboxes[cls].append(list(bb[2:]))
    bboxes = dict(sorted(bboxes.items()))
    print("\nMean bbox size per class:")
    for cls, boxes_list in bboxes.items():
        print(f"{class_to_name[cls]} : {np.mean(boxes_list, axis=0)}")

    print("\n\n")
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    x_boxes = [np.array(val)[:, 0] for val in bboxes.values()]
    y_boxes = [np.array(val)[:, 1] for val in bboxes.values()]
    labels = [class_to_name[cls] for cls in class_to_name.keys()]
    colors = [class_to_color[cls] for cls in class_to_color.keys()]
    for ax, box, direction in zip(axes, [x_boxes, y_boxes], ["width", "hight"]):
        bplot = ax.boxplot(box, patch_artist=True, tick_labels=labels)
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(np.array(color) / 255)
        ax.set_ylabel(f"Bbox size by {direction}")
        ax.set_title(f"Bboxes distribution per class by {direction}")

