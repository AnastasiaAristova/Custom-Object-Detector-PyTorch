# Custom-Object-Detector-PyTorch

Реализация **кастомной модели детекции объектов** с нуля, включая:

- Кастомный Dataset для набора данных halo-infinite-angel-videogame
- Backbone (timm) с разморозкой слоев
- FPN-like Neck (Top-Down + Bottom-Up path)
- Anchor-Free Detection Heads (FCOS-style)
- TAL (Task-Aligned Assigner)
- Кастомные функции потерь (Focal Loss, Distance IoU Loss)
- Полный training pipeline (Runner)
- Валидацию и инференс
- Визуализацию предсказаний

Формат проекта: Исследовательский, реализованный в Jupyter Notebook (notebooks/training.ipynb) с модульной организацией кода в папке src/. Обучение, валидация и инференс производятся через ячейки ноутбука.

---

## Архитектура модели

**Backbone:**  
Используется pretrained backbone из `timm` (по умолчанию ResNet50).

**Neck:**  
FPN-подобная архитектура с:

- Top-Down path
- Bottom-Up path
- Nearest-neighbor upsampling
- 1×1 и 3×3 convolution layers с ReLU

**Detection Head:**  
Отдельные heads для:

- Classification
- Bounding box regression
- Objectness (IoU prediction)

**Anchor-Free формат:**  
Боксы предсказываются в FCOS-формате относительно центров ячеек feature maps.

---

## Assignment Strategy

Используется **Task-Aligned Assigner (TAL)**:

- Совмещает IoU и classification score
- Учитывает center sampling

---

## Функции потерь

Реализованы кастомные лоссы:

- **Focal Loss** — для objectness
- **Distance IoU Loss (DIoU)** — для регрессии bbox
- **ComputeLoss** — агрегатор loss components

---

## Эксперименты

### Обучение

Параметры:

- Optimizer: Adam

- LR: 1e-3

- Scheduler: CosineAnnealingLR

- Epochs: 30 (+10 fine-tuning)

- Input size: 640×640


### Результаты на валидации

| Epochs | mAP |
|--------|------|
| 5  | 0.092 |
| 10 | 0.173 |
| 15 | 0.181 |
| 20 | 0.215 |
| 25 | 0.238 |
| 30 | **0.264** |

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/9639b393-fbfe-4736-b83a-d7f95c27ff44" />

Рисунок: Динамика функции потерь (Train Loss) и точности (mAP) на валидации в процессе обучения модели за первые 30 эпох.

- Train Loss — синий график - кривая функции потерь на каждом батче обучения, оранжевый - усреднённая по эпохам кривая. Наблюдается устойчивое снижение loss с ≈6.5 до ≈1.5 на протяжении 30 эпох, что говорит о том, что модель обучается.

- Validation mAP — нижний график показывает изменение mean Average Precision (mAP) на валидационной выборке. Наблюдается рост mAP от ≈9% до ≈27% к 30-й эпохе. 

### Результаты на тестовом датасете

mAP: 0.2429

mAP50: 0.6187

mAP75: 0.1638


Дополнительное обучение (ещё 10 эпох) привело к снижению mAP → наблюдается **переобучение**.

---


## Future Work

В дальнейшем планируются следующие экcперименты с целью повышения качества модели:

- Использовать другой backbones и/или дообучать его последние слои 

- Упростить структуру шеи (только Top-Down part)

- Использовать GIoU/CIoU loss

- Выполнить подбор гиперпараметров 

---

## Структура проекта

.

├── notebooks/

│ └── training.ipynb          # **Основной ноутбук** с полным пайплайном: обучение, валидация, инференс, визуализация

│

├── src/                        # Вспомогательные модули (импортируются в ноутбук)

│   ├── model.py                # Архитектура модели (Backbone, Neck, Detection Heads, Detector)

│   ├── assigners.py            # Task-Aligned Assigner (TAL)

│   ├── losses.py               # Функции потерь (Focal Loss, DIoU Loss)

│   ├── datasets.py             # Кастомный датасет

│   ├── filters.py              # NMS и постобработка

│   ├── train.py                # Класс Runner для обучения модели

│   ├── test.py                 # Функции валидации и предсказания на тестовой выборке

│   └── visualization.py        # Функции для визуализации предсказаний

│

├── README.md

├── requirements.txt

└── .gitignore


