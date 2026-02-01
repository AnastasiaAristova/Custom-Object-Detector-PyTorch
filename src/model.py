import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

def get_activation(name="silu", inplace=False):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported activation type: {}".format(name))
    return module


def get_bboxs_centers(size_h, size_w, stride, device):
    """
    Вычисляет координаты центров ячеек на изображении для FCOS формата.

    Параметры
    ---------
    size_h : int
        Высота карты призаков.
    size_w : int
        Ширина карты признаков.
    stride : int
        Коэффициент сокращеня карты признаков относительно входного изображения.
    device : torch.device

    Returns
    -------
    Tensor
        Тензор координат центров формой [H*W, 2].
    """
    ys, xs = torch.meshgrid(
        torch.arange(size_h, device=device),
        torch.arange(size_w, device=device),
        indexing="ij"
    )
    centers = torch.stack(
        [
            (xs + 0.5) * stride,
            (ys + 0.5) * stride
        ],
        dim=-1
    )
    return centers.view(-1, 2)


class Backbone(nn.Module):
    """
    Backbone feature extractor.
    Использует готовые модели из библиотеки timm с возможностью заморозки слоев.

    Входные параметры
    -----------------
    model_name : str, default="resnet50"
        Имя модели из библиотеки timm.
    unfreeze_last : int, default=0
        Число слоев для разморозки для дообучения.
    out_indices : tuple[int]
        Индексы карт признаков, которые возвращает backbone (FPN levels).
    """
    def __init__(self, model_name="resnet50", unfreeze_last=0, out_indices=(-1, -2, -3, -4)):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=out_indices)
        all_params = list(self.backbone.parameters())
        if unfreeze_last > 0:
            for param in all_params[:-unfreeze_last]:
                param.requires_grad = False
        else:
            for param in all_params:
                param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


class Neck(nn.Module):
    """
    Feature pyramid neck с top-down FPN и bottom-up PAN частями.

    Входные параметры
    -----------------
    in_channels : list[int]
        Число каналов карт признаков из backbone.
    out_channels : int
        Число каналов выходных карт признаков для всех уровней.
    use_activations : bool, default=True
        Использовать функцию активации ReLU после 1x1 Conv слоя для выходов из backbone.

    """
    def __init__(self, in_channels, out_channels, use_activations=True):
        super().__init__()
        # сверточные слои 1x1 для фичмапов (для скипконнекшена) из backbone (фичмапы: C5,C4,C3,C2)
        self.backbone_convs = nn.ModuleList()
        # сверточные слои между слоями части bottom-up (фичмапы: N2,N3,N4,N5,)
        self.bottom_up_convs = nn.ModuleList()
        for i, c in enumerate(in_channels):
            self.backbone_convs.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU() if use_activations else nn.Identity()
                )
            )
            if i > 0:
                self.bottom_up_convs.append(
                    nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU()
                    )
                )
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, features):
        C = list(reversed(features))  # [C2,C3,C4,C5,]->[C5,C4,C3,C2]
        P = []
        N = []
        # top-down process
        for i in range(len(C)):
            P.append(self.backbone_convs[i](C[i]))
            if i != 0:
                P[i] = P[i] + self.upsampling(P[i - 1])
        P.reverse()  # [P5,P4,P3,P2]->[P2,P3,P4,P5,]
        # botoom-up process
        for i in range(len(P)):
            N.append(P[i])
            if i > 0:  # N2=P2
                N[i] = N[i] + self.bottom_up_convs[i - 1](N[i - 1])

        # на выходе фичмапы c разных уровней
        return N  # [N2,N3,N4,N5]


class DetectionHead(nn.Module):
    """
    Голова модели. Структура основана на YOLOX decoupled head.

    Входные параметры
    -----------------
    in_channels : int
        Число каналов на входе.
    num_classes : int
        Число классов.
    activation_name : str, default="silu"
        Имя функции активации.

    Выходные параметры
    ------------------
    cls_preds : Tensor
        Предсказания классов (логиты) [B, num_classes, H, W].
    reg_preds : Tensor
        Bounding box offsets [B, 4, H, W].
    obj_preds : Tensor
        Предсказания уверенности модели (логиты) [B, 1, H, W].

    """
    def __init__(self, in_channels, num_classes, activation_name="silu"):
        super().__init__()

        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            get_activation(activation_name)
        ) if in_channels != 256 else nn.Identity()

        # Classification task
        self.cls_convs = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            get_activation(activation_name),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            get_activation(activation_name),
        )

        # Regression task
        self.reg_convs = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            get_activation(activation_name),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            get_activation(activation_name),
        )

        self.cls_head = nn.Conv2d(256, num_classes, kernel_size=1)
        self.reg_head = nn.Conv2d(256, 4, kernel_size=1)
        self.iou_head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv_reduce(x)
        cls = self.cls_convs(x)
        reg = self.reg_convs(x)
        cls_preds = self.cls_head(cls)
        reg_preds = self.reg_head(reg)
        obj_preds = self.iou_head(reg)
        return cls_preds, reg_preds, obj_preds


class Detector(nn.Module):
    """
    Anchor-free детектор с backbone, FPN/PAN шеей and decoupled heads.

    Входные параметры
    -----------------
    backbone_model_name : str
        Имя модели из библиотеки timm.
    neck_n_channels : int
        Количество каналов карт признаков для FPN/PAN.
    num_classes : int
        Число классов оъектов.
    input_size : tuple(int, int)
        Размеры входного изображения (H, W).
    unfreeze_last_backbone : int
        Число слоев для разморозки для дообучения.
    out_indices : tuple[int]
        Индексы карт признаков, которые возвращает backbone (FPN levels).
    act_head_name : str, default="silu"
        Имя функции активации.

    Выходные параметры
    ------------------
    bbox_preds : list[Tensor]
        Список тензоров предсказанных bounding box размера [B, N_i, 4] для каждого уровня.
    conf_logits : list[Tensor]
        Список тензоров предсказанных значений уверенности модели (логиты) [B, N_i, 1] для каждого уровня.
    cls_logits : list[Tensor]
        Список тензоров предсказанных классов (логиты) [B, N_i, C] для каждого уровня.

    """
    def __init__(self,
                 backbone_model_name="resnet50",
                 neck_n_channels=256,
                 num_classes=4,
                 input_size=(640, 640),
                 unfreeze_last_backbone=0,
                 out_indices=(-1, -2, -3, -4),
                 act_head_name="silu"
                 ):
        super().__init__()
        self.num_chanels = len(out_indices)
        self.num_classes = num_classes
        self.backbone = Backbone(backbone_model_name, unfreeze_last=unfreeze_last_backbone, out_indices=out_indices)
        self.neck = Neck(self.backbone.backbone.feature_info.channels(), out_channels=neck_n_channels)
        self.heads = nn.ModuleList(
            [DetectionHead(in_channels=neck_n_channels, num_classes=num_classes, activation_name=act_head_name) for _ in range(self.num_chanels)])
        self.strides = [r for r in reversed(
            self.backbone.backbone.feature_info.reduction())]  # выходы из бэкбоуна с верхнего, а из головы с нижнего, поэтому переворачиваем
        self.grid_sizes = [[input_size[0] // s, input_size[1] // s] for s in self.strides]

    def forward(self, x):
        features = self.backbone(x)
        neck_features = self.neck(features)  # [N2_out, N3_out, N4_out, N5_out,]
        cls_logits = [0] * self.num_chanels
        bbox_offsets = [0] * self.num_chanels
        conf_logits = [0] * self.num_chanels
        bbox_preds = [0] * self.num_chanels

        B = x.shape[0]
        for i, neck_f in enumerate(neck_features):
            cls_logits[i], bbox_offsets[i], conf_logits[i] = self.heads[i](neck_f)
            # Преобразуем сырые выходы модели в формат, в котором будет удобно считать лосс
            cls_logits[i] = cls_logits[i].permute(0, 2, 3, 1).contiguous()
            cls_logits[i] = cls_logits[i].view(B, -1, self.num_classes)         # [B, W * H, NUM_CLASSES]
            bbox_offsets[i] = bbox_offsets[i].permute(0, 2, 3, 1).contiguous()
            bbox_offsets[i] = bbox_offsets[i].view(B, -1, 4)                    # [B, W * H, 4]
            conf_logits[i] = conf_logits[i].permute(0, 2, 3, 1).contiguous()
            conf_logits[i] = conf_logits[i].view(B, -1, 1)                      # [B, W * H, 1]
            bbox_preds[i] = self.decode_bboxes_FCOS(bbox_offsets[i], self.grid_sizes[i] + [self.strides[i]])

        return bbox_preds, conf_logits, cls_logits

    def decode_bboxes_FCOS(self, bbox_offsets, sizes):
        """
        Декодирования bounding box в FCOS формате из предсказанных смещений в абсолютные координаты.

        Параметры
        ---------
        bbox_offsets: Tensor
            Предсказанные смещения [B, N, 4] в FCOS-формате (l, t, r, b).
        sizes: tuple
            (H, W, stride) карты признаков.

        Returns
        -------
        Tensor
            Bounding boxes [B, N, 4] в xyxy формате (x_min, y_min, x_max, y_max).
        """
        size_h, size_w, stride = sizes
        centers = get_bboxs_centers(size_h, size_w, stride, bbox_offsets.device).unsqueeze(0)
        xc = centers[:, :, 0]
        yc = centers[:, :, 1]

        l = F.relu(bbox_offsets[:, :, 0])
        t = F.relu(bbox_offsets[:, :, 1])
        r = F.relu(bbox_offsets[:, :, 2])
        b = F.relu(bbox_offsets[:, :, 3])

        x_min = xc - l
        y_min = yc - t
        x_max = xc + r
        y_max = yc + b

        return torch.stack([x_min, y_min, x_max, y_max], dim=-1)
