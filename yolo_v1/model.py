from __future__ import annotations

import torch
import torch.nn as nn

from yolo_v1.config import YoloConfig


class ConvBNLeaky(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int = 1, bn: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, k, stride, padding=k // 2, bias=not bn)
        ]
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


def _repeat_pairs(n: int, in_ch: int, mid_ch: int, out_ch: int, bn: bool) -> list[nn.Module]:
    layers: list[nn.Module] = []
    ch = in_ch
    for _ in range(n):
        layers.append(ConvBNLeaky(ch, mid_ch, 1, 1, bn))
        layers.append(ConvBNLeaky(mid_ch, out_ch, 3, 1, bn))
        ch = out_ch
    return layers


class YOLOv1(nn.Module):
    """Original YOLOv1: 24 conv + 2 FC, output S x S x (B*5+C)."""

    def __init__(
        self,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        bn: bool = True,
        fc_hidden: int = 4096,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.features = nn.Sequential(
            ConvBNLeaky(3, 64, 7, 2, bn),
            nn.MaxPool2d(2, 2),
            ConvBNLeaky(64, 192, 3, 1, bn),
            nn.MaxPool2d(2, 2),
            ConvBNLeaky(192, 128, 1, 1, bn),
            ConvBNLeaky(128, 256, 3, 1, bn),
            ConvBNLeaky(256, 256, 1, 1, bn),
            ConvBNLeaky(256, 512, 3, 1, bn),
            nn.MaxPool2d(2, 2),
            *_repeat_pairs(4, 512, 256, 512, bn),
            ConvBNLeaky(512, 512, 1, 1, bn),
            ConvBNLeaky(512, 1024, 3, 1, bn),
            nn.MaxPool2d(2, 2),
            *_repeat_pairs(2, 1024, 512, 1024, bn),
            ConvBNLeaky(1024, 1024, 3, 1, bn),
            ConvBNLeaky(1024, 1024, 3, 2, bn),
            ConvBNLeaky(1024, 1024, 3, 1, bn),
            ConvBNLeaky(1024, 1024, 3, 1, bn),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, fc_hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, S * S * (B * 5 + C)),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x.view(-1, self.S, self.S, self.B * 5 + self.C)


class YOLOv1Tiny(nn.Module):
    """Fast YOLO / tiny: 9 conv + 2 FC (arXiv:2305.17786 baseline)."""

    def __init__(
        self,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        bn: bool = True,
        fc_hidden: int = 2048,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        layers: list[nn.Module] = []
        ch = 3
        for out_ch in (16, 32, 64, 128, 256, 512):
            layers.append(ConvBNLeaky(ch, out_ch, 3, 1, bn))
            layers.append(nn.MaxPool2d(2, 2))
            ch = out_ch
        layers.extend(
            [
                ConvBNLeaky(512, 1024, 3, 1, bn),
                ConvBNLeaky(1024, 1024, 3, 1, bn),
                ConvBNLeaky(1024, 1024, 3, 1, bn),
            ]
        )
        self.features = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, fc_hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, S * S * (B * 5 + C)),
        )
        self._init_weights()

    _init_weights = YOLOv1._init_weights
    forward = YOLOv1.forward


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_RESNET_BUILDERS = {
    "resnet18": ("resnet18", "ResNet18_Weights", 512, 512),
    "resnet34": ("resnet34", "ResNet34_Weights", 512, 512),
    "resnet50": ("resnet50", "ResNet50_Weights", 2048, 1024),
}


class YOLOv1ResNet(nn.Module):
    """YOLOv1 7x7 grid head on an ImageNet-pretrained ResNet backbone."""

    def __init__(
        self,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        bn: bool = True,
        fc_hidden: int = 512,
        dropout: float = 0.5,
        pretrained: bool = True,
        backbone: str = "resnet18",
    ):
        super().__init__()
        del bn
        self.S, self.B, self.C = S, B, C
        key = backbone.lower()
        if key not in _RESNET_BUILDERS:
            raise ValueError(f"Unsupported backbone {backbone}")
        fn_name, weights_name, feat_ch, neck_ch = _RESNET_BUILDERS[key]
        import torchvision.models as tvm

        ctor = getattr(tvm, fn_name)
        weights_enum = getattr(tvm, weights_name)
        weights = weights_enum.IMAGENET1K_V1 if pretrained else None
        net = ctor(weights=weights)
        # 448 -> 14 x 14 x feat_ch
        self.backbone = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
        # 14 -> 7, then a compact detection FC (much smaller than original 4096-d)
        self.head = nn.Sequential(
            ConvBNLeaky(feat_ch, neck_ch, 3, stride=2, bn=True),
            ConvBNLeaky(neck_ch, neck_ch, 3, stride=1, bn=True),
            ConvBNLeaky(neck_ch, neck_ch, 3, stride=1, bn=True),
            nn.Flatten(),
            nn.Linear(S * S * neck_ch, fc_hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, S * S * (B * 5 + C)),
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        for module in self.head.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=0.1, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.backbone(x)
        x = self.head(x)
        return x.view(-1, self.S, self.S, self.B * 5 + self.C)


def build_model(cfg: YoloConfig) -> nn.Module:
    kwargs = dict(
        S=cfg.S,
        B=cfg.B,
        C=cfg.C,
        bn=cfg.batch_norm,
        fc_hidden=cfg.resolved_fc_hidden(),
        dropout=cfg.dropout,
    )
    name = cfg.model.lower()
    if name in {"yolov1", "full"}:
        return YOLOv1(**kwargs)
    if name in {"tiny", "yolov1-tiny", "yolov1_tiny"}:
        return YOLOv1Tiny(**kwargs)
    if name in _RESNET_BUILDERS or name in {"yolov1-r18", "r18", "yolov1-r50", "r50"}:
        alias = {
            "yolov1-r18": "resnet18",
            "r18": "resnet18",
            "yolov1-r50": "resnet50",
            "r50": "resnet50",
        }
        backbone = alias.get(name, name)
        default_fc = 1024 if backbone == "resnet50" else 512
        kwargs["fc_hidden"] = default_fc if cfg.fc_hidden is None else cfg.fc_hidden
        kwargs["pretrained"] = cfg.pretrained
        kwargs["backbone"] = backbone
        return YOLOv1ResNet(**kwargs)
    raise ValueError(f"Unknown model: {cfg.model}")
