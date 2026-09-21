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
    raise ValueError(f"Unknown model: {cfg.model}")
