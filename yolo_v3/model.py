"""Pure PyTorch YOLOv3: Darknet-53 backbone + 3-scale detection head.

Architecture follows the YOLOv3 paper (Redmon & Farhadi, arXiv:1804.02767)
and the layer layout used by Darknet / Ultralytics YAML — implemented here
without importing Ultralytics model code.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from yolo_v3.config import ANCHORS_COCO, STRIDES, YoloV3Config


def autopad(k: int, p: int | None = None) -> int:
    return k // 2 if p is None else p


class ConvBNLeaky(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k), bias=False)
        self.bn = nn.BatchNorm2d(c2, momentum=0.03, eps=1e-4)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Darknet residual: 1x1 then 3x3, add shortcut."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = channels // 2
        self.block = nn.Sequential(
            ConvBNLeaky(channels, hidden, 1, 1),
            ConvBNLeaky(hidden, channels, 3, 1),
        )

    def forward(self, x):
        return x + self.block(x)


class Darknet53(nn.Module):
    """Darknet-53 feature extractor; returns P3, P4, P5 (stride 8/16/32)."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNLeaky(3, 32, 3, 1),
            ConvBNLeaky(32, 64, 3, 2),
        )
        self.stage1 = self._make_stage(64, 1)  # /2
        self.down2 = ConvBNLeaky(64, 128, 3, 2)
        self.stage2 = self._make_stage(128, 2)  # /4
        self.down3 = ConvBNLeaky(128, 256, 3, 2)
        self.stage3 = self._make_stage(256, 8)  # /8  -> P3
        self.down4 = ConvBNLeaky(256, 512, 3, 2)
        self.stage4 = self._make_stage(512, 8)  # /16 -> P4
        self.down5 = ConvBNLeaky(512, 1024, 3, 2)
        self.stage5 = self._make_stage(1024, 4)  # /32 -> P5

    @staticmethod
    def _make_stage(channels: int, n: int) -> nn.Sequential:
        return nn.Sequential(*[ResidualBlock(channels) for _ in range(n)])

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down2(x)
        x = self.stage2(x)
        x = self.down3(x)
        p3 = self.stage3(x)
        x = self.down4(p3)
        p4 = self.stage4(x)
        x = self.down5(p4)
        p5 = self.stage5(x)
        return p3, p4, p5


class DarknetTiny(nn.Module):
    """Compact backbone for YOLOv3-tiny (two detection scales)."""

    def __init__(self):
        super().__init__()
        self.features = nn.ModuleList(
            [
                ConvBNLeaky(3, 16, 3, 1),
                nn.MaxPool2d(2, 2),
                ConvBNLeaky(16, 32, 3, 1),
                nn.MaxPool2d(2, 2),
                ConvBNLeaky(32, 64, 3, 1),
                nn.MaxPool2d(2, 2),
                ConvBNLeaky(64, 128, 3, 1),
                nn.MaxPool2d(2, 2),
                ConvBNLeaky(128, 256, 3, 1),  # P4 /16 index 8
                nn.MaxPool2d(2, 2),
                ConvBNLeaky(256, 512, 3, 1),
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.MaxPool2d(2, 1),
                ConvBNLeaky(512, 1024, 3, 1),  # P5 /32
            ]
        )

    def forward(self, x):
        p4 = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 8:
                p4 = x
        return p4, x


class DetectionHead(nn.Module):
    """1x1 conv predicting na * (5 + nc) channels per spatial location."""

    def __init__(self, in_ch: int, num_anchors: int, num_classes: int):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.pred = nn.Conv2d(in_ch, num_anchors * (5 + num_classes), 1)

    def forward(self, x):
        return self.pred(x)


class YOLOv3(nn.Module):
    """Full YOLOv3: Darknet-53 + FPN-style upsampling head, 3 scales."""

    def __init__(self, num_classes: int = 80, anchors=ANCHORS_COCO, strides=STRIDES):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.register_buffer(
            "anchors",
            torch.tensor(anchors, dtype=torch.float32).view(3, -1, 2),
            persistent=False,
        )
        self.backbone = Darknet53()

        # P5 branch
        self.head5 = nn.Sequential(
            ConvBNLeaky(1024, 512, 1, 1),
            ConvBNLeaky(512, 1024, 3, 1),
            ConvBNLeaky(1024, 512, 1, 1),
            ConvBNLeaky(512, 1024, 3, 1),
            ConvBNLeaky(1024, 512, 1, 1),
        )
        self.conv5 = ConvBNLeaky(512, 1024, 3, 1)
        self.detect5 = DetectionHead(1024, 3, num_classes)

        # P4 branch
        self.lat5 = ConvBNLeaky(512, 256, 1, 1)
        self.up5 = nn.Upsample(scale_factor=2, mode="nearest")
        self.head4 = nn.Sequential(
            ConvBNLeaky(768, 256, 1, 1),
            ConvBNLeaky(256, 512, 3, 1),
            ConvBNLeaky(512, 256, 1, 1),
            ConvBNLeaky(256, 512, 3, 1),
            ConvBNLeaky(512, 256, 1, 1),
        )
        self.conv4 = ConvBNLeaky(256, 512, 3, 1)
        self.detect4 = DetectionHead(512, 3, num_classes)

        # P3 branch
        self.lat4 = ConvBNLeaky(256, 128, 1, 1)
        self.up4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.head3 = nn.Sequential(
            ConvBNLeaky(384, 128, 1, 1),
            ConvBNLeaky(128, 256, 3, 1),
            ConvBNLeaky(256, 128, 1, 1),
            ConvBNLeaky(128, 256, 3, 1),
            ConvBNLeaky(256, 128, 1, 1),
        )
        self.conv3 = ConvBNLeaky(128, 256, 3, 1)
        self.detect3 = DetectionHead(256, 3, num_classes)

        self._init_biases()

    def _init_biases(self):
        # Objectness prior ~0.01, class logits near zero (similar to YOLO practice)
        for det, stride in zip(
            (self.detect3, self.detect4, self.detect5), self.strides
        ):
            b = det.pred.bias.view(3, -1)
            with torch.no_grad():
                b[:, 4] = -4.5  # objectness
                b[:, 5:] = 0.0
            det.pred.bias = nn.Parameter(b.view(-1))

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)

        x5 = self.head5(p5)
        out5 = self.detect5(self.conv5(x5))

        x4 = self.up5(self.lat5(x5))
        x4 = torch.cat([x4, p4], dim=1)
        x4 = self.head4(x4)
        out4 = self.detect4(self.conv4(x4))

        x3 = self.up4(self.lat4(x4))
        x3 = torch.cat([x3, p3], dim=1)
        x3 = self.head3(x3)
        out3 = self.detect3(self.conv3(x3))

        # small -> large (stride 8, 16, 32)
        return out3, out4, out5


class YOLOv3Tiny(nn.Module):
    """YOLOv3-tiny: two scales (stride 16 and 32)."""

    def __init__(self, num_classes: int = 80):
        super().__init__()
        self.num_classes = num_classes
        self.strides = (16, 32)
        anchors = (((10, 14), (23, 27), (37, 58)), ((81, 82), (135, 169), (344, 319)))
        self.register_buffer(
            "anchors",
            torch.tensor(anchors, dtype=torch.float32).view(2, -1, 2),
            persistent=False,
        )
        self.backbone = DarknetTiny()
        self.conv5 = nn.Sequential(
            ConvBNLeaky(1024, 256, 1, 1),
            ConvBNLeaky(256, 512, 3, 1),
        )
        self.detect5 = DetectionHead(512, 3, num_classes)
        self.lat = ConvBNLeaky(256, 128, 1, 1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv4 = ConvBNLeaky(384, 256, 3, 1)
        self.detect4 = DetectionHead(256, 3, num_classes)

    def forward(self, x):
        p4, p5 = self.backbone(x)
        x5 = self.conv5[0](p5)
        out5 = self.detect5(self.conv5[1](x5))
        x4 = self.up(self.lat(x5))
        x4 = torch.cat([x4, p4], dim=1)
        out4 = self.detect4(self.conv4(x4))
        return out4, out5


def build_model(cfg: YoloV3Config) -> nn.Module:
    name = cfg.model.lower()
    if name in {"yolov3", "v3", "full"}:
        return YOLOv3(num_classes=cfg.num_classes, anchors=cfg.anchors, strides=cfg.strides)
    if name in {"tiny", "yolov3-tiny", "yolov3_tiny"}:
        return YOLOv3Tiny(num_classes=cfg.num_classes)
    raise ValueError(f"Unknown model: {cfg.model}")


def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
