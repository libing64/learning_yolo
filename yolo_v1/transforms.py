from __future__ import annotations

import random

import torch
from PIL import Image, ImageFilter, ImageOps

from yolo_v1.boxes import clip_xywh_normalized, xywh_to_xyxy, xyxy_to_xywh


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes):
        for t in self.transforms:
            image, boxes = t(image, boxes)
        return image, boxes


class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.5, saturation=0.7, hue=0.07):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, boxes):
        from torchvision.transforms import functional as F

        image = F.adjust_brightness(image, 1.0 + random.uniform(-self.brightness, self.brightness))
        image = F.adjust_contrast(image, 1.0 + random.uniform(-self.contrast, self.contrast))
        image = F.adjust_saturation(image, 1.0 + random.uniform(-self.saturation, self.saturation))
        image = F.adjust_hue(image, random.uniform(-self.hue, self.hue))
        return image, boxes


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, boxes):
        if random.random() < self.p:
            image = ImageOps.mirror(image)
            if boxes.numel():
                boxes = boxes.clone()
                boxes[:, 1] = 1.0 - boxes[:, 1]
        return image, boxes


class RandomVerticalFlip:
    def __init__(self, p=0.05):
        self.p = p

    def __call__(self, image, boxes):
        if random.random() < self.p:
            image = ImageOps.flip(image)
            if boxes.numel():
                boxes = boxes.clone()
                boxes[:, 2] = 1.0 - boxes[:, 2]
        return image, boxes


class RandomGrayscale:
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, image, boxes):
        if random.random() < self.p:
            image = ImageOps.grayscale(image).convert("RGB")
        return image, boxes


class RandomBlur:
    def __init__(self, p=0.1, radius=(0.1, 2.0)):
        self.p = p
        self.radius = radius

    def __call__(self, image, boxes):
        if random.random() < self.p:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.radius)))
        return image, boxes


class RandomRotationJitter:
    """Small rotation; boxes become axis-aligned envelopes of rotated corners."""

    def __init__(self, p=0.5, degrees=10.0):
        self.p = p
        self.degrees = degrees

    def __call__(self, image, boxes):
        if random.random() >= self.p:
            return image, boxes
        angle = random.uniform(-self.degrees, self.degrees)
        image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
        if boxes.numel() == 0:
            return image, boxes
        rad = torch.deg2rad(torch.tensor(angle))
        # PIL rotates counter-clockwise around the image center; y is down so
        # the 2D rotation that matches the pixel grid uses a negated angle.
        c, s = torch.cos(-rad), torch.sin(-rad)
        xyxy = xywh_to_xyxy(boxes[:, 1:5])
        corners = torch.stack(
            [
                xyxy[:, [0, 1]],
                xyxy[:, [2, 1]],
                xyxy[:, [2, 3]],
                xyxy[:, [0, 3]],
            ],
            dim=1,
        )
        corners = corners - 0.5
        rot = torch.tensor([[c, -s], [s, c]], dtype=corners.dtype)
        corners = corners @ rot.T + 0.5
        x1 = corners[..., 0].min(dim=1).values
        y1 = corners[..., 1].min(dim=1).values
        x2 = corners[..., 0].max(dim=1).values
        y2 = corners[..., 1].max(dim=1).values
        xywh = xyxy_to_xywh(torch.stack([x1, y1, x2, y2], dim=1))
        boxes = torch.cat([boxes[:, :1], xywh], dim=1)
        boxes = clip_xywh_normalized(boxes)
        return image, boxes


class YOLOJitter:
    """Random scale / aspect / translation similar to original YOLO jitter=0.2."""

    def __init__(self, jitter=0.2, scale=(0.8, 1.2), fill=(128, 128, 128)):
        self.jitter = jitter
        self.scale = scale
        self.fill = fill

    def __call__(self, image, boxes):
        w, h = image.size
        dw = self.jitter * w
        dh = self.jitter * h
        new_ar = (w + random.uniform(-dw, dw)) / (h + random.uniform(-dh, dh) + 1e-6)
        scale = random.uniform(*self.scale)
        if new_ar < 1:
            nh = max(1, int(scale * h))
            nw = max(1, int(nh * new_ar))
        else:
            nw = max(1, int(scale * w))
            nh = max(1, int(nw / new_ar))

        resized = image.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (w, h), self.fill)
        dx = int(random.uniform(0, w - nw)) if nw < w else int(random.uniform(w - nw, 0))
        dy = int(random.uniform(0, h - nh)) if nh < h else int(random.uniform(h - nh, 0))
        canvas.paste(resized, (dx, dy))

        if boxes.numel():
            boxes = boxes.clone()
            boxes[:, 1] = (boxes[:, 1] * nw + dx) / w
            boxes[:, 2] = (boxes[:, 2] * nh + dy) / h
            boxes[:, 3] = boxes[:, 3] * nw / w
            boxes[:, 4] = boxes[:, 4] * nh / h
            boxes = clip_xywh_normalized(boxes)
        return canvas, boxes


class Resize:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image, boxes):
        image = image.resize(self.size, Image.BILINEAR)
        return image, boxes


class ToTensor:
    def __call__(self, image, boxes):
        from torchvision.transforms import functional as F

        return F.to_tensor(image), boxes


def build_train_transforms(image_size=448, extra_aug=True, jitter=0.2):
    ts = [ColorJitter(0.2, 0.5, 0.7, 0.07)]
    if extra_aug:
        ts += [
            RandomBlur(p=0.1),
            RandomGrayscale(p=0.1),
        ]
    ts += [RandomHorizontalFlip(0.5)]
    if extra_aug:
        ts += [
            RandomVerticalFlip(0.05),
            RandomRotationJitter(p=0.5, degrees=8),
        ]
    ts += [
        YOLOJitter(jitter=jitter),
        Resize(image_size),
        ToTensor(),
    ]
    return Compose(ts)


def build_val_transforms(image_size=448):
    return Compose([Resize(image_size), ToTensor()])
