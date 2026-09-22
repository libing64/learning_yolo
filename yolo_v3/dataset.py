from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


# COCO category_id is not contiguous 1..90; map to 0..79
def build_coco_category_maps(instances_json: Path):
    with open(instances_json) as f:
        data = json.load(f)
    cats = sorted(data["categories"], key=lambda c: c["id"])
    cat_id_to_contiguous = {c["id"]: i for i, c in enumerate(cats)}
    contiguous_to_name = {i: c["name"] for i, c in enumerate(cats)}
    return cat_id_to_contiguous, contiguous_to_name, data


class LetterBox:
    """Resize with unchanged aspect ratio and pad to square (YOLO-style)."""

    def __init__(self, size: int = 416, fill: int = 114):
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image, boxes_xyxy: torch.Tensor):
        w0, h0 = image.size
        scale = min(self.size / w0, self.size / h0)
        nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
        image = image.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (self.fill, self.fill, self.fill))
        pad_x = (self.size - nw) // 2
        pad_y = (self.size - nh) // 2
        canvas.paste(image, (pad_x, pad_y))
        if boxes_xyxy.numel():
            boxes = boxes_xyxy.clone()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_y
        else:
            boxes = boxes_xyxy
        meta = {"scale": scale, "pad": (pad_x, pad_y), "orig_size": (w0, h0)}
        return canvas, boxes, meta


class ColorAug:
    def __init__(self, hue=0.1, sat=1.5, val=1.5):
        self.hue = hue
        self.sat = sat
        self.val = val

    def __call__(self, image: Image.Image):
        # simple torchvision jitter approximating Darknet hue/sat/exposure
        image = TF.adjust_brightness(image, 1.0 + random.uniform(-0.4, 0.4))
        image = TF.adjust_contrast(image, 1.0 + random.uniform(-0.4, 0.4))
        image = TF.adjust_saturation(image, 1.0 + random.uniform(-0.7, 0.7))
        image = TF.adjust_hue(image, random.uniform(-self.hue, self.hue))
        return image


class CocoDetectionYolo(Dataset):
    """COCO detection in YOLO target format: class, cx, cy, w, h (normalized)."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 416,
        augment: bool = False,
        multi_scale: bool = False,
        multi_scale_range: tuple[int, int] = (320, 608),
    ):
        self.root = Path(root).expanduser()
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.multi_scale = multi_scale and augment
        self.multi_scale_range = multi_scale_range
        ann_name = f"instances_{'train' if split == 'train' else 'val'}2017.json"
        img_dir = self.root / ("train2017" if split == "train" else "val2017")
        ann_path = self.root / "annotations" / ann_name
        if not ann_path.exists():
            raise FileNotFoundError(ann_path)

        self.cat_id_to_idx, self.idx_to_name, coco = build_coco_category_maps(ann_path)
        self.images = {im["id"]: im for im in coco["images"]}
        anns_by_img: dict[int, list] = {i: [] for i in self.images}
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            if ann["image_id"] not in anns_by_img:
                continue
            x, y, bw, bh = ann["bbox"]
            if bw < 1 or bh < 1:
                continue
            anns_by_img[ann["image_id"]].append(ann)

        self.samples = []
        for img_id, im in self.images.items():
            path = img_dir / im["file_name"]
            if path.exists():
                self.samples.append((path, img_id, anns_by_img[img_id], im))

        self.letterbox = LetterBox(image_size)
        self.color_aug = ColorAug()
        self._current_size = image_size

    def __len__(self):
        return len(self.samples)

    def set_image_size(self, size: int):
        size = int(round(size / 32) * 32)
        size = max(self.multi_scale_range[0], min(self.multi_scale_range[1], size))
        self._current_size = size
        self.letterbox = LetterBox(size)

    def __getitem__(self, index: int):
        path, img_id, anns, im = self.samples[index]
        image = Image.open(path).convert("RGB")
        w0, h0 = image.size

        boxes = []
        labels = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            boxes.append([x, y, x + bw, y + bh])
            labels.append(self.cat_id_to_idx[ann["category_id"]])
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0,))

        if self.augment and random.random() < 0.5 and boxes.numel():
            image = TF.hflip(image)
            boxes = boxes.clone()
            boxes[:, [0, 2]] = w0 - boxes[:, [2, 0]]

        if self.augment:
            image = self.color_aug(image)

        size = self._current_size
        if self.multi_scale and random.random() < 0.5:
            # size is controlled externally each few batches; keep current
            pass
        self.letterbox.size = size
        image, boxes, meta = self.letterbox(image, boxes)

        # to normalized xywh
        if boxes.numel():
            xywh = torch.zeros((boxes.shape[0], 4))
            xywh[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2) / size
            xywh[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2) / size
            xywh[:, 2] = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) / size
            xywh[:, 3] = (boxes[:, 3] - boxes[:, 1]).clamp(min=0) / size
            # clip centers inside image
            keep = (xywh[:, 2] > 1e-3) & (xywh[:, 3] > 1e-3)
            keep &= (xywh[:, 0] > 0) & (xywh[:, 0] < 1) & (xywh[:, 1] > 0) & (xywh[:, 1] < 1)
            xywh, labels = xywh[keep], labels[keep]
            targets = torch.cat([labels[:, None], xywh], dim=1)
        else:
            targets = torch.zeros((0, 5), dtype=torch.float32)

        tensor = TF.to_tensor(image)
        info = {
            "image_id": img_id,
            "path": str(path),
            "orig_size": meta["orig_size"],
            "pad": meta["pad"],
            "scale": meta["scale"],
            "input_size": size,
        }
        return tensor, targets, info


def collate_fn(batch):
    images, targets, infos = zip(*batch)
    # pad images if multi-scale caused different sizes within a worker — enforce same
    sizes = {im.shape[-1] for im in images}
    if len(sizes) != 1:
        # resize all to max size in batch (rare)
        size = max(sizes)
        images = [
            TF.resize(im, [size, size]) if im.shape[-1] != size else im for im in images
        ]
    images = torch.stack(images, 0)
    out_targets = []
    for i, t in enumerate(targets):
        if t.numel() == 0:
            continue
        idx = torch.full((t.shape[0], 1), i, dtype=t.dtype)
        out_targets.append(torch.cat([idx, t], dim=1))
    targets = torch.cat(out_targets, 0) if out_targets else torch.zeros((0, 6))
    return images, targets, list(infos)
