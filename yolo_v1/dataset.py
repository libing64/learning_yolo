from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from yolo_v1.boxes import encode_target, xyxy_to_xywh
from yolo_v1.config import CLASS_TO_IDX


def parse_voc_xml(xml_path: Path, skip_difficult: bool = True):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))
    boxes = []
    for obj in root.findall("object"):
        difficult = int(obj.findtext("difficult") or 0)
        if skip_difficult and difficult:
            continue
        name = obj.findtext("name")
        if name not in CLASS_TO_IDX:
            continue
        bb = obj.find("bndbox")
        xmin = float(bb.findtext("xmin"))
        ymin = float(bb.findtext("ymin"))
        xmax = float(bb.findtext("xmax"))
        ymax = float(bb.findtext("ymax"))
        xmin = min(max(xmin, 1.0), width)
        ymin = min(max(ymin, 1.0), height)
        xmax = min(max(xmax, 1.0), width)
        ymax = min(max(ymax, 1.0), height)
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append(
            {
                "class_id": CLASS_TO_IDX[name],
                "name": name,
                "xyxy": [xmin, ymin, xmax, ymax],
                "difficult": difficult,
            }
        )
    return {"width": width, "height": height, "boxes": boxes}


class VOCYoloDataset(Dataset):
    """PASCAL VOC 2007/2012 images encoded as YOLOv1 grid tensors."""

    def __init__(
        self,
        root: str | Path,
        years=("2007", "2012"),
        image_sets=("trainval", "trainval"),
        transform=None,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        encode: bool = True,
        skip_difficult: bool = True,
    ):
        self.root = Path(root).expanduser()
        self.transform = transform
        self.S, self.B, self.C = S, B, C
        self.encode = encode
        self.skip_difficult = skip_difficult
        self.samples: list[tuple[Path, Path]] = []

        years = (years,) if isinstance(years, str) else years
        image_sets = (image_sets,) if isinstance(image_sets, str) else image_sets
        if len(years) != len(image_sets):
            raise ValueError("years and image_sets must have the same length")

        for year, split in zip(years, image_sets):
            voc = self.root / "VOCdevkit" / f"VOC{year}"
            id_file = voc / "ImageSets" / "Main" / f"{split}.txt"
            if not id_file.exists():
                raise FileNotFoundError(
                    f"Missing {id_file}. Download VOC to {self.root} first "
                    "(python -m yolo_v1.voc_download)."
                )
            ids = [line.strip() for line in id_file.read_text().splitlines() if line.strip()]
            for img_id in ids:
                img = voc / "JPEGImages" / f"{img_id}.jpg"
                ann = voc / "Annotations" / f"{img_id}.xml"
                if img.exists() and ann.exists():
                    self.samples.append((img, ann))

        if not self.samples:
            raise RuntimeError(f"No VOC samples found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, ann_path = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        meta = parse_voc_xml(ann_path, skip_difficult=self.skip_difficult)
        w, h = meta["width"], meta["height"]

        if meta["boxes"]:
            xyxy = torch.tensor([b["xyxy"] for b in meta["boxes"]], dtype=torch.float32)
            xyxy[:, [0, 2]] /= w
            xyxy[:, [1, 3]] /= h
            cls = torch.tensor([b["class_id"] for b in meta["boxes"]], dtype=torch.float32)[:, None]
            boxes = torch.cat([cls, xyxy_to_xywh(xyxy)], dim=1)
        else:
            boxes = torch.zeros((0, 5), dtype=torch.float32)

        if self.transform is not None:
            image, boxes = self.transform(image, boxes)

        info = {
            "image_id": img_path.stem,
            "path": str(img_path),
            "width": w,
            "height": h,
        }
        if self.encode:
            target = encode_target(boxes, self.S, self.B, self.C)
            return image, target, info
        return image, boxes, info


def collate_train(batch):
    images, targets, infos = zip(*batch)
    return torch.stack(images, 0), torch.stack(targets, 0), list(infos)


def collect_gt(dataset: VOCYoloDataset) -> dict[str, torch.Tensor]:
    """Load GT boxes with difficult flags for VOC-style mAP."""
    gts = {}
    for img_path, ann_path in dataset.samples:
        meta = parse_voc_xml(ann_path, skip_difficult=False)
        w, h = meta["width"], meta["height"]
        rows = []
        for b in meta["boxes"]:
            x1, y1, x2, y2 = b["xyxy"]
            rows.append([x1 / w, y1 / h, x2 / w, y2 / h, b["class_id"], b["difficult"]])
        gts[img_path.stem] = (
            torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros((0, 6))
        )
    return gts
