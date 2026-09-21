from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo_v1.boxes import box_iou_xyxy, cell_to_image_xywh, xywh_to_xyxy


class YOLOv1Loss(nn.Module):
    """YOLOv1 multi-part SSE loss (Redmon et al. / arXiv:2305.17786).

    Layout of each cell: [x, y, w, h, c] * B + class_probs(C)
    Confidence target for the responsible box is the IoU with GT (paper definition).
    """

    def __init__(self, S=7, B=2, C=20, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        n = pred.shape[0]
        pred = pred.reshape(n, self.S, self.S, self.B * 5 + self.C)
        target = target.reshape(n, self.S, self.S, self.B * 5 + self.C)

        pred_boxes = pred[..., : self.B * 5].reshape(n, self.S, self.S, self.B, 5)
        tgt_box = target[..., :5]
        obj = target[..., 4] > 0.5  # (N,S,S)
        pred_cls = pred[..., self.B * 5 :]
        tgt_cls = target[..., self.B * 5 :]

        pred_xyxy = xywh_to_xyxy(cell_to_image_xywh(pred_boxes[..., :4], self.S))
        tgt_xyxy = xywh_to_xyxy(cell_to_image_xywh(tgt_box[..., :4], self.S)).unsqueeze(-2)
        ious = box_iou_xyxy(pred_xyxy, tgt_xyxy)  # (N,S,S,B)

        best = ious.argmax(dim=-1)
        responsible = F.one_hot(best, self.B).bool()  # (N,S,S,B)
        box_obj = obj.unsqueeze(-1) & responsible
        box_noobj = ~box_obj

        # ---- coordinate loss (sqrt on w,h as in the paper) ----
        pred_xywh = pred_boxes[..., :4]
        tgt_xywh = tgt_box[..., :4].unsqueeze(-2).expand_as(pred_xywh)
        pred_wh = torch.sign(pred_xywh[..., 2:4]) * torch.sqrt(
            pred_xywh[..., 2:4].abs() + 1e-6
        )
        tgt_wh = torch.sqrt(tgt_xywh[..., 2:4].clamp(min=1e-6))
        pred_coord = torch.cat([pred_xywh[..., :2], pred_wh], dim=-1)
        tgt_coord = torch.cat([tgt_xywh[..., :2], tgt_wh], dim=-1)
        coord_loss = _masked_mse(pred_coord, tgt_coord, box_obj)

        # ---- confidence: IoU for responsible boxes, 0 otherwise ----
        conf_pred = pred_boxes[..., 4]
        conf_tgt = ious.detach()
        obj_loss = _masked_mse(conf_pred, conf_tgt, box_obj)
        noobj_loss = _masked_mse(conf_pred, torch.zeros_like(conf_pred), box_noobj)

        # ---- classification (only cells that contain an object) ----
        cls_mask = obj.unsqueeze(-1).expand_as(pred_cls)
        cls_loss = _masked_mse(pred_cls, tgt_cls, cls_mask)

        total = (
            self.lambda_coord * coord_loss
            + obj_loss
            + self.lambda_noobj * noobj_loss
            + cls_loss
        ) / n
        return {
            "loss": total,
            "coord": (coord_loss / n).detach(),
            "obj": (obj_loss / n).detach(),
            "noobj": (noobj_loss / n).detach(),
            "cls": (cls_loss / n).detach(),
        }


def _masked_mse(pred, target, mask):
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(pred)
    if mask.sum() == 0:
        return pred.new_zeros(())
    return F.mse_loss(pred[mask], target[mask], reduction="sum")
