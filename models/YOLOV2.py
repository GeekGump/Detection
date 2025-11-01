import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOV2(nn.Module):
    def __init__(self, num_classes=20, anchors=None):
        super(YOLOV2, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors if anchors is not None else [
            (1.08, 1.19), (3.42, 4.41), (6.63, 11.38),
            (9.42, 5.11), (16.62, 10.52)
        ]
        self.num_anchors = len(self.anchors)

        # Define the convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )

        # Final prediction conv: for each anchor predict (tx, ty, tw, th, obj) + num_classes
        out_channels = self.num_anchors * (5 + self.num_classes)
        self.pred = nn.Conv2d(1024, out_channels, kernel_size=1, stride=1, padding=0)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Input:
            x: tensor (N, 3, H, W)
        Returns:
            preds: tensor shaped (N, S, S, num_anchors, 5 + num_classes)
                   where S is the feature map spatial size (H/32 for this architecture)
                   The tensor contains raw predictions with tx, ty and objectness passed through sigmoid.
                   tw, th and class logits are returned as raw values (suitable for loss functions).
        """
        N = x.size(0)
        features = self.features(x)
        preds = self.pred(features)  # (N, A*(5+C), S, S)

        A = self.num_anchors
        NC = 5 + self.num_classes
        N, _, S_h, S_w = preds.shape

        # reshape to (N, A, NC, S_h, S_w) then permute to (N, S_h, S_w, A, NC)
        preds = preds.view(N, A, NC, S_h, S_w).permute(0, 3, 4, 1, 2).contiguous()

        # apply sigmoid to tx, ty and objectness (index 0,1 and 4)
        preds[..., 0] = torch.sigmoid(preds[..., 0])  # tx
        preds[..., 1] = torch.sigmoid(preds[..., 1])  # ty
        preds[..., 4] = torch.sigmoid(preds[..., 4])  # objectness

        # Note: tw, th (preds[...,2:4]) and class logits (preds[...,5:]) are left as raw values.
        return preds


def yolo_v2(num_classes=20, anchors=None):
    return YOLOV2(num_classes=num_classes, anchors=anchors)


class YOLOV2LOSS(nn.modules.loss._Loss):
    """
    YOLOv2 loss.

    Expected predictions shape: (N, S, S, A, 5 + C) with
      preds[..., 0] = tx (sigmoid applied in model),
      preds[..., 1] = ty (sigmoid applied),
      preds[..., 2] = tw (raw),
      preds[..., 3] = th (raw),
      preds[..., 4] = objectness (sigmoid applied),
      preds[..., 5:] = class logits (raw).

    Expected target format:
      - either a list of length N where each element is a tensor (n_obj, 5)
        with columns (class_idx, x, y, w, h) normalized [0,1],
      - or a tensor of shape (N, max_obj, 5) padded with zeros for missing objects.
    """
    def __init__(self, S=13, B=5, C=20, lambda_coord=5.0, lambda_noobj=0.5, anchors=None, ignore_thresh=0.5):
        super(YOLOV2LOSS, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.ignore_thresh = ignore_thresh

        # anchors expected in "grid cell" units like YOLOv2 defaults (e.g. [(1.08,1.19), ...])
        self.anchors = anchors if anchors is not None else [
            (1.08, 1.19), (3.42, 4.41), (6.63, 11.38),
            (9.42, 5.11), (16.62, 10.52)
        ]
        assert len(self.anchors) == self.B, "Number of anchors must equal B"

    def _box_iou_wh(self, box_wh, anchors_wh):
        # box_wh: (..., 2) [w,h], anchors_wh: (A,2)
        # IoU assuming centered at same point
        b_w, b_h = box_wh[..., 0:1], box_wh[..., 1:2]  # keep dims
        a_w = anchors_wh[:, 0].view(1, -1)
        a_h = anchors_wh[:, 1].view(1, -1)

        inter_w = torch.min(b_w, a_w)
        inter_h = torch.min(b_h, a_h)
        inter_area = (inter_w * inter_h).squeeze(-1)

        box_area = (b_w * b_h).squeeze(-1)
        anchor_area = (a_w * a_h).squeeze(0)

        union = box_area + anchor_area - inter_area
        iou = inter_area / (union + 1e-16)
        return iou  # shape (..., A)

    def build_targets(self, predictions, target):
        """
        Build masks and target tensors for loss computation.

        Returns:
          obj_mask, noobj_mask: (N,S,S,A) float tensors (1.0/0.0)
          tx, ty, tw, th: (N,S,S,A) target tensors
          cls_target: (N,S,S,A,C) one-hot class targets
        """
        device = predictions.device
        N = predictions.size(0)
        S = self.S
        A = self.B
        C = self.C

        # initialize
        obj_mask = torch.zeros((N, S, S, A), dtype=torch.float32, device=device)
        noobj_mask = torch.ones((N, S, S, A), dtype=torch.float32, device=device)

        tx = torch.zeros((N, S, S, A), dtype=torch.float32, device=device)
        ty = torch.zeros((N, S, S, A), dtype=torch.float32, device=device)
        tw = torch.zeros((N, S, S, A), dtype=torch.float32, device=device)
        th = torch.zeros((N, S, S, A), dtype=torch.float32, device=device)
        cls_target = torch.zeros((N, S, S, A, C), dtype=torch.float32, device=device)

        # prepare anchors normalized to image coordinates: anchor / S
        anchors = torch.tensor(self.anchors, dtype=torch.float32, device=device)  # (A,2) in grid units
        anchors_norm = anchors / float(S)  # (A,2) in image relative coordinates

        # accept target as list or tensor
        if isinstance(target, torch.Tensor) and target.dim() == 3:
            # shape (N, max_obj, 5)
            target_list = []
            for b in range(N):
                objs = target[b]
                # filter rows with nonzero width & height or non-negative class
                mask = (objs[:, 4] > 0) | (objs[:, 2] > 0)  # w>0 or x>0 so not all-zero
                objs_f = objs[mask]
                target_list.append(objs_f)
        elif isinstance(target, list):
            target_list = target
        else:
            raise ValueError("Unsupported target type. Expect list or tensor (N,max_obj,5).")

        for b in range(N):
            t_b = target_list[b]
            if t_b is None or t_b.numel() == 0:
                continue
            for t in range(t_b.size(0)):
                cls_idx = int(t_b[t, 0].item())
                gx = t_b[t, 1].item()
                gy = t_b[t, 2].item()
                gw = t_b[t, 3].item()
                gh = t_b[t, 4].item()

                if gw <= 0 or gh <= 0:
                    continue

                # which cell
                gi = int(gx * S)
                gj = int(gy * S)
                gi = max(0, min(S - 1, gi))
                gj = max(0, min(S - 1, gj))

                # compute best anchor by IoU between gt w,h and anchors_norm
                box_wh = torch.tensor([gw, gh], dtype=torch.float32, device=device).view(1, 2)  # (1,2)
                ious = self._box_iou_wh(box_wh, anchors_norm * S)  # compare in grid units -> anchors_norm*S == anchors
                # Note: anchors were passed in grid units originally; comparing gw,gh (normalized)
                # to anchors_norm requires consistent units. we used anchors as grid-units in anchors tensor,
                # so adjust IoU computation by comparing gw*S,gh*S to anchors in grid units.
                # The above call used anchors_norm*S (back to grid units) for consistency.
                # But simpler: recompute properly:
                # recompute ious correctly:
                ious = self._box_iou_wh(torch.tensor([[gw * S, gh * S]], device=device), anchors)

                best_n = int(torch.argmax(ious, dim=-1).item())

                # set masks/targets for responsible anchor
                obj_mask[b, gj, gi, best_n] = 1.0
                noobj_mask[b, gj, gi, best_n] = 0.0

                # Set noobj mask to 0 for anchors with high IoU to avoid penalizing them
                iou_mask = (ious.squeeze(0) > self.ignore_thresh)
                noobj_mask[b, gj, gi, iou_mask] = 0.0

                # tx,ty are offsets inside cell
                tx[b, gj, gi, best_n] = gx * S - gi
                ty[b, gj, gi, best_n] = gy * S - gj

                # tw,th are log-space scalings: target_tw = log(gt_w / anchor_w)
                anchor_w = anchors[best_n, 0] / float(S)  # normalized anchor w
                anchor_h = anchors[best_n, 1] / float(S)
                tw_target_val = torch.log((gw / (anchor_w + 1e-16)) + 1e-16)
                th_target_val = torch.log((gh / (anchor_h + 1e-16)) + 1e-16)
                tw[b, gj, gi, best_n] = tw_target_val
                th[b, gj, gi, best_n] = th_target_val

                # class one-hot
                if 0 <= cls_idx < C:
                    cls_target[b, gj, gi, best_n, cls_idx] = 1.0

        return obj_mask, noobj_mask, tx, ty, tw, th, cls_target

    def forward(self, predictions, target):
        """
        predictions: tensor (N, S, S, A, 5+C)
        target: list or (N, max_obj, 5) tensor
        """
        device = predictions.device
        N = predictions.size(0)
        S = self.S
        A = self.B
        C = self.C

        preds = predictions  # expected already shaped (N,S,S,A,5+C)
        if preds.dim() != 5 or preds.size(1) != S:
            # try to handle predictions shaped (N, S, S, A, 5+C) or (N, A, 5+C, S, S)
            raise ValueError(f"Predictions must be shaped (N, S, S, A, 5+C). Got {preds.shape}")

        # split preds
        pred_x = preds[..., 0]
        pred_y = preds[..., 1]
        pred_w = preds[..., 2]
        pred_h = preds[..., 3]
        pred_obj = preds[..., 4]
        pred_cls = preds[..., 5:]  # (N,S,S,A,C)

        # build targets
        obj_mask, noobj_mask, tx, ty, tw, th, cls_target = self.build_targets(preds, target)

        # coordinate losses (only where obj_mask == 1)
        mse = nn.MSELoss(reduction="sum")

        # xy loss
        loss_x = ((pred_x - tx) ** 2 * obj_mask).sum()
        loss_y = ((pred_y - ty) ** 2 * obj_mask).sum()

        # wh loss (pred_w, pred_h are raw logits; targets are log-space)
        loss_w = ((pred_w - tw) ** 2 * obj_mask).sum()
        loss_h = ((pred_h - th) ** 2 * obj_mask).sum()

        coord_loss = self.lambda_coord * (loss_x + loss_y + loss_w + loss_h) / max(1, N)

        # objectness loss
        loss_obj = ((pred_obj - 1) ** 2 * obj_mask).sum()
        loss_noobj = ((pred_obj - 0) ** 2 * noobj_mask).sum()
        obj_loss = (loss_obj + self.lambda_noobj * loss_noobj) / max(1, N)

        # class loss (MSE on logits vs one-hot)
        loss_cls = ((pred_cls - cls_target) ** 2 * obj_mask.unsqueeze(-1)).sum() / max(1, N)

        total_loss = coord_loss + obj_loss + loss_cls

        return total_loss
