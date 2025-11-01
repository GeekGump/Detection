import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOLoss(nn.Module):
    def __init__(self, grid_size=13, num_anchors=5, num_classes=36, object_scale=5.0, 
                 noobject_scale=1.0, class_scale=1.0, coord_scale=1.0, anchors=None):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors if anchors is not None else [
            (1.08, 1.19), (3.42, 4.41), (6.63, 11.38),
            (9.42, 5.11), (16.62, 10.52)
        ]
        self.grid_size = grid_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.object_scale = object_scale
        self.noobject_scale = noobject_scale
        self.class_scale = class_scale
        self.coord_scale = coord_scale
        self.reducetion = 32
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')

    # target: [batch_size, num_anchors, grid_size, grid_size, 5 + num_classes]
    def forward(self, predictions, target):
        batch_size = predictions.size(0)
        predictions = predictions.view(batch_size, self.num_anchors, 5 + self.num_classes, self.grid_size, self.grid_size)
        predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()

        coord_mask, conf_mask, class_mask, tcoord, tconf, tclass = self.build_targets(predictions, target)

        pred_coord = predictions[..., 1:5]
        pred_conf = torch.sigmoid(predictions[..., 0])
        pred_class = predictions[..., 5:]

        loss_coord = self.coord_scale * self.mse_loss(pred_coord * coord_mask, tcoord * coord_mask) / batch_size
        loss_conf = self.mse_loss(pred_conf * conf_mask, tconf * conf_mask) / batch_size
        loss_class = self.class_scale * self.ce_loss(pred_class.view(-1, self.num_classes), 
                                                    torch.argmax(tclass.view(-1, self.num_classes), dim=1)) / batch_size

        total_loss = loss_coord + loss_conf + loss_class
        return total_loss, loss_coord, loss_conf, loss_class
    
    def bbox_iou(self, box1, box2):
        # box: (x_center, y_center, width, height)
        b1_x1 = box1[0] - box1[2] / 2
        b1_y1 = box1[1] - box1[3] / 2
        b1_x2 = box1[0] + box1[2] / 2
        b1_y2 = box1[1] + box1[3] / 2

        b2_x1 = box2[0] - box2[2] / 2
        b2_y1 = box2[1] - box2[3] / 2
        b2_x2 = box2[0] + box2[2] / 2
        b2_y2 = box2[1] + box2[3] / 2

        inter_rect_x1 = max(b1_x1, b2_x1)
        inter_rect_y1 = max(b1_y1, b2_y1)
        inter_rect_x2 = min(b1_x2, b2_x2)
        inter_rect_y2 = min(b1_y2, b2_y2)

        inter_area = max(0, inter_rect_x2 - inter_rect_x1) * max(0, inter_rect_y2 - inter_rect_y1)

        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
        return iou
        

    def build_targets(self, predictions, target):
        device = predictions.device
        batch_size = predictions.size(0)
        grid_size = self.grid_size

        conf_mask = torch.ones_like(predictions[..., 0],requires_grad=False) * self.noobject_scale
        coord_mask = torch.zeros_like(predictions[..., 1:5],requires_grad=False)
        class_mask = torch.zeros_like(predictions[..., 5:],requires_grad=False)

        tconf = torch.zeros_like(predictions[..., 0],requires_grad=False)
        tcoord = torch.zeros_like(predictions[..., 1:5],requires_grad=False)
        tclass = torch.zeros_like(predictions[..., 5:],requires_grad=False)

        for b in range(batch_size):
            for t in range(len(target[b])):
                if sum(target[b][t]) == 0:
                    continue
                gx = target[b][t][0] * grid_size
                gy = target[b][t][1] * grid_size
                gw = target[b][t][2] * grid_size
                gh = target[b][t][3] * grid_size
                gi = int(gx)
                gj = int(gy)

                best_iou = 0
                best_n = -1
                for n in range(self.num_anchors):
                    anchor_w, anchor_h = self.anchors[n]
                    iou = self.bbox_iou((0, 0, gw, gh), (0, 0, anchor_w, anchor_h))
                    if iou > best_iou:
                        best_iou = iou
                        best_n = n

                coord_mask[b, best_n, gj, gi] = 1
                conf_mask[b, best_n, gj, gi] = self.object_scale
                class_mask[b, best_n, gj, gi] = self.class_scale

                tcoord[b, best_n, gj, gi, 0] = gx - gi
                tcoord[b, best_n, gj, gi, 1] = gy - gj
                tcoord[b, best_n, gj, gi, 2] = torch.log(torch.tensor(gw / self.anchors[best_n][0] + 1e-16))
                tcoord[b, best_n, gj, gi, 3] = torch.log(torch.tensor(gh / self.anchors[best_n][1] + 1e-16))

                tconf[b, best_n, gj, gi] = 1
                tclass[b, best_n, gj, gi, int(target[b][t][4])] = 1
        return coord_mask, conf_mask, class_mask, tcoord, tconf, tclass

