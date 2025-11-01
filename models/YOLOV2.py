import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOV2(nn.Module):
    def __init__(self, num_classes=36, anchors=None):
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


def yolo_v2(num_classes=36, anchors=None):
    return YOLOV2(num_classes=num_classes, anchors=anchors)

