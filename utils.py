from config import data_path
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import cv2
train_transforms = transforms.Compose([
    transforms.Resize(416),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

test_transforms = transforms.Compose([
    transforms.Resize(416),
    transforms.ToTensor()])



def get_dataloader(batch_size=32, img_size=256):
    image_folder = os.path.join(data_path, 'images')
    label_folder = os.path.join(data_path, 'labels')
    train_image_folder = os.path.join(image_folder, 'train')
    test_image_folder = os.path.join(image_folder, 'test')
    train_label_folder = os.path.join(label_folder, 'train')
    test_label_folder = os.path.join(label_folder, 'test')

    train_dataset = DetectionDataset(train_image_folder, train_label_folder, transform=train_transforms)
    test_dataset = DetectionDataset(test_image_folder, test_label_folder, transform=test_transforms)

    # custom collate to allow variable number of boxes per image (labels have shape [N,5])
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images, dim=0)
        targets = list(targets)
        return images, targets

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    return train_loader, test_loader


class DetectionDataset:
    def __init__(self, image_dir, label_dir, transform=None):
        self.transform = transform

        image_files = sorted(os.listdir(image_dir))
        label_files = sorted(os.listdir(label_dir))

        # Preload all images and labels into memory (use paths only during init)
        self.images = []
        self.labels = []
        for img_fname, lbl_fname in zip(image_files, label_files):
            img_path = os.path.join(image_dir, img_fname)
            lbl_path = os.path.join(label_dir, lbl_fname)

            img = datasets.folder.default_loader(img_path)
            boxes = []
            with open(lbl_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                    except ValueError:
                        continue
                    boxes.append([x, y, w, h,cls])
            if boxes:
                lbl = torch.tensor(boxes, dtype=torch.float32)
                # lbl = torch.as_tensor(boxes, dtype=torch.float32)
            else:
                lbl = torch.empty((0, 5), dtype=torch.float32)


            # if transform:
            #     img = transform(img)
            self.images.append(img)
            self.labels.append(boxes)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Do not use paths here; return the already-loaded image and label
        image = self.images[idx]
        label = self.labels[idx]

        # Use a copy to avoid accidental in-place modifications of the cached image
        if self.transform:
            image = self.transform(image.copy())
        return image, label
    

import enum

class ModelType(enum.Enum):
    YOLOV8 = 1
    SSD = 2
    FAST_RCNN = 3
    RETINANET = 4
    
    