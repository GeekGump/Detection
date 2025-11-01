from ultralytics import YOLO
import torch
from config import model_path, yaml_path
print("PyTorch 版本:", torch.__version__)
# model = YOLO("yolo8n.yaml").load("yolo8n.pt")  # build from YAML and transfer weights


model = YOLO(model_path)  # load a pretrained model (recommended for training)
# Train the model
results = model.train(data=yaml_path, epochs=1, imgsz=640)