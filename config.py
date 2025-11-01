import os

model_path = ""
yaml_path = ""
data_path = ""

if os.name == 'nt':  # Windows
    model_path = "D:\\SHIVAAAA\\Documents\\MinGW\\YOLO\\yolov8n.pt"
    yaml_path = "D:\\SHIVAAAA\\Documents\\MinGW\\NeuralNetwork\\dataset\\OCR.yaml"
    data_path = "D:\\SHIVAAAA\\Documents\\MinGW\\NeuralNetwork\\dataset"
else:  # POSIX (Linux, macOS, etc.)
    model_path = "yolov8n.pt"
    yaml_path = "dataset/data.yaml"
    data_path = "dataset"

print(f"Model path set to: {model_path}")
print(f"YAML path set to: {yaml_path}")
print(f"Data path set to: {data_path}")