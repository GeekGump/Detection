from models.YOLOV2 import YOLOV2,YOLOV2LOSS
import torch
import numpy as np
import torch
from utils import get_dataloader

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, targets in dataloader:
        images = images.to(device)
        # Assuming targets is a list of tensors
        targets = [t.to(device) for t in targets]

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = [t.to(device) for t in targets]

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == "__main__":
    # Hyperparameters
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 16

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model, Loss, Optimizer
    model = YOLOV2(num_classes=20).to(device)
    criterion = YOLOV2LOSS()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # DataLoader
    train_dataloader,test_dataloader = get_dataloader(batch_size=batch_size)

    # Training loop
    for epoch in range(num_epochs):
        avg_loss = train_model(model, train_dataloader, criterion, optimizer, device)
        eval_loss = evaluate_model(model, test_dataloader, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Eval Loss: {eval_loss:.4f}')
