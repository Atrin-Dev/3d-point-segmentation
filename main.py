import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.pointnet_plus_plus_segmentation import *  # Assuming you have a PointNet-based segmentation model
from dataset import PointCloudPartSegmentationDataset # Import dataset class
import os
import csv
import glob
import argparse

# Hyperparameters
NUM_POINTS = 2048
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 50
CHECKPOINT_DIR = "checkpoints"
#METRIC_DIR = "checkpoints_test"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Define color-to-label mapping
color_to_label = {
    (0, 0, 128): 0,
    (0, 77, 255): 1,
    (41, 255, 206): 2,
    (125, 255, 122): 3,
    (128, 0, 0): 4,
    (206, 255, 41): 5,
    (255, 104, 0): 6,
    (255, 148, 0): 7
}

# Load dataset
# file_paths = ["data/cloud1.npy", "data/cloud2.npy"]  # Update with actual file paths
file_paths = glob.glob("/content/drive/My Drive/Course/PointCloud/DataSet/Chair_1/Test0/*.ply")
dataset = PointCloudPartSegmentationDataset(file_paths, color_to_label, num_points=NUM_POINTS, augment=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model, loss function, and optimizer
model = PointNetPlusPlusSegmentation(num_classes=len(set(color_to_label.values())))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def save_checkpoint(epoch, model, optimizer, loss):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


# Function to save loss and accuracy to separate CSV files
def save_metrics(epoch, loss, accuracy, metric_file):
    metrics_file = os.path.join(CHECKPOINT_DIR, metric_file)
    # Write headers if the file is empty
    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'accuracy'])

    # Append the current epoch's metrics
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, loss, accuracy])
    print(f"Metrics saved: {metrics_file}")

def train():
    model.train()
    total_loss = 0.0
    correct, total = 0, 0  # Initialize these variables at the start of each epoch
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for points, labels in dataloader:
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(points)  # Forward pass
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total * 100
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(dataloader)}")
        save_checkpoint(epoch, model, optimizer, avg_loss)
        save_metrics(epoch, avg_loss, accuracy, metric_file= "metrics_train.csv")  # Save loss and accuracy separately

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {checkpoint_path}, epoch {checkpoint['epoch']}, loss {checkpoint['loss']}")
    return checkpoint['epoch']

def evaluate(checkpoint_path):
    model.eval()
    epoch = load_checkpoint(checkpoint_path, model, optimizer)
    correct, total = 0, 0
    total_loss = 0.0
    with torch.no_grad():
        for points, labels in dataloader:
            points, labels = points.to(device), labels.to(device)
            outputs = model(points)
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1))
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total * 100
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        save_metrics(epoch, avg_loss, accuracy, metric_file= "metrics_eval.csv")  # Save metrics with the correct epoch
        return epoch, avg_loss, accuracy

def evaluate_all_checkpoints(checkpoint_dir):
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    checkpoint_paths.sort()  # Ensure they are processed in order
    print(f"Found {len(checkpoint_paths)} checkpoints.")

    # Clear existing metrics file
    metrics_file = os.path.join(CHECKPOINT_DIR, "all_checkpoints_metrics.csv")
    if os.path.exists(metrics_file):
        os.remove(metrics_file)

    # Evaluate each checkpoint and save metrics
    for checkpoint_path in checkpoint_paths:
        print(f"Evaluating {checkpoint_path}...")
        evaluate(checkpoint_path)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == 'train':
        train()
    elif args.mode == 'eval':
        evaluate_all_checkpoints(CHECKPOINT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test PointNet++ for point cloud segmentation")
    parser.add_argument("--mode", required=True, choices=['train', 'eval'], help="Choose 'train' or 'test' mode")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    main(args)