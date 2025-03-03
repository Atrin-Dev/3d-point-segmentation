import torch
import numpy as np
import os
import json
from plyfile import PlyData, PlyElement
from models.pointnet_plus_plus_segmentation import PointNetPlusPlusSegmentation
import glob

# Load color-to-label mapping
with open('color_to_label.json', 'r') as f:
    color_to_label = json.load(f)

# Reverse mapping: label -> color
label_to_color = {int(v): [int(c) for c in k.strip('()').split(',')] for k, v in color_to_label.items()}

# Initialize model, loss function, and optimizer
model = PointNetPlusPlusSegmentation(num_classes=len(set(color_to_label.values())))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Load point cloud and extract xyz, rgb
def load_point_cloud(file_path):
    ply_data = PlyData.read(file_path)
    vertices = ply_data['vertex'].data
    xyz = np.vstack((vertices['x'], vertices['y'], vertices['z'])).T
    rgb = np.vstack((vertices['red'], vertices['green'], vertices['blue'])).T.astype(np.uint8)

    # Print the shape of the loaded data
    print(f"Loaded {file_path}:")
    print(f"XYZ shape: {xyz.shape} (points, 3 coordinates)")
    print(f"RGB shape: {rgb.shape} (points, 3 channels)")
    return xyz, rgb

# Assign labels based on RGB colors
def assign_labels(rgb):
    labels = np.array([color_to_label.get(tuple(color), 0) for color in rgb])
    # Print the number of labels assigned
    print(f"Assigned labels: {np.unique(labels)}")
    return labels

# Save segmented point cloud to .ply
def save_segmented_ply(xyz, labels, output_path):
    colors = np.array([label_to_color[label] for label in labels])
    vertices = np.array([tuple(p) + tuple(c) for p, c in zip(xyz, colors)], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=True)
    ply.write(output_path)
    print(f"Saved segmented point cloud to {output_path}")

# Load model checkpoint
def load_checkpoint(model, checkpoint_path, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {checkpoint_path}, epoch {checkpoint['epoch']}, loss {checkpoint['loss']}")

# Segment point clouds
def segment_point_clouds(ply_files, checkpoint_path, output_dir="segmented_outputs", num_points=2048):
    os.makedirs(output_dir, exist_ok=True)
    model = PointNetPlusPlusSegmentation(num_classes=len(label_to_color))
    load_checkpoint(model, checkpoint_path, optimizer)

    for file_path in ply_files:
        xyz, rgb = load_point_cloud(file_path)

        # Print the original shape of the point cloud before any modifications
        print(f"Original points shape: {xyz.shape}")
        points = torch.tensor(xyz, dtype=torch.float32).unsqueeze(0).to(device)

        if points.shape[1] > num_points:
            indices = np.random.choice(points.shape[1], num_points, replace=False)
            points = points[:, indices, :]
            # Print the shape after random downsampling
            print(f"Downsampled points shape: {points.shape}")

        with torch.no_grad():
            outputs = model(points)
            predicted_labels = torch.argmax(outputs, dim=-1).cpu().numpy().flatten()

        # Print out the shape and unique labels for debugging
        print(f"Predicted labels shape: {predicted_labels.shape}")
        print(f"Unique predicted labels: {np.unique(predicted_labels)}")

        output_path = os.path.join(output_dir, os.path.basename(file_path).replace('.ply', '_segmented.ply'))
        save_segmented_ply(xyz[:num_points], predicted_labels, output_path)

if __name__ == "__main__":
    print('-----Hello------')
    checkpoint_path = "checkpoints/checkpoint_epoch_2.pth"  # Update with your checkpoint path
    ply_files = glob.glob("./data/*.ply")  # Add your .ply file paths
    segment_point_clouds(ply_files, checkpoint_path)
