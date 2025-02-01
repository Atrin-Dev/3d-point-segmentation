import torch
from models.pointnet_set_abstraction import PointNetSetAbstraction
from models.pointnet_feature_propagation import PointNetFeaturePropagation
from models.pointnet_plus_plus_segmentation import PointNetPlusPlusSegmentation

def main():
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example synthetic data: A batch of 3D points with shape (batch_size, num_points, 3)
    batch_size = 2
    num_points = 1024
    xyz = torch.rand(batch_size, num_points, 3).to(device)  # Random 3D points

    # Initialize the model
    num_classes = 10  # For segmentation, you can set the number of classes
    model = PointNetPlusPlusSegmentation(num_classes).to(device)

    # Forward pass through the model
    logits = model(xyz)

    # Print the output shape
    print(f"Output logits shape: {logits.shape}")

    # Loss function: CrossEntropyLoss is commonly used for segmentation tasks
    loss_fn = torch.nn.CrossEntropyLoss()

    # Example target: random labels for the points (batch_size, num_points)
    # These should be the ground truth segmentation labels for each point
    target = torch.randint(0, num_classes, (batch_size, num_points)).to(device)

    # Calculate the loss
    loss = loss_fn(logits.view(-1, num_classes), target.view(-1))

    # Print the loss value
    print(f"Loss: {loss.item()}")

if __name__ == "__main__":
    main()
