import torch
import torch.nn as nn
from .pointnet_set_abstraction import PointNetSetAbstraction
from .pointnet_feature_propagation import PointNetFeaturePropagation

class PointNetPlusPlusSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(PointNetPlusPlusSegmentation, self).__init__()

        # Set in_channels based on XYZ (3 channels for coordinates)
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channels=3, mlp=[64, 128, 256])
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channels=256 + 3, mlp=[128, 256, 512])

        self.fp1 = PointNetFeaturePropagation(518, [256, 128])
        self.fp2 = PointNetFeaturePropagation(131, [128, 128, 128])

        self.fc = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, points):
        # points should be (B, N, C) where C >= 3 (x, y, z, [features...])
        xyz = points[:, :, :3]  # Extract XYZ coordinates
        features = points[:, :, 3:] if points.size(-1) > 3 else xyz   # Additional features if available

        # Pass through PointNetSetAbstraction layers
        new_xyz1, new_features1, fps_idx1 = self.sa1(xyz, features)  # Unpack three values (new_xyz1, new_features1, fps_idx1)
        new_features1 = new_features1.permute(0, 2, 1).contiguous()  # Shape: [B, N, C]
        new_xyz2, new_features2, fps_idx2 = self.sa2(new_xyz1, new_features1)  # Same for the second set abstraction layer
        new_features2 = new_features2.permute(0, 2, 1).contiguous()  # Shape: [B, N, C]

        # Pass through PointNetFeaturePropagation layers
        up_features1 = self.fp1(new_xyz1, new_xyz2, new_features1, new_features2)
        up_features1 = up_features1.permute(0, 2, 1).contiguous()  # Shape: [B, N, C]
        up_features2 = self.fp2(xyz, new_xyz1, features, up_features1)

        # Final segmentation output
        logits = self.fc(up_features2)
        return logits.permute(0, 2, 1)
