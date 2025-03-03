import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channels, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp = nn.Sequential(
            *[nn.Conv1d(in_channels if i == 0 else mlp[i - 1], mlp[i], 1) for i in range(len(mlp))]
        )

    def forward(self, xyz1, xyz2, features1, features2):
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        # Compute pairwise distances between xyz1 and xyz2
        dists = torch.cdist(xyz1, xyz2)

        # Get indices of the 3 nearest neighbors (B, N, 3)
        idx = dists.argsort(dim=-1)[:, :, :3]

        # Initialize an empty tensor for interpolated features (B, N, F2)
        interpolated_features = torch.zeros(B, N, features2.shape[-1], device=features2.device)

        # Gather features of the 3 nearest neighbors by direct indexing
        for i in range(3):
            batch_indices = torch.arange(B, device=features2.device).view(-1, 1)
            interpolated_features += features2[batch_indices, idx[:, :, i]]

        # Average the features across the 3 neighbors
        interpolated_features /= 3

        # Concatenate with features1 if available
        if features1 is not None:
            new_features = torch.cat([features1, interpolated_features], dim=2)
        else:
            new_features = interpolated_features
        print(f"new_features shape before permute: {new_features.shape}")

        # Apply MLP after permuting dimensions to (B, C, N)
        new_features2 = self.mlp(new_features.permute(0, 2, 1))
        print(f"MLP first layer weight shape: {self.mlp[0].weight.shape}")

        return new_features2
