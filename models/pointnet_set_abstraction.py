import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channels, mlp):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        # Modify the output channels of the MLP layers to 259
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mlp[0], kernel_size=1),
            nn.BatchNorm2d(mlp[0]),
            nn.ReLU(),
            nn.Conv2d(mlp[0], mlp[1], kernel_size=1),
            nn.BatchNorm2d(mlp[1]),
            nn.ReLU(),
            nn.Conv2d(mlp[1], 259, kernel_size=1),  # Set final output to 259 channels
            nn.BatchNorm2d(259),
            nn.ReLU()
        )

    def forward(self, xyz, features):
        """
        :param xyz: Input point cloud (B, N, 3)
        :param features: Features corresponding to the points (B, N, C)
        :return: new_xyz: Downsampled points (B, M, 3)
                new_features: Associated features (B, M, C)
                fps_idx: Indices of selected points (B, M)
        """

        # Ensure that the input features are permuted correctly
        grouped_features = features.permute(0, 2, 1).contiguous()  # Shape: [B, C, N]

        # Apply MLP layers (which expects [B, C, N] input)
        grouped_features = self.mlp(grouped_features.unsqueeze(3))  # Unsqueeze to [B, C, N, 1] for Conv2d

        # Squeeze the last dimension to get the output shape [B, C, N]
        grouped_features = grouped_features.squeeze(3)  # [B, C, N]

        # Ensure we have the correct output features to return
        new_xyz = xyz[:, :self.npoint, :]  # Take the first npoint points as new centers
        new_features = grouped_features[:, :, :self.npoint]  # Corresponding features for new centers
        fps_idx = torch.arange(self.npoint).expand(xyz.size(0), self.npoint)  # Indices of sampled points
        return new_xyz, new_features, fps_idx

    def farthest_point_sampling(self, xyz, npoint):
        B, N, _ = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
        distance = torch.ones(B, N, device=xyz.device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)

        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[torch.arange(B), farthest].unsqueeze(1)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            distance = torch.min(distance, dist)
            farthest = torch.max(distance, -1)[1]
        return centroids

    def index_points(self, points, idx):
        B = points.shape[0]
        batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(B, 1).repeat(1, idx.shape[1])
        new_points = points[batch_indices, idx, :]
        return new_points

    def query_and_group(self, xyz, features, new_xyz):
        B, N, C = xyz.shape
        _, S, _ = new_xyz.shape

        # Calculate distances between new_xyz and all xyz
        grouped_xyz = new_xyz.unsqueeze(2) - xyz.unsqueeze(1)  # (B, S, N, C)
        distances = torch.sum(grouped_xyz ** 2, -1)  # (B, S, N)

        # Get indices of the nearest nsample points
        grouped_idx = torch.argsort(distances, dim=-1)[:, :, :self.nsample] # (B, S, nsample)

        # Gather features using the indices
        grouped_features = index_points(features, grouped_idx)  # (B, S, nsample, C)
        return grouped_xyz, grouped_features

"""
# Example usage
if __name__ == "__main__":
    # Dummy data to simulate point cloud and features
    B, N, C = 16, 2048, 3  # Batch size, number of points, and channels
    xyz = torch.rand(B, N, 3)  # Point cloud data (B, N, 3)
    feature
"""