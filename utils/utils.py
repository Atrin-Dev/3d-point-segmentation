import torch
import torch.nn as nn
import torch.nn.functional as F


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
    grouped_xyz = new_xyz.unsqueeze(2) - xyz.unsqueeze(1)
    distances = torch.sum(grouped_xyz ** 2, -1)
    grouped_idx = torch.argsort(distances, dim=-1)[:, :, :self.nsample]
    grouped_features = features.gather(1, grouped_idx.unsqueeze(-1).expand(-1, -1, -1, features.shape[-1]))
    return grouped_xyz, grouped_features

