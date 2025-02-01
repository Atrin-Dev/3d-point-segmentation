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

        dists = torch.cdist(xyz1, xyz2)
        idx = dists.argsort(dim=-1)[:, :, :3]
        interpolated_features = features2.gather(1, idx.unsqueeze(-1).expand(-1, -1, -1, features2.shape[-1])).mean(
            dim=2)

        if features1 is not None:
            new_features = torch.cat([features1, interpolated_features], dim=1)
        else:
            new_features = interpolated_features

        return self.mlp(new_features)
