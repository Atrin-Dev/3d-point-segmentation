import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import farthest_point_sampling, index_points, query_and_group


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channels, mlp):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp = nn.Sequential(
            *[nn.Conv2d(in_channels if i == 0 else mlp[i - 1], mlp[i], 1) for i in range(len(mlp))]
        )
        self.bn = nn.BatchNorm2d(mlp[-1])

    def forward(self, xyz, features):
        fps_idx = self.farthest_point_sampling(xyz, self.npoint)
        new_xyz = self.index_points(xyz, fps_idx)
        grouped_xyz, grouped_features = self.query_and_group(xyz, features, new_xyz)
        grouped_features = self.mlp(grouped_features)
        new_features = F.max_pool2d(grouped_features, kernel_size=[self.nsample, 1]).squeeze(-1)
        return new_xyz, new_features, fps_idx