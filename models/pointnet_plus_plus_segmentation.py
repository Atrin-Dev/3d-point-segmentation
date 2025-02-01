import torch
import torch.nn as nn
from models.pointnet_set_abstraction import PointNetSetAbstraction
from models.pointnet_feature_propagation import PointNetFeaturePropagation


class PointNetPlusPlusSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(PointNetPlusPlusSegmentation, self).__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, 3, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128, [128, 128, 256])
        self.fp1 = PointNetFeaturePropagation(384, [256, 128])
        self.fp2 = PointNetFeaturePropagation(131, [128, 128, 128])
        self.fc = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, xyz):
        B, N, C = xyz.shape
        features = xyz.permute(0, 2, 1)
        new_xyz1, new_features1, fps_idx1 = self.sa1(xyz, features)
        new_xyz2, new_features2, fps_idx2 = self.sa2(new_xyz1, new_features1)

        up_features1 = self.fp1(new_xyz1, new_xyz2, new_features1, new_features2)
        up_features2 = self.fp2(xyz, new_xyz1, features, up_features1)
        logits = self.fc(up_features2)
        return logits.permute(0, 2, 1)
