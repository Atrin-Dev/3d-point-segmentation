import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from plyfile import PlyData
import glob

class PointCloudPartSegmentationDataset(Dataset):
    def __init__(self, file_paths, color_to_label, num_points=2048, augment=False):
        """
        Args:
            file_paths (list): List of paths to PLY files.
            num_points (int): Number of points to sample for each point cloud.
        """
        self.file_paths = file_paths
        self.color_to_label = {tuple(k): v for k, v in color_to_label.items()}  # Ensure keys are tuples
        self.num_points = num_points
        self.augment = augment

    def _load_point_cloud(self, file_path):
        """Load a PLY file and extract points (x, y, z) and colors (r, g, b)."""
        ply_data = PlyData.read(file_path)
        vertices = ply_data['vertex'].data

        # Extract xyz and rgb
        xyz = np.vstack((vertices['x'], vertices['y'], vertices['z'])).T
        rgb = np.vstack((vertices['red'], vertices['green'], vertices['blue'])).T.astype(np.uint8)

        # If labels exist, extract them (optional)
        #labels = vertices['label'] if 'label' in vertices.dtype.names else np.zeros(xyz.shape[0])

        return xyz, rgb

    def _assign_labels(self, rgb):
        """Maps RGB colors to label indices."""
        if rgb is None:
            raise ValueError("Point cloud data does not contain color information.")
        labels = np.array([self.color_to_label.get(tuple(color), 0) for color in rgb])
        return labels

    def _random_sample(self, xyz, rgb, labels):
        """Sample num_points points from the point cloud (upsample or downsample)."""
        N = xyz.shape[0]

        if N >= self.num_points:
            # Downsample using random sampling or FPS (implement FPS later if needed)
            indices = np.random.choice(N, self.num_points, replace=False)
        else:
            # Upsample by sampling with replacement
            indices = np.random.choice(N, self.num_points, replace=True)

        return xyz[indices], rgb[indices], labels[indices]

    def _augment(self, xyz):
        """Applies random jittering to point coordinates."""
        jitter = np.clip(0.02 * np.random.randn(*xyz.shape), -0.05, 0.05)
        xyz += jitter
        return xyz

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        xyz, rgb = self._load_point_cloud(file_path)
        labels = self._assign_labels(rgb)

        xyz, rgb, labels = self._random_sample(xyz, rgb, labels)

        if self.augment:
            xyz = self._augment(xyz)

        # Convert to tensors
        points = torch.from_numpy(np.hstack((xyz, rgb / 255.0)).astype(np.float32))  # (num_points, 6)
        # Convert to tensors
        # points = torch.from_numpy(xyz.astype(np.float32))  # Only XYZ coordinates (num_points, 3)
        labels = torch.from_numpy(labels.astype(np.int64))  # (num_points,)

        return points, labels

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

# Example usage:
# Assuming you have a folder with PLY files
#ply_files = glob.glob("/content/drive/My Drive/Course/PointCloud/Git_3DSeg/dataset.py")
ply_files = glob.glob("/content/drive/My Drive/Course/PointCloud/DataSet/Extract/*.ply")
dataset = PointCloudPartSegmentationDataset(ply_files, color_to_label, num_points=2048, augment=True)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Example of getting one batch
for points, labels in dataset:
    print("Points shape:", points.shape)  # (2048, 6)
    print("Labels shape:", labels.shape)  # (2048,)
    break


