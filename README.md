# 3d-point-segmentation
This repository implements PointNet for 3D point cloud segmentation, focusing on segmenting and classifying 3D object parts. It includes data preprocessing, model training, and visualization of segmented point clouds using libraries like Open3D, with applications in robotics, autonomous vehicles, and AR/VR.

## Usage

Run the following command to segment a point cloud:

```bash
python segment.py --input data/sample1.ply --checkpoint path/to/checkpoint.pth --label_map path/to/labels.json --output results/segmented_sample1.ply

