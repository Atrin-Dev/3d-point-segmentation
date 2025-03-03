# 3d-point-segmentation
This repository implements PointNet for 3D point cloud segmentation, focusing on segmenting and classifying 3D object parts. It includes data preprocessing, model training, and visualization of segmented point clouds using libraries like Open3D.

## Point Cloud Processing Pipeline

```
Raw Point Cloud (N x 3)
        │
        ▼
Sampling and Grouping (FPS + Ball Query)
        │
        ▼
Set Abstraction (SA) Layers
        │
        ▼
Multi-scale Grouping (MSG) / Multi-resolution Grouping (MRG)
        │
        ▼
Feature Propagation (FP) Layers
        │
        ▼
Point-wise Classification or Segmentation
```


## Usage

Run the following command to train a point cloud:

```bash
python main.py --mode train

