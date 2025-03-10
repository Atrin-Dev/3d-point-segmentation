# 3d-point-segmentation
This repository implements PointNet for 3D point cloud segmentation, focusing on segmenting and classifying 3D object parts. It includes data preprocessing, model training, and visualization of segmented point clouds using libraries like Open3D.

## Point Cloud Processing Pipeline
PointNet++ builds on PointNet by introducing hierarchical feature learning through sampling and grouping layers. The core pipeline consists of
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


## train

Run the following command to train a point cloud:

```bash
python main.py --mode train
```
## Inference

Run the following command to test segmenting a point cloud:

```bash
python main.py --mode eval
```
