# Waymo Unsupervised Driving Perception

Unsupervised object discovery, tracking, and clustering on Waymo Open Dataset driving videos.

## Overview

This system processes raw driving video from the Waymo Open Dataset without any labels or annotations. It discovers moving objects via optical flow, extracts visual representations using DINOv2, clusters objects into semantic categories using HDBSCAN/KMeans, and tracks instances across frames using a DeepSORT-style tracker.

## Pipeline

1. Data Ingestion - Read Waymo v2 camera_image parquet files, extract front-camera RGB frames
2. Object Discovery - Farneback optical flow to detect motion, morphological cleanup, bounding box proposals
3. Representation Learning - DINOv2 ViT-S/14 extracts 384-dim L2-normalized embeddings per crop
4. Clustering - KMeans and HDBSCAN group embeddings into semantic categories
5. Tracking - Kalman filter motion prediction + DINOv2 cosine similarity + Hungarian matching

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python scripts/run_full_pipeline.py --max-seq 1
python scripts/visualize_results.py
```

## Structure

```
configs/         YAML configs for data, model, clustering, tracking
src/dataio/      Parquet reader, PyTorch datasets
src/discovery/   Motion segmentation, proposal generation
src/models/      DINOv2 backbone, projection head
src/clustering/  KMeans and HDBSCAN wrappers
src/tracking/    Kalman filter, association, DeepSORT tracker
src/utils/       Visualization, logging, metrics
src/pipelines/   Stage orchestrators
scripts/         CLI entry points
```

## Outputs

- Embeddings: data/processed/embeddings/*.npz
- Cluster labels: data/processed/clusters/labels_*.npy
- Tracked videos: data/processed/tracks/*_tracked.mp4
- t-SNE plots: data/processed/clusters/tsne_*.png
