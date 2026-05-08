import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import glob
import argparse

import numpy as np
import yaml
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.utils.logging import setup_logging


def tsne_plot(embeddings, labels, save_path):
    print(f"Computing t-SNE on {len(embeddings)} embeddings")
    coords = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    valid = labels >= 0
    plt.scatter(coords[valid, 0], coords[valid, 1], c=labels[valid], cmap="tab20", s=6, alpha=0.7)
    if (~valid).any():
        plt.scatter(coords[~valid, 0], coords[~valid, 1], c="gray", s=3, alpha=0.3, label="noise")
        plt.legend()
    plt.colorbar(label="Cluster ID")
    plt.title("t-SNE of DINOv2 embeddings coloured by cluster")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved t-SNE -> {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters-dir", default=None)
    args = parser.parse_args()

    setup_logging()

    with open("configs/data.yaml") as f:
        data_cfg = yaml.safe_load(f)

    emb_dir = data_cfg["output"]["embeddings_dir"]
    clusters_dir = args.clusters_dir or data_cfg["output"]["clusters_dir"]

    npz_files = sorted(glob.glob(os.path.join(emb_dir, "*.npz")))
    if not npz_files:
        print("No embedding files found")
        return

    E = np.concatenate([np.load(f)["embeddings"] for f in npz_files])

    for label_file in ["labels_hdbscan.npy", "labels_kmeans.npy"]:
        path = os.path.join(clusters_dir, label_file)
        if not os.path.exists(path):
            continue
        labels = np.load(path)
        name = label_file.replace("labels_", "").replace(".npy", "")
        tsne_plot(E, labels, os.path.join(clusters_dir, f"tsne_{name}.png"))

    print("Visualisation complete.")


if __name__ == "__main__":
    main()
