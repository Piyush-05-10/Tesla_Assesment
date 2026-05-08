import os
import glob

import numpy as np
import yaml

from src.clustering.kmeans_cluster import KMeansClusterer
from src.clustering.hdbscan_cluster import HDBSCANClusterer
from src.utils.viz import make_cluster_grid


def run(data_cfg, cluster_cfg):
    emb_dir = data_cfg["output"]["embeddings_dir"]
    out_dir = data_cfg["output"]["clusters_dir"]
    os.makedirs(out_dir, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(emb_dir, "*.npz")))
    if not npz_files:
        print(f"No embedding files found in {emb_dir}")
        return

    all_embs = []
    for npz_path in npz_files:
        data = np.load(npz_path)
        all_embs.append(data["embeddings"])

    E = np.concatenate(all_embs)
    print(f"Loaded {len(E)} embeddings from {len(npz_files)} files")

    km_cfg = cluster_cfg["kmeans"]
    km = KMeansClusterer(
        n_clusters=km_cfg["n_clusters"],
        n_init=km_cfg["n_init"],
        max_iter=km_cfg["max_iter"],
        random_state=km_cfg["random_state"],
    )
    labels_km = km.fit_predict(E)
    print(f"KMeans metrics: {km.evaluate(E)}")
    np.save(os.path.join(out_dir, "labels_kmeans.npy"), labels_km)

    hdb_cfg = cluster_cfg["hdbscan"]
    hdb = HDBSCANClusterer(
        min_cluster_size=hdb_cfg["min_cluster_size"],
        min_samples=hdb_cfg["min_samples"],
        metric=hdb_cfg["metric"],
        cluster_selection_method=hdb_cfg["cluster_selection_method"],
    )
    labels_hdb = hdb.fit_predict(E)
    print(f"HDBSCAN metrics: {hdb.evaluate(E)}")
    np.save(os.path.join(out_dir, "labels_hdbscan.npy"), labels_hdb)

    crops_dir = data_cfg["output"].get("crops_dir", "data/intermediate/crops")
    crop_files = sorted(glob.glob(os.path.join(crops_dir, "*_crops.npy")))
    if crop_files:
        all_crops = np.concatenate([np.load(f) for f in crop_files])
        if len(all_crops) == len(labels_hdb):
            grid_size = cluster_cfg["evaluation"].get("sample_grid_size", 20)
            make_cluster_grid(list(all_crops), labels_hdb, grid_size=grid_size,
                              save_path=os.path.join(out_dir, "hdbscan_grid.png"))
            make_cluster_grid(list(all_crops), labels_km, grid_size=grid_size,
                              save_path=os.path.join(out_dir, "kmeans_grid.png"))

    print(f"Clustering complete -> {out_dir}")


if __name__ == "__main__":
    from src.utils.logging import setup_logging
    setup_logging()

    with open("configs/data.yaml") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/clustering.yaml") as f:
        cluster_cfg = yaml.safe_load(f)

    run(data_cfg, cluster_cfg)
