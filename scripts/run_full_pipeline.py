import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import yaml

from src.utils.logging import setup_logging
from src.pipelines import extract_embeddings, run_clustering, run_tracking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq", type=int, default=None)
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--skip-cluster", action="store_true")
    parser.add_argument("--skip-track", action="store_true")
    args = parser.parse_args()

    setup_logging()

    with open("configs/data.yaml") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/clustering.yaml") as f:
        cluster_cfg = yaml.safe_load(f)
    with open("configs/tracking.yaml") as f:
        tracking_cfg = yaml.safe_load(f)

    if not args.skip_embed:
        print("\n" + "=" * 60)
        print("STAGE 1: Extracting DINOv2 embeddings")
        print("=" * 60)
        extract_embeddings.run(data_cfg, model_cfg, tracking_cfg, max_sequences=args.max_seq)

    if not args.skip_cluster:
        print("\n" + "=" * 60)
        print("STAGE 2: Running KMeans + HDBSCAN clustering")
        print("=" * 60)
        run_clustering.run(data_cfg, cluster_cfg)

    if not args.skip_track:
        print("\n" + "=" * 60)
        print("STAGE 3: Running DeepSORT tracking")
        print("=" * 60)
        run_tracking.run(data_cfg, model_cfg, tracking_cfg, max_sequences=args.max_seq)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
