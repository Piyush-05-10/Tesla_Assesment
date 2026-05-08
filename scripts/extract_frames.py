import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse

import cv2
import yaml
from tqdm import tqdm

from src.dataio.waymo_reader import WaymoParquetReader
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq", type=int, default=1)
    args = parser.parse_args()

    setup_logging()

    with open("configs/data.yaml") as f:
        cfg = yaml.safe_load(f)

    reader = WaymoParquetReader(
        cfg["waymo"]["raw_parquet_dir"],
        camera_id=cfg["waymo"]["camera_name"],
        subsample=cfg["preprocessing"]["subsample_rate"],
        max_files=args.max_seq,
    )

    out_root = cfg["output"]["frames_dir"]

    for seg_name, frames in reader.iterate_all():
        seg_dir = os.path.join(out_root, seg_name)
        os.makedirs(seg_dir, exist_ok=True)

        for i, (ts, img) in enumerate(tqdm(frames, desc=seg_name)):
            path = os.path.join(seg_dir, f"{i:05d}_{ts}.png")
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print(f"Saved {len(frames)} frames -> {seg_dir}")


if __name__ == "__main__":
    main()
