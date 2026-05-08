import os

import numpy as np
import yaml
from tqdm import tqdm

from src.dataio.waymo_reader import WaymoParquetReader
from src.discovery.motion_segmentation import MotionSegmenter
from src.discovery.proposals import ProposalGenerator
from src.models.dinov2_backbone import DinoV2Backbone


def run(data_cfg, model_cfg, tracking_cfg, max_sequences=None):
    reader = WaymoParquetReader(
        data_cfg["waymo"]["raw_parquet_dir"],
        camera_id=data_cfg["waymo"]["camera_name"],
        subsample=data_cfg["preprocessing"]["subsample_rate"],
        max_files=max_sequences or data_cfg["waymo"].get("max_files"),
    )

    md_cfg = tracking_cfg["motion_discovery"]
    segmenter = MotionSegmenter(
        flow_threshold=md_cfg["flow_threshold"],
        morph_kernel_size=md_cfg["morph_kernel_size"],
    )
    proposer = ProposalGenerator(
        min_area_ratio=md_cfg["min_area_ratio"],
        max_area_ratio=md_cfg["max_area_ratio"],
        expand_pixels=md_cfg["expand_pixels"],
    )

    backbone = DinoV2Backbone(
        variant=model_cfg["dinov2"]["variant"],
        device=model_cfg["dinov2"]["device"],
        use_cls_token=model_cfg["dinov2"]["use_cls_token"],
        batch_size=model_cfg["dinov2"]["batch_size"],
    )

    out_dir = data_cfg["output"]["embeddings_dir"]
    os.makedirs(out_dir, exist_ok=True)

    for seg_name, frames in reader.iterate_all():
        print(f"Segment {seg_name} – {len(frames)} frames")
        if len(frames) < 2:
            continue

        seg_embeddings = []
        seg_boxes = []
        seg_frame_ids = []
        all_crops = []

        for i in tqdm(range(len(frames) - 1), desc=seg_name, leave=False):
            fa = frames[i][1]
            fb = frames[i + 1][1]

            mask = segmenter.compute_mask(fa, fb)
            boxes = proposer.extract_boxes(mask, fa.shape)

            if not boxes:
                continue

            crops = proposer.extract_crops(fa, boxes)
            embeds = backbone.extract(crops)

            for j, (box, emb) in enumerate(zip(boxes, embeds)):
                seg_embeddings.append(emb)
                seg_boxes.append(box)
                seg_frame_ids.append(i)
                all_crops.append(crops[j])

        if seg_embeddings:
            E = np.stack(seg_embeddings)
            save_path = os.path.join(out_dir, f"{seg_name}.npz")
            np.savez_compressed(
                save_path,
                embeddings=E,
                boxes=np.array(seg_boxes),
                frame_ids=np.array(seg_frame_ids),
            )
            print(f"  Saved {len(E)} embeddings -> {save_path}")

            crops_dir = data_cfg["output"].get("crops_dir", "data/intermediate/crops")
            os.makedirs(crops_dir, exist_ok=True)
            np.save(os.path.join(crops_dir, f"{seg_name}_crops.npy"), np.stack(all_crops))


if __name__ == "__main__":
    from src.utils.logging import setup_logging
    setup_logging()

    with open("configs/data.yaml") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/tracking.yaml") as f:
        tracking_cfg = yaml.safe_load(f)

    run(data_cfg, model_cfg, tracking_cfg)
