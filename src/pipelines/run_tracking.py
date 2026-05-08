import os

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from src.dataio.waymo_reader import WaymoParquetReader
from src.discovery.motion_segmentation import MotionSegmenter
from src.discovery.proposals import ProposalGenerator
from src.models.dinov2_backbone import DinoV2Backbone
from src.tracking.tracker import DeepSORTTracker
from src.utils.viz import draw_tracks


def run(data_cfg, model_cfg, tracking_cfg, save_video=True, max_sequences=None):
    reader = WaymoParquetReader(
        data_cfg["waymo"]["raw_parquet_dir"],
        camera_id=data_cfg["waymo"]["camera_name"],
        subsample=data_cfg["preprocessing"]["subsample_rate"],
        max_files=max_sequences or data_cfg["waymo"].get("max_files"),
    )

    md = tracking_cfg["motion_discovery"]
    segmenter = MotionSegmenter(
        flow_threshold=md["flow_threshold"],
        morph_kernel_size=md["morph_kernel_size"],
    )
    proposer = ProposalGenerator(
        min_area_ratio=md["min_area_ratio"],
        max_area_ratio=md["max_area_ratio"],
        expand_pixels=md["expand_pixels"],
    )

    backbone = DinoV2Backbone(
        variant=model_cfg["dinov2"]["variant"],
        device=model_cfg["dinov2"]["device"],
        use_cls_token=model_cfg["dinov2"]["use_cls_token"],
        batch_size=model_cfg["dinov2"]["batch_size"],
    )

    tm = tracking_cfg["track_management"]
    assoc = tracking_cfg["association"]
    tracker = DeepSORTTracker(
        max_age=tm["max_age"],
        min_hits=tm["min_hits"],
        appearance_weight=assoc["appearance_weight"],
        max_cost=assoc["max_cost_threshold"],
        embedding_momentum=tm["embedding_momentum"],
    )

    tracks_dir = data_cfg["output"]["tracks_dir"]
    os.makedirs(tracks_dir, exist_ok=True)

    for seg_name, frames in reader.iterate_all():
        print(f"Tracking segment {seg_name} – {len(frames)} frames")
        if len(frames) < 2:
            continue

        tracker.reset()

        h, w = frames[0][1].shape[:2]
        writer = None
        if save_video:
            vid_path = os.path.join(tracks_dir, f"{seg_name}_tracked.mp4")
            writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))

        all_tracks = []

        for i in tqdm(range(len(frames) - 1), desc=seg_name, leave=False):
            fa = frames[i][1]
            fb = frames[i + 1][1]

            mask = segmenter.compute_mask(fa, fb)
            boxes = proposer.extract_boxes(mask, fa.shape)

            if boxes:
                crops = proposer.extract_crops(fa, boxes)
                embeds = backbone.extract(crops)
                active = tracker.step(boxes, embeds)
            else:
                active = tracker.step([], np.empty((0, backbone.embedding_dim)))

            for trk in active:
                all_tracks.append({
                    "frame": i,
                    "track_id": int(trk[4]),
                    "box": trk[:4].tolist(),
                })

            if writer is not None:
                bgr = cv2.cvtColor(fa, cv2.COLOR_RGB2BGR)
                vis = draw_tracks(bgr, active)
                writer.write(vis)

        if writer is not None:
            writer.release()

        save_path = os.path.join(tracks_dir, f"{seg_name}_tracks.npz")
        if all_tracks:
            frames_arr = np.array([t["frame"] for t in all_tracks])
            ids_arr = np.array([t["track_id"] for t in all_tracks])
            boxes_arr = np.array([t["box"] for t in all_tracks])
            np.savez_compressed(save_path, frames=frames_arr, ids=ids_arr, boxes=boxes_arr)
        print(f"  Tracks -> {save_path}")


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
