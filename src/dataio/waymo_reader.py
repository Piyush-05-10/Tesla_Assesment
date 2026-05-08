import io
import os

import numpy as np
import pandas as pd
from PIL import Image

CAMERA_NAMES = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}


class WaymoParquetReader:

    def __init__(self, folder_path, camera_id=1, subsample=2, max_files=None, target_size=None):
        self.folder_path = folder_path
        self.camera_id = camera_id
        self.subsample = subsample
        self.target_size = target_size

        all_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".parquet"))
        if max_files is not None:
            all_files = all_files[:max_files]
        self.file_paths = [os.path.join(folder_path, f) for f in all_files]

    def read_sequence(self, file_path):
        df = pd.read_parquet(file_path)
        df_cam = df[df["key.camera_name"] == self.camera_id].copy()
        df_cam = df_cam.sort_values("key.frame_timestamp_micros")
        df_cam = df_cam.iloc[::self.subsample]

        sequence = []
        for _, row in df_cam.iterrows():
            raw = row["[CameraImageComponent].image"]
            ts = int(row["key.frame_timestamp_micros"])
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            if self.target_size is not None:
                img = img.resize(self.target_size, Image.BILINEAR)
            sequence.append((ts, np.asarray(img)))

        return sequence

    def iterate_all(self):
        for path in self.file_paths:
            seg = os.path.splitext(os.path.basename(path))[0]
            yield seg, self.read_sequence(path)

    def __len__(self):
        return len(self.file_paths)
