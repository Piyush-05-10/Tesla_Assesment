import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataio.dataset import CropDataset


class DinoV2Backbone:

    def __init__(self, variant="dinov2_vits14", device="auto", use_cls_token=True, batch_size=64):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.use_cls_token = use_cls_token
        self.batch_size = batch_size

        self.model = torch.hub.load("facebookresearch/dinov2", variant, verbose=False)
        self.model.to(self.device).eval()
        self.embedding_dim = self.model.embed_dim

    @torch.no_grad()
    def extract(self, crops):
        if not crops:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        ds = CropDataset(crops)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        parts = []
        for batch in loader:
            batch = batch.to(self.device)

            if self.use_cls_token:
                feats = self.model(batch)
            else:
                out = self.model.forward_features(batch)
                patch_tokens = out["x_norm_patchtokens"]
                feats = patch_tokens.mean(dim=1)

            feats = F.normalize(feats, dim=1)
            parts.append(feats.cpu().numpy())

        return np.concatenate(parts, axis=0)
