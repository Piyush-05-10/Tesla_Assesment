import os
import glob

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.dataio.dataset import AugmentedCropDataset
from src.models.dinov2_backbone import DinoV2Backbone
from src.models.projection_head import ProjectionHead
from src.models.contrastive_loss import InfoNCELoss


def run(data_cfg, model_cfg):
    crops_dir = data_cfg["output"].get("crops_dir", "data/intermediate/crops")
    crop_files = sorted(glob.glob(os.path.join(crops_dir, "*_crops.npy")))
    if not crop_files:
        print("No crop files found. Run extract_embeddings first.", flush=True)
        return

    all_crops = np.concatenate([np.load(f) for f in crop_files])
    print(f"Loaded {len(all_crops)} crops for SSL fine-tuning", flush=True)

    ssl_cfg = model_cfg["ssl"]
    dinov2_cfg = model_cfg["dinov2"]
    proj_cfg = model_cfg["projection_head"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    backbone = DinoV2Backbone(
        variant=dinov2_cfg["variant"],
        device=device,
        use_cls_token=dinov2_cfg["use_cls_token"],
        batch_size=dinov2_cfg["batch_size"],
    )
    print("DINOv2 loaded", flush=True)

    print("Pre-computing DINOv2 features (one-time) ...", flush=True)
    all_features = backbone.extract(list(all_crops))
    features_tensor = torch.from_numpy(all_features).float()
    print(f"Features shape: {features_tensor.shape}", flush=True)

    projector = ProjectionHead(
        input_dim=dinov2_cfg["embedding_dim"],
        hidden_dim=proj_cfg["hidden_dim"],
        output_dim=proj_cfg["output_dim"],
        num_layers=proj_cfg["num_layers"],
    ).to(device)

    criterion = InfoNCELoss(temperature=ssl_cfg["temperature"])

    optimizer = torch.optim.Adam(
        projector.parameters(),
        lr=ssl_cfg["lr"],
    )

    dataset = TensorDataset(features_tensor)
    loader = DataLoader(dataset, batch_size=dinov2_cfg["batch_size"], shuffle=True, drop_last=True)

    print(f"Training projector for {ssl_cfg['epochs']} epochs ...", flush=True)

    for epoch in range(ssl_cfg["epochs"]):
        total_loss = 0.0
        count = 0

        for (feats,) in loader:
            feats = feats.to(device)

            noise1 = torch.randn_like(feats) * 0.05
            noise2 = torch.randn_like(feats) * 0.05

            z1 = projector(feats + noise1)
            z2 = projector(feats + noise2)

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / max(count, 1)
        print(f"Epoch {epoch+1}/{ssl_cfg['epochs']} | Loss: {avg_loss:.4f}", flush=True)

    save_path = os.path.join(data_cfg["output"]["embeddings_dir"], "projector.pt")
    torch.save(projector.state_dict(), save_path)
    print(f"Saved projector -> {save_path}", flush=True)


if __name__ == "__main__":
    from src.utils.logging import setup_logging
    setup_logging()

    with open("configs/data.yaml") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/model.yaml") as f:
        model_cfg = yaml.safe_load(f)

    run(data_cfg, model_cfg)
