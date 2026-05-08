import cv2
import numpy as np
import matplotlib.pyplot as plt


_PALETTE = [
    (75, 25, 230), (75, 180, 60), (25, 225, 255), (200, 130, 0),
    (49, 130, 245), (180, 30, 145), (240, 240, 70), (230, 50, 240),
    (60, 245, 210), (212, 190, 250), (80, 120, 200), (255, 255, 0),
    (52, 209, 183), (128, 64, 64), (64, 0, 128), (0, 128, 64),
    (128, 128, 0), (0, 0, 128), (192, 192, 192), (255, 128, 0),
]


def _colour(idx):
    return _PALETTE[int(idx) % len(_PALETTE)]


def draw_tracks(frame_bgr, tracks, cluster_labels=None, thickness=2, font_scale=0.6):
    vis = frame_bgr.copy()
    for i, trk in enumerate(tracks):
        x1, y1, x2, y2, tid = trk.astype(int)
        cid = int(cluster_labels[i]) if cluster_labels is not None else int(tid)
        col = _colour(cid)

        cv2.rectangle(vis, (x1, y1), (x2, y2), col, thickness)
        label = f"ID:{tid}"
        if cluster_labels is not None:
            label += f" C:{cid}"
        cv2.putText(vis, label, (x1, y1 - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, thickness)
    return vis


def make_cluster_grid(crops, labels, grid_size=20, cell_px=80, save_path=None):
    unique = sorted(set(labels))
    if -1 in unique:
        unique.remove(-1)

    if not unique:
        return None

    n_clusters = len(unique)
    fig, axes = plt.subplots(n_clusters, grid_size, figsize=(grid_size * 1.2, n_clusters * 1.5), dpi=80)
    if n_clusters == 1:
        axes = [axes]

    for row, cid in enumerate(unique):
        idxs = np.where(labels == cid)[0]
        np.random.shuffle(idxs)
        for col in range(grid_size):
            ax = axes[row][col] if grid_size > 1 else axes[row]
            if col < len(idxs):
                ax.imshow(crops[idxs[col]])
            ax.axis("off")
            if col == 0:
                ax.set_title(f"C{cid}", fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return None

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return buf
