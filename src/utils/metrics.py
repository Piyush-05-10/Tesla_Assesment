import numpy as np


def iou_matrix(boxes_a, boxes_b):
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def detection_recall(gt_boxes, pred_boxes, iou_threshold=0.5):
    if len(gt_boxes) == 0:
        return 1.0
    if len(pred_boxes) == 0:
        return 0.0
    ious = iou_matrix(gt_boxes, pred_boxes)
    matched = (ious.max(axis=1) >= iou_threshold).sum()
    return float(matched) / len(gt_boxes)


def cluster_quality(embeddings, labels):
    from sklearn.metrics import silhouette_score, calinski_harabasz_score

    valid = labels >= 0
    metrics = {"n_clusters": float(len(set(labels[valid])))}
    if valid.sum() > 1 and len(set(labels[valid])) > 1:
        metrics["silhouette"] = float(silhouette_score(embeddings[valid], labels[valid]))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(embeddings[valid], labels[valid]))
    return metrics
