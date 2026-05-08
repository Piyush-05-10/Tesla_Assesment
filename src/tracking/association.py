import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(bb_a, bb_b):
    xx1 = max(bb_a[0], bb_b[0])
    yy1 = max(bb_a[1], bb_b[1])
    xx2 = min(bb_a[2], bb_b[2])
    yy2 = min(bb_a[3], bb_b[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = (bb_a[2] - bb_a[0]) * (bb_a[3] - bb_a[1])
    area_b = (bb_b[2] - bb_b[0]) * (bb_b[3] - bb_b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


def cosine_distance(a, b):
    return 1.0 - float(np.dot(a, b))


def compute_cost_matrix(track_boxes, track_embeddings, det_boxes, det_embeddings, appearance_weight=0.7):
    n_trk = len(track_boxes)
    n_det = len(det_boxes)
    cost = np.zeros((n_trk, n_det), dtype=np.float64)

    for i in range(n_trk):
        for j in range(n_det):
            if track_embeddings[i] is not None:
                c_app = cosine_distance(track_embeddings[i], det_embeddings[j])
            else:
                c_app = 1.0

            c_mot = 1.0 - iou(track_boxes[i], np.asarray(det_boxes[j]))
            cost[i, j] = appearance_weight * c_app + (1 - appearance_weight) * c_mot

    return cost


def associate_detections(cost_matrix, max_cost=1.5):
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_trk = set()
    matched_det = set()
    matches = []

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > max_cost:
            continue
        matches.append((r, c))
        matched_trk.add(r)
        matched_det.add(c)

    unmatched_tracks = [i for i in range(cost_matrix.shape[0]) if i not in matched_trk]
    unmatched_dets = [j for j in range(cost_matrix.shape[1]) if j not in matched_det]

    return matches, unmatched_tracks, unmatched_dets
