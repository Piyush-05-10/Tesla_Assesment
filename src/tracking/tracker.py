import numpy as np

from .kalman_filter import KalmanBoxTracker
from .association import compute_cost_matrix, associate_detections


class DeepSORTTracker:

    def __init__(self, max_age=5, min_hits=2, appearance_weight=0.7, max_cost=1.5, embedding_momentum=0.9):
        self.max_age = max_age
        self.min_hits = min_hits
        self.appearance_weight = appearance_weight
        self.max_cost = max_cost
        self.embedding_momentum = embedding_momentum

        self.trackers = []
        self.frame_count = 0

    def reset(self):
        self.trackers.clear()
        self.frame_count = 0
        KalmanBoxTracker._next_id = 0

    def step(self, detections, embeddings):
        self.frame_count += 1

        for trk in self.trackers:
            trk.predict()

        stale = [i for i, t in enumerate(self.trackers) if np.any(np.isnan(t.get_state()))]
        for s in reversed(stale):
            self.trackers.pop(s)

        if self.trackers and detections:
            trk_boxes = [t.get_state() for t in self.trackers]
            trk_embs = [t.embedding for t in self.trackers]

            cost = compute_cost_matrix(
                trk_boxes, trk_embs,
                detections, embeddings,
                appearance_weight=self.appearance_weight,
            )
            matches, unmatched_trks, unmatched_dets = associate_detections(cost, max_cost=self.max_cost)
        else:
            matches = []
            unmatched_trks = list(range(len(self.trackers)))
            unmatched_dets = list(range(len(detections)))

        for ti, di in matches:
            self.trackers[ti].update(
                detections[di],
                embedding=embeddings[di],
                momentum=self.embedding_momentum,
            )

        for di in unmatched_dets:
            trk = KalmanBoxTracker(detections[di])
            trk.update(detections[di], embedding=embeddings[di])
            self.trackers.append(trk)

        active = []
        survivors = []

        for trk in self.trackers:
            if trk.time_since_update > self.max_age:
                continue
            survivors.append(trk)

            if trk.time_since_update < 1 and (
                trk.hits >= self.min_hits or self.frame_count <= self.min_hits
            ):
                box = trk.get_state()
                active.append(np.append(box, trk.id))

        self.trackers = survivors

        if active:
            return np.stack(active)
        return np.empty((0, 5))
