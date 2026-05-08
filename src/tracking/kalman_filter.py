import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker:

    _next_id = 0

    def __init__(self, bbox):
        kf = KalmanFilter(dim_x=7, dim_z=4)

        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float64)

        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float64)

        kf.R[2:, 2:] *= 10.0
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 10.0
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01

        kf.x[:4] = self._xyxy_to_z(bbox)

        self.kf = kf
        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1

        self.hits = 0
        self.age = 0
        self.time_since_update = 0
        self.embedding = None

    @staticmethod
    def _xyxy_to_z(bbox):
        bbox = np.asarray(bbox, dtype=np.float64)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return np.array([
            bbox[0] + w / 2.0,
            bbox[1] + h / 2.0,
            w * h,
            w / max(h, 1e-6),
        ]).reshape((4, 1))

    @staticmethod
    def _z_to_xyxy(z):
        cx, cy, s, r = z.ravel()[:4]
        w = np.sqrt(max(s * r, 1e-6))
        h = s / max(w, 1e-6)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hits = 0
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox, embedding=None, momentum=0.9):
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(self._xyxy_to_z(bbox))

        if embedding is not None:
            if self.embedding is None:
                self.embedding = embedding.copy()
            else:
                self.embedding = momentum * self.embedding + (1 - momentum) * embedding
                norm = np.linalg.norm(self.embedding)
                if norm > 0:
                    self.embedding /= norm

    def get_state(self):
        return self._z_to_xyxy(self.kf.x)
