import hdbscan
import numpy as np
from sklearn.metrics import silhouette_score


class HDBSCANClusterer:

    def __init__(self, min_cluster_size=30, min_samples=10, metric="euclidean", cluster_selection_method="eom"):
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            prediction_data=True,
        )
        self.labels_ = None

    def fit_predict(self, embeddings):
        self.labels_ = self.model.fit_predict(embeddings)
        return self.labels_

    def predict(self, embeddings):
        labels, _ = hdbscan.approximate_predict(self.model, embeddings)
        return labels

    def evaluate(self, embeddings):
        labels = self.labels_ if self.labels_ is not None else self.predict(embeddings)
        valid = labels >= 0
        metrics = {}
        if valid.sum() > 1 and len(set(labels[valid])) > 1:
            metrics["silhouette_valid"] = float(silhouette_score(embeddings[valid], labels[valid]))
        metrics["noise_fraction"] = float((~valid).mean())
        return metrics
