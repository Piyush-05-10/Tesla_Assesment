import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score


class KMeansClusterer:

    def __init__(self, n_clusters=8, n_init="auto", max_iter=300, random_state=42):
        self.model = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.labels_ = None

    def fit_predict(self, embeddings):
        self.labels_ = self.model.fit_predict(embeddings)
        return self.labels_

    def predict(self, embeddings):
        return self.model.predict(embeddings)

    def evaluate(self, embeddings):
        labels = self.labels_ if self.labels_ is not None else self.predict(embeddings)
        metrics = {}
        if len(set(labels)) > 1:
            metrics["silhouette"] = float(silhouette_score(embeddings, labels))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(embeddings, labels))
        return metrics
