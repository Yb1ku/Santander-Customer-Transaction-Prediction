from preprocessing import (
RobustScalerWrapper, StandardScalerWrapper,
PCAWrapper, RowStatsCalculator
)
from models.clustering import ClusterWrapper
import numpy as np
import joblib

class FullPipeline:
    def __init__(self, model=None):
        self.robust_scaler = RobustScalerWrapper()
        self.standard_scaler = StandardScalerWrapper()
        self.pca = PCAWrapper(n_components=0.95)
        self.row_stats = RowStatsCalculator()
        self.cluster_model = ClusterWrapper(n_clusters=2)
        self.model = model

    def fit(self, x, y):
        x_scaled = self.robust_scaler.fit_transform(x)
        x_pca = self.pca.fit_transform(x_scaled)
        x_row_stats = self.row_stats.transform(x_scaled)
        x_concat = np.hstack((x_pca, x_row_stats))
        x_standardized = self.standard_scaler.fit_transform(x_concat)

        self.cluster_model.fit(x_pca)
        cluster_ids = self.cluster_model.predict(x_pca)

        x_final = np.hstack((x_standardized, cluster_ids.reshape(-1, 1)))
        if self.model:
            self.model.fit(x_final, y)

        return self

    def transform(self, x):
        x_scaled = self.robust_scaler.transform(x)
        x_pca = self.pca.transform(x_scaled)
        x_row_stats = self.row_stats.transform(x_scaled)
        x_concat = np.hstack((x_pca, x_row_stats))
        x_standardized = self.standard_scaler.transform(x_concat)
        cluster_ids = self.cluster_model.predict(x_pca)
        return np.hstack((x_standardized, cluster_ids.reshape(-1, 1)))

    def predict(self, x):
        x_scaled = self.robust_scaler.transform(x)
        x_pca = self.pca.transform(x_scaled)
        x_row_stats = self.row_stats.transform(x_scaled)
        x_concat = np.hstack((x_pca, x_row_stats))
        x_standardized = self.standard_scaler.transform(x_concat)

        cluster_ids = self.cluster_model.predict(x_pca)
        x_final = np.hstack((x_standardized, cluster_ids.reshape(-1, 1)))

        if self.model:
            return self.model.predict(x_final)
        return None

    def save(self, path):
        joblib.dump(self, path)
        print(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path):
        pipeline = joblib.load(path)
        print(f"Pipeline loaded from {path}")
        return pipeline