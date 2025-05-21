import joblib
from sklearn.cluster import KMeans

class ClusterWrapper:
    def __init__(self, n_clusters=2):
        self.model = KMeans(n_clusters=n_clusters)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        labels = self.model.predict(X)
        return labels

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        return self