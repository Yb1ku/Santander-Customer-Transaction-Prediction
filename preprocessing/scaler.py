from sklearn.preprocessing import RobustScaler, StandardScaler

class RobustScalerWrapper:
    def __init__(self):
        self.scaler = RobustScaler()

    def fit(self, X):
        self.scaler.fit(X)

    def transform(self, X):
        return self.scaler.transform(X)

    def fit_transform(self, X):
        return self.scaler.fit_transform(X)


class StandardScalerWrapper:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)

    def transform(self, X):
        return self.scaler.transform(X)

    def fit_transform(self, X):
        return self.scaler.fit_transform(X)