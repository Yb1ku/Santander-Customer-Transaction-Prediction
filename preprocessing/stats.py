import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

class RowStatsCalculator:
    def __init__(self, stats=("mean", "std", "min", "max",
                              "sum", "skew", "kurtosis")):
        self.stats = stats

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X)

        result = []

        if "mean" in self.stats:
            result.append(df.mean(axis=1).values.reshape(-1, 1))
        if "std" in self.stats:
            result.append(df.std(axis=1).values.reshape(-1, 1))
        if "min" in self.stats:
            result.append(df.min(axis=1).values.reshape(-1, 1))
        if "max" in self.stats:
            result.append(df.max(axis=1).values.reshape(-1, 1))
        if "skew" in self.stats:
            result.append(df.skew(axis=1).values.reshape(-1, 1))

        return np.hstack(result)