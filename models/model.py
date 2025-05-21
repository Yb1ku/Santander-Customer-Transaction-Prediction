from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import numpy as np

class XGBoostWrapper:
    def __init__(self, params=None):
        self.params = params if params is not None else {}
        self.model = None

    def fit(self, X_train, y_train):
        if "scale_pos_weight" not in self.params:
            n_pos = (y_train == 1).sum()
            n_neg = (y_train == 0).sum()
            scale_pos_weight = n_neg / n_pos
            self.params["scale_pos_weight"] = scale_pos_weight
            print(f"[XGBoostWrapper] Using scale_pos_weight={scale_pos_weight:.2f}")

        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)


class RandomForestWrapper:
    def __init__(self, params=None):
        self.params = params if params is not None else {}
        self.model = RandomForestClassifier(**self.params)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)


class StackingWrapper:
    def __init__(self, model1, model2, meta_model=None):
        self.model1 = model1
        self.model2 = model2
        self.meta_model = meta_model if meta_model else LogisticRegression()
        self.fitted = False

    def fit(self, X, y):
        self.model1.fit(X, y)
        self.model2.fit(X, y)

        proba1 = self.model1.predict_proba(X)[:, 1]
        proba2 = self.model2.predict_proba(X)[:, 1]
        X_meta = np.vstack([proba1, proba2]).T

        self.meta_model.fit(X_meta, y)
        self.fitted = True

    def predict_proba(self, X):
        proba1 = self.model1.predict_proba(X)[:, 1]
        proba2 = self.model2.predict_proba(X)[:, 1]
        X_meta = np.vstack([proba1, proba2]).T
        return self.meta_model.predict_proba(X_meta)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


class StackingWrapperCV:
    def __init__(self, model1, model2, meta_model=None, n_folds=5, random_state=42):
        self.model1 = model1
        self.model2 = model2
        self.meta_model = meta_model if meta_model else LogisticRegression()
        self.n_folds = n_folds
        self.random_state = random_state
        self.fitted = False

        # Se almacenan los modelos base entrenados en todo el dataset
        self.final_model1 = None
        self.final_model2 = None

    def fit(self, X, y):
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_preds1 = np.zeros(len(X))
        oof_preds2 = np.zeros(len(X))

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            model1_fold = self._clone(self.model1)
            model2_fold = self._clone(self.model2)

            model1_fold.fit(X_train, y_train)
            model2_fold.fit(X_train, y_train)

            oof_preds1[val_idx] = model1_fold.predict_proba(X_val)[:, 1]
            oof_preds2[val_idx] = model2_fold.predict_proba(X_val)[:, 1]

        X_meta = np.vstack([oof_preds1, oof_preds2]).T
        self.meta_model.fit(X_meta, y)
        self.fitted = True

        # Finalmente entrenamos los modelos base sobre todo el dataset para usar en predicción
        self.final_model1 = self._clone(self.model1)
        self.final_model2 = self._clone(self.model2)
        self.final_model1.fit(X, y)
        self.final_model2.fit(X, y)

    def predict_proba(self, X):
        proba1 = self.final_model1.predict_proba(X)[:, 1]
        proba2 = self.final_model2.predict_proba(X)[:, 1]
        X_meta = np.vstack([proba1, proba2]).T
        return self.meta_model.predict_proba(X_meta)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def _clone(self, wrapper):
        # Copia profunda de los wrappers de modelos base
        from copy import deepcopy
        return deepcopy(wrapper)























