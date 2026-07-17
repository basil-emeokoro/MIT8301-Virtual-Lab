from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


class IQRClipper(BaseEstimator, TransformerMixin):
    """Training-fitted winsorisation using 1.5-IQR fences."""

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(values, [25, 75], axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - 1.5 * iqr
        self.upper_ = q3 + 1.5 * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = ["Age", "Credit amount", "Duration"]
    categorical = [c for c in frame.columns if c not in numeric]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", IQRClipper()),
        ("scale", RobustScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown_or_No_Account")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ], verbose_feature_names_out=True)
