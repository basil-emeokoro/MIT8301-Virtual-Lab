from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]


def test_stratified_split_counts_and_percentages():
    data = pd.read_csv(ROOT / "data/processed/german_credit_with_verified_target.csv")
    X = data.drop(columns="credit_risk")
    y = data["credit_risk"]
    _, _, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    assert y_train.value_counts().sort_index().to_dict() == {0: 560, 1: 240}
    assert y_test.value_counts().sort_index().to_dict() == {0: 140, 1: 60}
    assert (y_train.value_counts(normalize=True).sort_index() * 100).to_dict() == {0: 70.0, 1: 30.0}
    assert (y_test.value_counts(normalize=True).sort_index() * 100).to_dict() == {0: 70.0, 1: 30.0}


def test_source_keeps_tuning_and_preprocessing_off_test_set():
    source = (ROOT / "scripts/run_pipeline.py").read_text(encoding="utf-8")
    assert "test_size=.20, random_state=RANDOM_STATE, stratify=y" in source
    assert "search.fit(X_train, y_train)" in source
    assert "search.fit(X, y)" not in source
    assert "build_preprocessor(X_train)" in source
    assert "fit_transform(X_train, y_train)" in source
    assert "transform(X_test)" in source
    assert "StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)" in source
