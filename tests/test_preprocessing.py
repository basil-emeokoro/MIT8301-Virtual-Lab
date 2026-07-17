from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from mit8301_credit_risk.preprocessing import build_preprocessor


ROOT = Path(__file__).resolve().parents[1]


def test_preprocessor_fits_training_data_and_handles_unknowns():
    df = pd.read_csv(ROOT / "data/processed/german_credit_with_verified_target.csv")
    X_train, X_test = train_test_split(df.drop(columns="credit_risk"), test_size=.2, random_state=42)
    processor = build_preprocessor(X_train).fit(X_train)
    transformed_train = processor.transform(X_train)
    changed = X_test.copy(); changed.loc[changed.index[0], "Purpose"] = "unseen-purpose"
    transformed_test = processor.transform(changed)
    assert transformed_train.shape[1] == transformed_test.shape[1]
    assert hasattr(processor.named_transformers_["num"].named_steps["clipper"], "lower_")
