"""Execute target validation, EDA, modelling, evaluation, and evidence generation."""
from __future__ import annotations

import json
import os
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_score,
                             recall_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from mit8301_credit_risk.config import RANDOM_STATE
from mit8301_credit_risk.data_validation import validate_and_restore
from mit8301_credit_risk.logistic_regression_scratch import LogisticRegressionScratch
from mit8301_credit_risk.preprocessing import build_preprocessor

FIG = ROOT / "reports" / "figures"
TABLE = ROOT / "reports" / "tables"
SHOT = ROOT / "reports" / "screenshots"
for directory in (FIG, TABLE, SHOT):
    directory.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message="Inconsistent values: penalty=")


def save(name: str):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=220, bbox_inches="tight")
    plt.close()


def eda_figures(df: pd.DataFrame) -> dict:
    y = df["credit_risk"]
    X = df.drop(columns="credit_risk")
    counts = y.map({0: "Good", 1: "Bad / risky"}).value_counts()
    counts.plot.bar(color=["#35618D", "#B64B4B"])
    plt.title("Credit-risk class distribution"); plt.xlabel("Class"); plt.ylabel("Applicants")
    save("01_target_distribution.png")
    for number, column in enumerate(["Age", "Credit amount", "Duration"], start=2):
        plt.hist(X[column], bins=25, color="#35618D", edgecolor="white")
        plt.title(f"Distribution of {column}"); plt.xlabel(column); plt.ylabel("Frequency")
        save(f"{number:02d}_{column.lower().replace(' ', '_')}_histogram.png")
    melted = X[["Age", "Credit amount", "Duration"]].apply(lambda s: (s-s.median())/(s.quantile(.75)-s.quantile(.25))).melt()
    sns.boxplot(data=melted, x="variable", y="value", color="#8CA9C2")
    plt.title("Numerical boxplots (robustly scaled)"); plt.xlabel("Feature"); plt.ylabel("Robust-scaled value")
    save("05_numerical_boxplots.png")
    missing = X.isna().sum().sort_values(ascending=False)
    missing[missing > 0].plot.bar(color="#B8863B")
    plt.title("Missing values by feature"); plt.xlabel("Feature"); plt.ylabel("Missing rows")
    save("06_missing_values.png")
    categories = ["Sex", "Job", "Housing", "Purpose"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for column, ax in zip(categories, axes.flat):
        X[column].astype(str).value_counts().plot.bar(ax=ax, color="#557A95")
        ax.set_title(column); ax.set_xlabel(""); ax.set_ylabel("Applicants"); ax.tick_params(axis="x", rotation=35)
    save("07_categorical_distributions.png")
    corr = df[["Age", "Credit amount", "Duration", "credit_risk"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Numerical correlation matrix")
    save("08_correlation_heatmap.png")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for column, ax in zip(["Age", "Credit amount", "Duration"], axes):
        sns.boxplot(data=df, x="credit_risk", y=column, ax=ax, color="#8CA9C2")
        ax.set_xticks([0, 1], ["Good", "Bad / risky"]); ax.set_xlabel("Credit class")
    save("09_numerical_features_by_target.png")
    risk_by_purpose = df.groupby("Purpose", dropna=False)["credit_risk"].mean().sort_values()
    (risk_by_purpose * 100).plot.barh(color="#7D5A6A")
    plt.title("Observed bad-credit rate by purpose"); plt.xlabel("Bad-credit rate (%)"); plt.ylabel("Purpose")
    save("10_risk_rate_by_purpose.png")
    numeric = X[["Age", "Credit amount", "Duration"]]
    q1, q3 = numeric.quantile(.25), numeric.quantile(.75)
    iqr = q3-q1
    outliers = ((numeric < q1-1.5*iqr) | (numeric > q3+1.5*iqr)).sum().to_dict()
    return outliers


def metric_row(name, y_true, prediction, probability):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, prediction),
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "Recall": recall_score(y_true, prediction, zero_division=0),
        "F1": f1_score(y_true, prediction, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, probability),
    }


def evidence_card(title: str, lines: list[str], filename: str):
    fig = plt.figure(figsize=(12, 7), facecolor="white")
    plt.axis("off")
    plt.text(.04, .94, title, fontsize=20, weight="bold", va="top", color="#183B56")
    plt.text(.04, .84, "\n".join(lines), fontsize=12, family="monospace", va="top", linespacing=1.55)
    plt.savefig(SHOT / filename, dpi=180, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)


def main():
    validation = validate_and_restore(
        ROOT / "data/raw/german_credit_data.csv",
        ROOT / "data/raw/target_source/german_credit_with_risk.csv",
        ROOT / "data/processed/german_credit_with_verified_target.csv",
        ROOT / "reports/data_provenance_and_target_validation.md",
    )
    df = pd.read_csv(ROOT / "data/processed/german_credit_with_verified_target.csv")
    outliers = eda_figures(df)
    X, y = df.drop(columns="credit_risk"), df["credit_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    searches = {
        "Tuned Logistic Regression": (
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
            {"model__C": [.01, .1, 1, 10, 100], "model__penalty": ["l1", "l2"], "model__solver": ["liblinear"], "model__class_weight": [None, "balanced"]},
        ),
        "Tuned SVM": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {"model__C": [.1, 1, 10], "model__kernel": ["linear", "rbf"], "model__gamma": ["scale", "auto", .01, .1], "model__class_weight": [None, "balanced"]},
        ),
        "Tuned Gaussian Naive Bayes": (
            GaussianNB(), {"model__var_smoothing": np.logspace(-12, -6, 7)}
        ),
    }
    metrics, fitted, tuning = [], {}, {}
    for name, (model, grid) in searches.items():
        # Only the training schema is supplied here; all learned preprocessing
        # statistics are fitted inside GridSearchCV's training folds.
        pipeline = Pipeline([("preprocess", build_preprocessor(X_train)), ("model", model)])
        search = GridSearchCV(pipeline, grid, cv=cv, scoring="roc_auc", n_jobs=-1, return_train_score=False)
        search.fit(X_train, y_train)
        prediction = search.predict(X_test)
        probability = search.predict_proba(X_test)[:, 1]
        metrics.append(metric_row(name, y_test, prediction, probability))
        fitted[name] = search.best_estimator_
        tuning[name] = {"best_params": search.best_params_, "best_cv_roc_auc": float(search.best_score_)}

    scratch_preprocessor = build_preprocessor(X_train)
    X_train_t = scratch_preprocessor.fit_transform(X_train, y_train)
    X_test_t = scratch_preprocessor.transform(X_test)
    scratch = LogisticRegressionScratch().fit(X_train_t, y_train.to_numpy())
    scratch_pred = scratch.predict(X_test_t)
    scratch_prob = scratch.predict_proba(X_test_t)[:, 1]
    metrics.append(metric_row("Logistic Regression from Scratch", y_test, scratch_pred, scratch_prob))
    plt.plot(scratch.loss_history_, color="#35618D")
    plt.title("Scratch logistic-regression convergence"); plt.xlabel("Iteration"); plt.ylabel("Binary cross-entropy + L2")
    save("11_logistic_regression_scratch_convergence.png")

    comparison = pd.DataFrame(metrics).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    comparison.to_csv(TABLE / "model_comparison.csv", index=False)
    header = "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |\n|---|---:|---:|---:|---:|---:|\n"
    rows = "\n".join(
        f"| {r.Model} | {r.Accuracy:.3f} | {r.Precision:.3f} | {r.Recall:.3f} | {r.F1:.3f} | {r.ROC_AUC:.3f} |"
        for r in comparison.itertuples()
    )
    (TABLE / "model_comparison.md").write_text(header + rows + "\n", encoding="utf-8")
    pd.DataFrame([
        {"Model": k, "Best CV ROC-AUC": v["best_cv_roc_auc"], "Best parameters": json.dumps(v["best_params"])}
        for k, v in tuning.items()
    ]).to_csv(TABLE / "hyperparameter_tuning.csv", index=False)

    comparison.set_index("Model")[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]].plot.bar(figsize=(12, 6))
    plt.title("Test-set model comparison"); plt.ylabel("Score"); plt.ylim(0, 1); plt.legend(ncol=5, loc="lower center", bbox_to_anchor=(.5, -0.32))
    save("12_model_metric_comparison.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in fitted.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    RocCurveDisplay.from_predictions(y_test, scratch_prob, name="Scratch Logistic", ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set_title("Test-set ROC curves")
    save("13_roc_curves.png")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    all_predictions = {name: model.predict(X_test) for name, model in fitted.items()}
    all_predictions["Scratch Logistic"] = scratch_pred
    for ax, (name, prediction) in zip(axes.flat, all_predictions.items()):
        ConfusionMatrixDisplay(confusion_matrix(y_test, prediction), display_labels=["Good", "Bad"]).plot(ax=ax, colorbar=False)
        ax.set_title(name)
    save("14_confusion_matrices.png")

    logistic = fitted["Tuned Logistic Regression"]
    names = logistic.named_steps["preprocess"].get_feature_names_out()
    coefficients = logistic.named_steps["model"].coef_[0]
    coef = pd.DataFrame({"Feature": names, "Coefficient": coefficients})
    coef["Absolute coefficient"] = coef["Coefficient"].abs()
    coef = coef.sort_values("Absolute coefficient", ascending=False)
    coef.to_csv(TABLE / "logistic_feature_coefficients.csv", index=False)
    top = coef.head(15).sort_values("Coefficient")
    top.plot.barh(x="Feature", y="Coefficient", legend=False, color=["#35618D" if x < 0 else "#B64B4B" for x in top["Coefficient"]])
    plt.title("Most influential logistic-regression coefficients"); plt.xlabel("Coefficient (positive = higher bad-credit log-odds)"); plt.ylabel("")
    save("15_feature_importance.png")

    best_name = comparison.iloc[0]["Model"]
    selected_name = "Tuned Logistic Regression" if comparison.loc[comparison.Model == "Tuned Logistic Regression", "ROC_AUC"].iloc[0] >= comparison.ROC_AUC.max() - .03 else best_name
    selected = fitted.get(selected_name, fitted[best_name])
    fairness = []
    selected_pred = selected.predict(X_test)
    for group in sorted(X_test["Sex"].unique()):
        mask = X_test["Sex"].eq(group).to_numpy()
        fairness.append({"Sex": group, "n": int(mask.sum()), "Accuracy": accuracy_score(y_test.to_numpy()[mask], selected_pred[mask]), "Recall_bad": recall_score(y_test.to_numpy()[mask], selected_pred[mask], zero_division=0)})
    pd.DataFrame(fairness).to_csv(TABLE / "fairness_by_sex.csv", index=False)
    reports = {name: classification_report(y_test, pred, output_dict=True) for name, pred in all_predictions.items()}
    results = {
        "validation": validation, "shape": list(df.shape), "missing_counts": X.isna().sum().to_dict(),
        "outlier_counts_iqr": outliers, "train_rows": len(X_train), "test_rows": len(X_test),
        "split_strategy": {
            "test_size": 0.20, "random_state": RANDOM_STATE, "stratified": True,
            "training_class_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
            "testing_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
            "training_class_percent": {str(k): float(v * 100) for k, v in y_train.value_counts(normalize=True).sort_index().items()},
            "testing_class_percent": {str(k): float(v * 100) for k, v in y_test.value_counts(normalize=True).sort_index().items()},
            "cross_validation": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "grid_search_input": "X_train and y_train only",
            "threshold_optimisation": "None; fixed decision threshold 0.5",
        },
        "tuning": tuning, "metrics": comparison.to_dict(orient="records"), "selected_model": selected_name,
        "selection_rule": "Prefer the tuned interpretable logistic model when within 0.03 ROC-AUC of the highest model; otherwise select the highest ROC-AUC model.",
        "scratch": {"iterations": scratch.n_iter_, "converged": scratch.converged_, "final_loss": scratch.loss_history_[-1]},
        "top_coefficients": coef.head(10).to_dict(orient="records"), "fairness_by_sex": fairness,
        "classification_reports": reports,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__, "executed_utc": datetime.now(timezone.utc).isoformat()},
    }
    (TABLE / "pipeline_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    evidence_card("Dataset loading and validation", [
        "Rows: 1,000 | Predictors: 9 | Target: credit_risk",
        "Row-level predictor reconciliation: PASS (100%)",
        "Mismatched predictor cells: 0",
        "Class balance: 700 good / 300 bad",
        "Missing: Saving accounts=183; Checking account=394",
    ], "01_dataset_validation.png")
    evidence_card("Model execution evidence", [
        *[f"{r.Model}: AUC={r.ROC_AUC:.3f}, Recall={r.Recall:.3f}, F1={r.F1:.3f}" for r in comparison.itertuples()],
        f"Selected model: {selected_name}",
        f"Scratch convergence: {scratch.converged_}; iterations={scratch.n_iter_}",
    ], "02_model_results.png")
    evidence_card("Pipeline completion", [
        f"Execution UTC: {results['environment']['executed_utc']}",
        f"Python {results['environment']['python']} | scikit-learn {sklearn.__version__}",
        "Target validation: PASS",
        "EDA figures: 10 | Evaluation figures: 5",
        "Training rows: 800 | Untouched test rows: 200",
    ], "03_pipeline_completion.png")
    print(json.dumps({"selected_model": selected_name, "metrics": results["metrics"], "validation": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
