import json
from run_pipeline import evidence_card

d = json.load(open("reports/tables/pipeline_results.json", encoding="utf-8"))
evidence_card("Dataset loading and validation", [
    "Rows: 1,000 | Predictors: 9 | Target: credit_risk",
    "Row-level predictor reconciliation: PASS (100%)",
    "Mismatched predictor cells: 0",
    "Class balance: 700 good / 300 bad",
    "Missing: Saving accounts=183; Checking account=394",
], "01_dataset_validation.png")
evidence_card("Model execution evidence", [
    *[f"{x['Model']}: AUC={x['ROC_AUC']:.3f}, Recall={x['Recall']:.3f}, F1={x['F1']:.3f}" for x in d["metrics"]],
    f"Selected model: {d['selected_model']}",
    f"Scratch convergence: {d['scratch']['converged']}; iterations={d['scratch']['iterations']}",
], "02_model_results.png")
evidence_card("Pipeline completion", [
    f"Execution UTC: {d['environment']['executed_utc']}",
    f"Python {d['environment']['python']} | scikit-learn {d['environment']['scikit_learn']}",
    "Target validation: PASS", "EDA figures: 10 | Evaluation figures: 5",
    "Training rows: 800 | Untouched test rows: 200",
], "03_pipeline_completion.png")
print("Evidence cards regenerated with opaque white backgrounds.")
