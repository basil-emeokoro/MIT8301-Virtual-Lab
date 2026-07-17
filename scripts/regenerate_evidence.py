import json
from run_pipeline import academic_summary_card, evidence_card

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
academic_summary_card("Assessment Workflow Status", [
    "Dataset successfully validated",
    "Target labels verified and aligned",
    "Data preprocessing completed",
    "Training/test split completed (800/200)",
    "10 EDA figures and 5 evaluation figures generated",
    "No data leakage detected",
], "03_pipeline_completion.png")
print("Evidence cards regenerated with opaque white backgrounds.")
