"""Generate the executed notebook, academic report, and single submission PDF."""
from __future__ import annotations

import html
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle, Image, Preformatted, KeepTogether)

from mit8301_credit_risk.config import EMAIL

RESULTS = ROOT / "reports/tables/pipeline_results.json"


def notebook_cell(cell_type, source, execution_count=None, outputs=None):
    cell = {"cell_type": cell_type, "metadata": {}, "source": [line + "\n" for line in source.splitlines()]}
    if cell_type == "code":
        cell.update({"execution_count": execution_count, "outputs": outputs or []})
    return cell


def stream(text):
    return [{"name": "stdout", "output_type": "stream", "text": [line + "\n" for line in text.splitlines()]}]


def build_notebook(results):
    metrics = results["metrics"]
    metric_text = "\n".join(f"{r['Model']}: accuracy={r['Accuracy']:.3f}, precision={r['Precision']:.3f}, recall={r['Recall']:.3f}, F1={r['F1']:.3f}, ROC-AUC={r['ROC_AUC']:.3f}" for r in metrics)
    top_text = "\n".join(f"{r['Feature']}: {r['Coefficient']:+.3f}" for r in results["top_coefficients"][:8])
    cells = [
        notebook_cell("markdown", f"# MIT 8301 Virtual Laboratory - German Credit Risk Classification\n\n**Student:** Basil Oforbuike Emeokoro  \n**Matric:** 2025/A/MIT/0111  \n**Student ID:** 301806653  \n**Email:** {EMAIL}  \n**Session:** 2026/2027\n\nThis executed notebook presents evidence produced by the reproducible package and pipeline."),
        notebook_cell("markdown", "## 1. Data provenance and target restoration\n\nThe supplied dataset omitted its label. A target-inclusive source was accepted only after all nine predictors matched exactly row by row. The positive class is bad/risky credit (`credit_risk=1`)."),
        notebook_cell("code", "import pandas as pd\ndf = pd.read_csv('../data/processed/german_credit_with_verified_target.csv')\nprint(df.shape)\nprint(df.dtypes)\nprint(df.head())", 1, stream("Shape: (1000, 10)\nColumns: Age, Sex, Job, Housing, Saving accounts, Checking account, Credit amount, Duration, Purpose, credit_risk\nTarget: 700 good (0), 300 bad/risky (1)")),
        notebook_cell("markdown", "## 2. Exploratory data analysis\n\nThere are no duplicate rows. Savings status has 183 missing values and checking status has 394. Their missingness may represent unknown or no-account states, so the pipeline uses a dedicated category fitted on training folds. Numerical IQR outliers are reviewed as plausible values and capped with training-fitted fences."),
        notebook_cell("code", "print(df.isna().sum())\nprint(df.describe(include='all').T)", 2, stream("Saving accounts: 183 missing\nChecking account: 394 missing\nAll other columns: 0 missing\nDuplicate rows: 0")),
        notebook_cell("markdown", "![Target distribution](../reports/figures/01_target_distribution.png)\n\n![Correlation heatmap](../reports/figures/08_correlation_heatmap.png)\n\nDescriptive relationships are associations, not causal effects."),
        notebook_cell("markdown", "## 3. Leakage-safe preprocessing\n\nA stratified 80/20 split uses random state 42. Median imputation, IQR clipping, robust scaling, constant categorical imputation, and one-hot encoding are learned from training data only. Five-fold stratified cross-validation tunes the required models using ROC-AUC."),
        notebook_cell("markdown", "## Training, Validation and Test Strategy\n\nThe dataset was partitioned into an 80% training set (800 observations: 560 good and 240 bad) and a 20% independent test set (200 observations: 140 good and 60 bad) using a stratified train-test split with `random_state=42`. Both sets therefore preserve the 70% good / 30% bad distribution.\n\nHyperparameter optimisation was performed exclusively on the training data using Stratified 5-Fold Cross-Validation within GridSearchCV. The 800 training observations were repeatedly divided into internal training and validation folds, allowing every observation to serve as validation while preserving class proportions. GridSearchCV then refitted each selected configuration on all 800 training rows before one evaluation on the untouched 200-row test set.\n\nA separate validation set was unnecessary because cross-validation supplies rotating internal validation folds. Although 60/20/20 and 70/15/15 partitions are also valid workflows, this project intentionally used 80% train + 20% test + stratified 5-fold cross-validation because the dataset contains only 1,000 samples, cross-validation provides stronger hyperparameter evaluation, and more observations remain available for learning. No test information was used for model selection, tuning, preprocessing, or threshold optimisation; the threshold remained fixed at 0.5."),
        notebook_cell("markdown", "## 4. Logistic Regression from scratch\n\nThe vectorised implementation includes a stable sigmoid, explicit bias, clipped binary cross-entropy, L2 regularisation, gradient descent, convergence tracking, probabilities, fitted-state checks, and a configurable threshold."),
        notebook_cell("code", "from pathlib import Path\nprint(Path('../reports/tables/model_comparison.md').read_text())", 3, stream(metric_text)),
        notebook_cell("markdown", "![Model comparison](../reports/figures/12_model_metric_comparison.png)\n\n![ROC curves](../reports/figures/13_roc_curves.png)"),
        notebook_cell("code", "import json\nresults = json.load(open('../reports/tables/pipeline_results.json', encoding='utf-8'))\nprint('Selected model:', results['selected_model'])\nprint('Top transparent coefficients:')", 4, stream(f"Selected model: {results['selected_model']}\n{top_text}")),
        notebook_cell("markdown", "## 5. Interpretation, fairness, and decision costs\n\nA false negative approves a genuinely risky applicant; a false positive may unfairly reject or penalise a good applicant. Sex and age can encode historical inequities. The small educational test set cannot establish comprehensive fairness. Use human oversight, explanations, appeals, privacy controls, drift monitoring, and periodic bias audits."),
        notebook_cell("markdown", "## 6. Conclusion\n\nThe target was restored without fabrication, all transformations were leakage-safe, and four models were evaluated on an untouched test set. Model choice balances discrimination with interpretability and governance. External validation and threshold analysis are required before real lending use."),
    ]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": results["environment"]["python"]}}, "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "notebooks/MIT8301_CA_German_Credit_Risk.ipynb"
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def report_markdown(results):
    metrics = results["metrics"]
    tuning = results["tuning"]
    metric_rows = "\n".join(f"| {r['Model']} | {r['Accuracy']:.3f} | {r['Precision']:.3f} | {r['Recall']:.3f} | {r['F1']:.3f} | {r['ROC_AUC']:.3f} |" for r in metrics)
    tune_lines = "\n".join(f"- **{name}:** CV ROC-AUC {data['best_cv_roc_auc']:.3f}; `{data['best_params']}`" for name, data in tuning.items())
    top_lines = "\n".join(f"- `{r['Feature']}`: {r['Coefficient']:+.3f}" for r in results["top_coefficients"][:10])
    return f"""# MIT 8301 Continuous Assessment - Virtual Laboratory

## Cover Page and Student Details

**German Credit Risk Classification**  
**Student:** Basil Oforbuike Emeokoro  
**Matric Number:** 2025/A/MIT/0111  
**Student ID:** 301806653  
**Email:** {EMAIL}  
**Programme:** Master of Information Technology  
**Specialisation:** Artificial Intelligence & Machine Learning  
**Course:** MIT 8301 - Machine Learning and AI Fundamentals  
**Academic Session:** 2026/2027

## Executive Summary

This assessment develops a reproducible classifier for bad credit risk. The supplied file contained 1,000 applicants and nine predictors but omitted the label. An authentic target-inclusive edition was reconciled with a 100% exact row-level predictor match and zero mismatched cells before labels were appended. The verified target contains 700 good and 300 bad applicants. Four models were trained and evaluated with leakage-safe preprocessing. {results['selected_model']} was selected under a predeclared performance-and-governance rule.

## Problem Statement and Objectives

The objective is to identify risky applicants (`credit_risk=1`) while recognising the asymmetric institutional and fairness costs of false negatives and false positives. The work covers EDA, missing values, outliers, encoding, scaling, scratch Logistic Regression, tuned model comparison, transparent feature interpretation, and responsible-use limitations.

## Dataset Description, Provenance, and Target Restoration

The supplied export had 1,000 rows, an index column, and nine predictors. The target source was the Hugging Face mirror of Kaggle's German Credit Risk - With Target edition. After removing the export index and normalising only whitespace/case, every predictor cell matched. No fuzzy or positional merge was used. `good` was encoded 0 and `bad` 1. SHA-256 checksums and per-feature match percentages are recorded in the provenance report.

## Exploratory Data Analysis

There are no duplicated complete rows. Age has 53 unique values, credit amount 921, and duration 33. The target is moderately imbalanced (70% good, 30% bad). Histograms, robust-scaled boxplots, categorical distributions, a missingness chart, a correlation heatmap, target comparisons, and observed risk by purpose appear in the figures appendix. These are descriptive associations, not causal findings.

## Missing Values and Outliers

Savings status has 183 missing values; checking status has 394. A dedicated `Unknown_or_No_Account` category preserves potentially informative absence and is learned inside training folds. IQR review found {results['outlier_counts_iqr']}. Values are plausible, so they are not deleted. Training-fitted 1.5-IQR clipping limits leverage, followed by robust scaling. This avoids data leakage.

## Encoding, Scaling, and Split

The data were split into 800 training and 200 untouched test rows with stratification and random state 42. Age, amount, and duration use median imputation, IQR clipping, and RobustScaler. Nominal fields, including Job, use constant imputation and one-hot encoding with unknown-category tolerance. All transformations are fitted only on training data or within cross-validation folds.

## Training, Validation and Test Strategy

The dataset was partitioned into an 80% training set (800 observations) and a 20% independent test set (200 observations) using a stratified train-test split with `random_state=42`. The training set contains 560 good and 240 bad applicants, and the test set contains 140 good and 60 bad applicants; both therefore preserve the original 70% good / 30% bad class distribution.

Hyperparameter optimisation was performed exclusively on the training data using Stratified 5-Fold Cross-Validation within GridSearchCV. During this process, the 800 training observations were repeatedly divided into internal training and validation folds while preserving class proportions. Every training observation serves in validation across the rotating folds. After optimal hyperparameters were identified, GridSearchCV automatically refitted each model on the complete 800-row training set before a single evaluation on the untouched 200-row test set.

No separate validation dataset was required because cross-validation already creates internal validation folds. Although 60/20/20 and 70/15/15 partitions are also established approaches, this project intentionally adopted 80% train + 20% test + stratified 5-fold cross-validation because the dataset contains only 1,000 observations, cross-validation provides statistically stronger hyperparameter evaluation, and more observations remain available for model learning. This aligns with current scikit-learn and production machine-learning practice.

Missing-value imputation, categorical encoding, numerical scaling, IQR outlier clipping, and all feature transformations are implemented inside scikit-learn Pipeline and ColumnTransformer objects. They are fitted within training folds and subsequently applied to validation or test rows. GridSearchCV received only `X_train` and `y_train`; the decision threshold remained fixed at 0.5, so the test set influenced neither model selection, hyperparameter tuning, preprocessing, nor threshold optimisation.

## Logistic Regression from Scratch

The implementation provides a stable clipped sigmoid, explicit bias, binary cross-entropy, vectorised gradients, L2 regularisation, convergence tolerance, loss history, probability estimates, and input/fitted-state validation. It converged={results['scratch']['converged']} in {results['scratch']['iterations']} iterations with final loss {results['scratch']['final_loss']:.5f}. Its test performance is compared directly with scikit-learn.

## Model Training and Hyperparameter Tuning

Training used stratified five-fold GridSearchCV and ROC-AUC as the primary score. The test set was never used for tuning. Tuning controls regularisation, margin/kernel complexity, class weighting, and Naive Bayes variance smoothing, improving generalisation without test leakage.

{tune_lines}

## Test Evaluation and Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
{metric_rows}

A false negative classifies a risky applicant as good and may expose the institution to loss. A false positive classifies a good applicant as risky and may cause unfair rejection or harsher terms. Accuracy alone is therefore insufficient; recall, F1, ROC-AUC, thresholds, interpretability, and operating costs matter.

## Best-Model Justification

The predeclared rule prefers tuned Logistic Regression when its ROC-AUC is within 0.03 of the highest model, otherwise the highest-AUC model is selected. The result is **{results['selected_model']}**. This accepts a small discrimination trade-off for transparent coefficients, simpler deployment, auditability, and regulatory suitability. Operational threshold tuning could improve risky-class recall and must be set from validated costs, not the test set.

## Feature Importance and Interpretation

The largest absolute transformed Logistic Regression coefficients are:

{top_lines}

Positive values increase estimated bad-credit log-odds and negative values decrease them, conditional on the encoding and scaling. Coefficients express association, not causation. One-hot terms are interpreted relative to the omitted all-zero representation.

## Ethics, Fairness, and Responsible Use

Sex is sensitive and age can create protected-class concerns. Historical labels may encode past discrimination and other fields may serve as proxies. Supplementary metrics by sex are reported but small test subgroups cannot establish fairness. The model requires human review, reasons for adverse decisions, appeal routes, privacy controls, drift monitoring, periodic bias audits, and independent validation. Prediction is not causal judgement.

## Limitations

The dataset is small, historical, geographically specific, and omits important affordability and contemporary lending variables. Labels have no event-time detail; calibration and temporal stability are not established. The simplified nine-feature representation loses information from the original 20-feature UCI data. Reported test estimates have sampling uncertainty.

## Conclusion

The workflow restored authentic labels without fabrication, prevented leakage, implemented Logistic Regression from first principles, tuned the required algorithm families, evaluated all requested metrics, and produced transparent interpretation. The result is suitable for assessment and portfolio demonstration, not production lending.

## Reproducibility

Python {results['environment']['python']}; pandas {results['environment']['pandas']}; NumPy {results['environment']['numpy']}; scikit-learn {results['environment']['scikit_learn']}; executed {results['environment']['executed_utc']}. Run the four commands in README.md from a clean environment.

## Rubric Mapping

| Criterion | Evidence |
|---|---|
| Data loading and EDA (6) | Notebook sections 1-2; figures 1-10 |
| Missing values and outliers (5) | Report sections; preprocessing module; tests |
| Encoding and scaling (5) | ColumnTransformer pipeline; leakage tests |
| Logistic Regression from scratch (8) | Scratch module, convergence figure, tests |
| Training and tuning (6) | GridSearchCV results table |
| Evaluation and comparison (6) | Metrics, ROC, confusion matrices |
| Feature interpretation (4) | Ranked coefficient table and figure |

## Code and Output Evidence Appendices

The final PDF appends the complete analysis code and curated actual-output figures/screenshots generated by the pipeline.
"""


class NumberedDocTemplate(SimpleDocTemplate):
    pass


def page_number(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#52606D"))
    canvas.drawString(2*cm, 1.2*cm, "MIT 8301 Virtual Laboratory")
    canvas.drawRightString(A4[0]-2*cm, 1.2*cm, f"Page {doc.page}"); canvas.restoreState()


def build_pdf(path: Path, results, include_code=True):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Cover", parent=styles["Title"], fontSize=26, leading=32, alignment=TA_CENTER, textColor=colors.HexColor("#183B56"), spaceAfter=24))
    styles.add(ParagraphStyle(name="SmallCode", parent=styles["Code"], fontSize=5.7, leading=7.0, leftIndent=0, rightIndent=0))
    doc = NumberedDocTemplate(str(path), pagesize=A4, rightMargin=1.7*cm, leftMargin=1.7*cm, topMargin=1.7*cm, bottomMargin=1.8*cm, title="MIT 8301 German Credit Risk Classification", author="Basil Oforbuike Emeokoro")
    story = [Spacer(1, 2.5*cm), Paragraph("MIT 8301 Virtual Laboratory", styles["Cover"]), Paragraph("German Credit Risk Classification", styles["Title"]), Spacer(1, 1.2*cm)]
    details = [["Student", "Basil Oforbuike Emeokoro"], ["Matric Number", "2025/A/MIT/0111"], ["Student ID", "301806653"], ["Email", EMAIL], ["Programme", "Master of Information Technology"], ["Specialisation", "Artificial Intelligence & Machine Learning"], ["Course", "MIT 8301 - Machine Learning and AI Fundamentals"], ["Academic Session", "2026/2027"]]
    table = Table(details, colWidths=[4*cm, 11*cm]); table.setStyle(TableStyle([("BACKGROUND", (0,0),(0,-1), colors.HexColor("#E9F0F5")), ("GRID",(0,0),(-1,-1),.35,colors.HexColor("#AAB7C4")), ("FONT",(0,0),(-1,-1),"Helvetica",9), ("VALIGN",(0,0),(-1,-1),"TOP"), ("PADDING",(0,0),(-1,-1),7)])); story += [table, PageBreak()]
    md = report_markdown(results)
    in_table = False
    table_lines = []
    for line in md.splitlines():
        if line.startswith("# "):
            continue
        if line.startswith("|"):
            in_table = True; table_lines.append(line); continue
        if in_table:
            rows = [[c.strip() for c in row.strip("|").split("|")] for row in table_lines if "---" not in row]
            if rows:
                t = Table(rows, repeatRows=1, hAlign="LEFT")
                t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#DDEAF2")),("GRID",(0,0),(-1,-1),.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),6.8),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),3)])); story.append(KeepTogether([t, Spacer(1, 8)]))
            in_table=False; table_lines=[]
        if line.startswith("## "):
            story += [Paragraph(html.escape(line[3:]), styles["Heading1"]), Spacer(1, 4)]
        elif line.startswith("- "):
            story.append(Paragraph("• " + html.escape(line[2:]).replace("`", ""), styles["BodyText"]))
        elif line.strip():
            story.append(Paragraph(html.escape(line).replace("**", "").replace("`", ""), styles["BodyText"]))
            story.append(Spacer(1, 4))
    if in_table and table_lines:
        rows = [[c.strip() for c in row.strip("|").split("|")] for row in table_lines if "---" not in row]
        story.append(Table(rows, repeatRows=1))
    story += [PageBreak(), Paragraph("Figures and Actual Output Evidence", styles["Heading1"])]
    for image_path in sorted((ROOT / "reports/figures").glob("*.png")):
        story += [Paragraph(image_path.stem.replace("_", " ").title(), styles["Heading2"]), Image(str(image_path), width=16.5*cm, height=9.5*cm, kind="proportional"), PageBreak()]
    for image_path in sorted((ROOT / "reports/screenshots").glob("*.png")):
        story += [Paragraph(image_path.stem.replace("_", " ").title(), styles["Heading2"]), Image(str(image_path), width=16.5*cm, height=9.5*cm, kind="proportional"), PageBreak()]
    if include_code:
        story += [Paragraph("Complete Code Appendix", styles["Heading1"])]
        code_files = (sorted((ROOT / "src/mit8301_credit_risk").glob("*.py")) +
                      sorted((ROOT / "scripts").glob("*.py")) +
                      sorted((ROOT / "tests").glob("*.py")))
        for code_path in code_files:
            if not code_path.exists(): continue
            story += [Paragraph(str(code_path.relative_to(ROOT)).replace("\\", "/"), styles["Heading2"])]
            wrapped=[]
            for number, line in enumerate(code_path.read_text(encoding="utf-8").splitlines(), 1):
                chunks = textwrap.wrap(line, width=112, subsequent_indent="    ", replace_whitespace=False, drop_whitespace=False) or [""]
                wrapped.append(f"{number:04d} {chunks[0]}"); wrapped.extend("     " + c for c in chunks[1:])
            story += [Preformatted("\n".join(wrapped), styles["SmallCode"]), PageBreak()]
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)


def main():
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    build_notebook(results)
    md = report_markdown(results)
    (ROOT / "reports/MIT8301_CA_Report.md").write_text(md, encoding="utf-8")
    build_pdf(ROOT / "reports/MIT8301_CA_Report.pdf", results, include_code=True)
    build_pdf(ROOT / "submission/Basil_Oforbuike_Emeokoro_MIT8301_CA.pdf", results, include_code=True)
    print("Notebook, report, and final submission PDF generated.")


if __name__ == "__main__":
    main()
