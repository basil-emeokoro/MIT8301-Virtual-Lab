# MIT 8301 Virtual Laboratory - German Credit Risk Classification

End-to-end, reproducible classification assessment for **Basil Oforbuike Emeokoro** (2025/A/MIT/0111; Student ID 301806653; b.emeokoro6653@miva.edu.ng), Master of Information Technology, Artificial Intelligence & Machine Learning, Miva Open University.

## Assessment context

The project restores the omitted authentic credit-risk target only after exact row-level reconciliation, explores the data, prevents preprocessing leakage, implements logistic regression from scratch, tunes Logistic Regression, SVM, and Gaussian Naive Bayes with stratified five-fold cross-validation, evaluates an untouched test set, and interprets transparent model coefficients.

## Data provenance

The supplied Kaggle-derived CSV contains 1,000 rows and nine predictors but no label. `scripts/run_pipeline.py` compares it cell by cell with the target-inclusive **German Credit Risk - With Target** edition. Validation passed at 100% for every predictor and appended `credit_risk` only afterward (`1 = bad/risky`, `0 = good`). See `reports/data_provenance_and_target_validation.md`.

## Reproduce

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\run_pipeline.py
python scripts\build_submission.py
pytest -q
python scripts\validate_submission.py
```

Randomness is fixed at 42. All imputers, outlier fences, encoders, and scalers are fitted on training folds only.

## Models

- Vectorised Logistic Regression from scratch
- Tuned scikit-learn Logistic Regression
- Tuned Support Vector Machine
- Tuned Gaussian Naive Bayes

Headline results and the selected model are generated from actual execution in `reports/tables/model_comparison.md` and copied into the report. Selection considers ROC-AUC, risky-class recall, F1, interpretability, generalisation, and deployment complexity rather than accuracy alone.

## Deliverables

- Executed notebook: `notebooks/MIT8301_CA_German_Credit_Risk.ipynb`
- Academic report: `reports/MIT8301_CA_Report.md` and `.pdf`
- Final one-file submission: `submission/Basil_Oforbuike_Emeokoro_MIT8301_CA.pdf`
- Figures: `reports/figures/`
- Rendered evidence: `reports/screenshots/`

## Responsible use

This small historical educational dataset cannot justify autonomous lending decisions. Sex and age may create fairness concerns; predictions require human review, explanations, appeal mechanisms, privacy protection, drift monitoring, and periodic bias audits.

Repository: https://github.com/basil-emeokoro/MIT8301-Virtual-Lab
