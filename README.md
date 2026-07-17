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

## Training, Validation and Test Strategy

The 1,000 observations are split into 800 training rows and 200 independent test rows using `test_size=0.20`, `random_state=42`, and `stratify=y`. The training set contains 560 good and 240 bad applicants; the test set contains 140 good and 60 bad applicants, preserving the 70%/30% distribution in both.

GridSearchCV receives only the training set and uses shuffled Stratified 5-Fold Cross-Validation. These rotating folds provide internal validation, so a separate validation set is unnecessary. After selection, GridSearchCV refits the best configuration on all training rows and the model is evaluated once on the untouched test set. The threshold remains fixed at 0.5.

Imputation, one-hot encoding, robust scaling, and training-fitted IQR clipping are contained in scikit-learn `Pipeline` and `ColumnTransformer` objects. No preprocessing statistics, model-selection decisions, hyperparameters, or thresholds are learned from the test set. This 80% train + 20% test + cross-validation strategy retains more learning data in a relatively small dataset while providing robust hyperparameter evaluation.

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

## Key Engineering Lessons

1. **Authenticate outcomes before modelling.** Supervised learning requires genuine labels; the missing target was restored only after 100% row-level predictor reconciliation and checksum-backed provenance validation.
2. **Treat preprocessing as fitted model behaviour.** Imputation, encoding, scaling, and outlier clipping belong inside training-fitted pipelines so test information cannot leak into the model.
3. **Use limited data efficiently.** A stratified 80/20 split with Stratified 5-Fold Cross-Validation preserved the 70%/30% class balance while retaining more observations for learning and robust tuning.
4. **Balance performance with governance.** Logistic Regression was preferred when close to the leading SVM because transparent coefficients improve auditability, explanation, and operational review in lending.
5. **Make evidence reproducible.** Assertions, tests, provenance reports, generated figures, documented environments, and Git history turn an analysis into an inspectable engineering artifact.

Repository: https://github.com/basil-emeokoro/MIT8301-Virtual-Lab
