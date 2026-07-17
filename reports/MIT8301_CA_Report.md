# MIT 8301 Continuous Assessment - Virtual Laboratory

## Cover Page and Student Details

**German Credit Risk Classification**  
**Student:** Basil Oforbuike Emeokoro  
**Matric Number:** 2025/A/MIT/0111  
**Student ID:** 301806653  
**Email:** b.emeokoro6653@miva.edu.ng  
**Programme:** Master of Information Technology  
**Specialisation:** Artificial Intelligence & Machine Learning  
**Course:** MIT 8301 - Machine Learning and AI Fundamentals  
**Academic Session:** 2026/2027

## Executive Summary

This assessment develops a reproducible classifier for bad credit risk. The supplied file contained 1,000 applicants and nine predictors but omitted the label. An authentic target-inclusive edition was reconciled with a 100% exact row-level predictor match and zero mismatched cells before labels were appended. The verified target contains 700 good and 300 bad applicants. Four models were trained and evaluated with leakage-safe preprocessing. Tuned Logistic Regression was selected under a predeclared performance-and-governance rule.

## Problem Statement and Objectives

The objective is to identify risky applicants (`credit_risk=1`) while recognising the asymmetric institutional and fairness costs of false negatives and false positives. The work covers EDA, missing values, outliers, encoding, scaling, scratch Logistic Regression, tuned model comparison, transparent feature interpretation, and responsible-use limitations.

## Dataset Description, Provenance, and Target Restoration

The supplied export had 1,000 rows, an index column, and nine predictors. The target source was the Hugging Face mirror of Kaggle's German Credit Risk - With Target edition. After removing the export index and normalising only whitespace/case, every predictor cell matched. No fuzzy or positional merge was used. `good` was encoded 0 and `bad` 1. SHA-256 checksums and per-feature match percentages are recorded in the provenance report.

## Exploratory Data Analysis

There are no duplicated complete rows. Age has 53 unique values, credit amount 921, and duration 33. The target is moderately imbalanced (70% good, 30% bad). Histograms, robust-scaled boxplots, categorical distributions, a missingness chart, a correlation heatmap, target comparisons, and observed risk by purpose appear in the figures appendix. These are descriptive associations, not causal findings.

## Missing Values and Outliers

Savings status has 183 missing values; checking status has 394. A dedicated `Unknown_or_No_Account` category preserves potentially informative absence and is learned inside training folds. IQR review found {'Age': 23, 'Credit amount': 72, 'Duration': 70}. Values are plausible, so they are not deleted. Training-fitted 1.5-IQR clipping limits leverage, followed by robust scaling. This avoids data leakage.

## Encoding, Scaling, and Split

The data were split into 800 training and 200 untouched test rows with stratification and random state 42. Age, amount, and duration use median imputation, IQR clipping, and RobustScaler. Nominal fields, including Job, use constant imputation and one-hot encoding with unknown-category tolerance. All transformations are fitted only on training data or within cross-validation folds.

## Training, Validation and Test Strategy

The dataset was partitioned into an 80% training set (800 observations) and a 20% independent test set (200 observations) using a stratified train-test split with `random_state=42`. The training set contains 560 good and 240 bad applicants, and the test set contains 140 good and 60 bad applicants; both therefore preserve the original 70% good / 30% bad class distribution.

Hyperparameter optimisation was performed exclusively on the training data using Stratified 5-Fold Cross-Validation within GridSearchCV. During this process, the 800 training observations were repeatedly divided into internal training and validation folds while preserving class proportions. Every training observation serves in validation across the rotating folds. After optimal hyperparameters were identified, GridSearchCV automatically refitted each model on the complete 800-row training set before a single evaluation on the untouched 200-row test set.

No separate validation dataset was required because cross-validation already creates internal validation folds. Although 60/20/20 and 70/15/15 partitions are also established approaches, this project intentionally adopted 80% train + 20% test + stratified 5-fold cross-validation because the dataset contains only 1,000 observations, cross-validation provides statistically stronger hyperparameter evaluation, and more observations remain available for model learning. This aligns with current scikit-learn and production machine-learning practice.

Missing-value imputation, categorical encoding, numerical scaling, IQR outlier clipping, and all feature transformations are implemented inside scikit-learn Pipeline and ColumnTransformer objects. They are fitted within training folds and subsequently applied to validation or test rows. GridSearchCV received only `X_train` and `y_train`; the decision threshold remained fixed at 0.5, so the test set influenced neither model selection, hyperparameter tuning, preprocessing, nor threshold optimisation.

## Logistic Regression from Scratch

The implementation provides a stable clipped sigmoid, explicit bias, binary cross-entropy, vectorised gradients, L2 regularisation, convergence tolerance, loss history, probability estimates, and input/fitted-state validation. It converged=True in 4233 iterations with final loss 0.50469. Its test performance is compared directly with scikit-learn.

## Model Training and Hyperparameter Tuning

Training used stratified five-fold GridSearchCV and ROC-AUC as the primary score. The test set was never used for tuning. Tuning controls regularisation, margin/kernel complexity, class weighting, and Naive Bayes variance smoothing, improving generalisation without test leakage.

- **Tuned Logistic Regression:** CV ROC-AUC 0.756; `{'model__C': 0.1, 'model__class_weight': None, 'model__penalty': 'l2', 'model__solver': 'liblinear'}`
- **Tuned SVM:** CV ROC-AUC 0.765; `{'model__C': 1, 'model__class_weight': 'balanced', 'model__gamma': 0.1, 'model__kernel': 'rbf'}`
- **Tuned Gaussian Naive Bayes:** CV ROC-AUC 0.666; `{'model__var_smoothing': 1e-12}`

## Test Evaluation and Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Tuned SVM | 0.710 | 0.511 | 0.750 | 0.608 | 0.786 |
| Tuned Logistic Regression | 0.730 | 0.594 | 0.317 | 0.413 | 0.766 |
| Logistic Regression from Scratch | 0.720 | 0.548 | 0.383 | 0.451 | 0.756 |
| Tuned Gaussian Naive Bayes | 0.675 | 0.472 | 0.700 | 0.564 | 0.702 |

A false negative classifies a risky applicant as good and may expose the institution to loss. A false positive classifies a good applicant as risky and may cause unfair rejection or harsher terms. Accuracy alone is therefore insufficient; recall, F1, ROC-AUC, thresholds, interpretability, and operating costs matter.

## Best-Model Justification

The predeclared rule prefers tuned Logistic Regression when its ROC-AUC is within 0.03 of the highest model, otherwise the highest-AUC model is selected. The result is **Tuned Logistic Regression**. This accepts a small discrimination trade-off for transparent coefficients, simpler deployment, auditability, and regulatory suitability. Operational threshold tuning could improve risky-class recall and must be set from validated costs, not the test set.

## Feature Importance and Interpretation

The largest absolute transformed Logistic Regression coefficients are:

- `cat__Checking account_Unknown_or_No_Account`: -0.879
- `num__Duration`: +0.514
- `cat__Checking account_little`: +0.471
- `cat__Saving accounts_Unknown_or_No_Account`: -0.392
- `cat__Purpose_radio/tv`: -0.360
- `cat__Housing_own`: -0.343
- `cat__Sex_male`: -0.330
- `cat__Saving accounts_little`: +0.313
- `cat__Purpose_education`: +0.307
- `cat__Saving accounts_rich`: -0.283

Positive values increase estimated bad-credit log-odds and negative values decrease them, conditional on the encoding and scaling. Coefficients express association, not causation. One-hot terms are interpreted relative to the omitted all-zero representation.

## Ethics, Fairness, and Responsible Use

Sex is sensitive and age can create protected-class concerns. Historical labels may encode past discrimination and other fields may serve as proxies. Supplementary metrics by sex are reported but small test subgroups cannot establish fairness. The model requires human review, reasons for adverse decisions, appeal routes, privacy controls, drift monitoring, periodic bias audits, and independent validation. Prediction is not causal judgement.

## Limitations

The dataset is small, historical, geographically specific, and omits important affordability and contemporary lending variables. Labels have no event-time detail; calibration and temporal stability are not established. The simplified nine-feature representation loses information from the original 20-feature UCI data. Reported test estimates have sampling uncertainty.

## Lessons Learned and Engineering Reflection

### Principal engineering challenge

The most consequential challenge was not model selection but the absence of a target variable in the supplied German Credit CSV. Supervised classification requires authentic observed outcomes: without a good/bad credit label, there is no valid response variable from which a classifier can learn or against which it can be evaluated. Creating synthetic labels, inferring them from predictor thresholds, clustering applicants and treating the clusters as outcomes, or deriving labels from the same features used for prediction would have introduced circular reasoning and invalidated the experiment. The appropriate engineering decision was therefore to pause supervised modelling until label provenance could be established. Preserving dataset integrity was more important than forcing a pipeline to produce metrics.

### Engineering solution and provenance control

The resolution began with an investigation of the supplied file's schema and the provenance of the simplified German Credit representation. An authentic target-inclusive edition with the same nine predictors was identified. The reconciliation process removed export-only identifiers, standardised column names, and normalised only harmless representation differences such as surrounding whitespace and demonstrably equivalent category case. It then compared every predictor, row by row and feature by feature, before accepting any label.

The verification produced a 100% match for every predictor and zero mismatched cells across 1,000 observations. SHA-256 checksums were recorded for both raw inputs, duplicate diagnostics and class counts were produced, and the procedure was documented in a machine-readable and human-readable validation report. Only after these checks passed was the authentic target appended and mapped to `credit_risk`, with 1 representing bad/risky credit and 0 representing good credit. This retained the original alignment between each applicant and the corresponding outcome.

This experience illustrates why data provenance is an operational requirement rather than an administrative detail. In a real machine-learning system, an apparently small join, row-order change, or unverified label source can silently corrupt the relationship between features and outcomes while leaving the code executable. Checksums, explicit reconciliation rules, fail-fast assertions, and durable validation reports make those risks observable and auditable.

### Machine-learning lessons

**Data quality.** Model quality depends first on the validity, alignment, and meaning of the data. A sophisticated estimator cannot compensate for an unauthenticated target or an incorrect feature-outcome join. The project also reinforced that reproducible preprocessing is more valuable than pursuing a marginal improvement in headline accuracy: the same inputs should pass through the same documented, testable transformations whenever the analysis is repeated.

**Preventing leakage.** Leakage was prevented by separating training and test data before any learned transformation entered the model workflow. Missing-value imputation, one-hot encoding, robust scaling, and IQR clipping were placed inside Pipeline and ColumnTransformer objects, causing each operation to learn its parameters from the relevant training fold only. The independent test set remained untouched until final evaluation. This matters because leakage can make a weak model appear deceptively successful by allowing information about held-out observations to influence preprocessing, tuning, or selection.

**Stratification.** The original target distribution contains 70% good credit and 30% bad credit. Stratified sampling preserved this balance exactly in the 800-row training set and 200-row test set. The resulting 560/240 and 140/60 class counts reduced the risk that an accidental shift in class proportions would distort comparison or make risky-class recall less reliable.

**Cross-validation.** Stratified 5-Fold Cross-Validation was selected instead of reserving a separate validation dataset. With only 1,000 observations, rotating validation folds make better use of the 800 training cases, provide more stable hyperparameter evidence than one small holdout, and preserve class proportions in every fold. GridSearchCV evaluated configurations only within those training folds and then refitted the selected configuration on the complete training set, supporting a stronger estimate of generalisation before the single test evaluation.

**Interpretability.** The tuned SVM achieved the highest test ROC-AUC, but tuned Logistic Regression was selected under the predeclared rule because its ROC-AUC was within 0.03 of the leader and its coefficients provide clearer decision logic. In credit-risk settings, engineering quality includes the ability to explain influential factors, reproduce a decision path, and support review by governance teams, auditors, and regulators. Predictive performance remains important, but a small gain from a less transparent model may not outweigh the operational value of understandable and contestable outputs.

### Responsible machine-learning reflection

Responsible credit-risk modelling requires controls tied to the actual decision context. Sex and age can expose applicants to disparate outcomes, and apparently neutral variables may act as proxies for protected characteristics. Transparency therefore includes documenting the target, transformations, evaluation population, decision threshold, subgroup diagnostics, and known limitations. Explainability must be usable by reviewers and affected applicants rather than treated as a decorative chart. Reproducible code, versioned data checks, tests, model documentation, approval controls, drift monitoring, and appeal procedures form part of governance. Most importantly, a prediction should support - not replace - professional judgement in a high-impact lending decision; adverse outcomes require human oversight and a meaningful route for review.

### Professional reflection

I found that this assignment strengthened my understanding of machine learning as an end-to-end engineering discipline rather than an isolated model-fitting exercise. Restoring the verified target required data engineering and provenance reasoning; the leakage-safe pipeline required careful separation of fitting and transformation; the scratch implementation connected mathematical concepts to tested software; and the tuning and evaluation stages required disciplined validation rather than selective reporting. Packaging the workflow with tests, reproducibility files, generated evidence, documentation, and Git history also demonstrated how software-engineering practices make analytical work easier to inspect, repeat, and maintain. The responsible-AI analysis further clarified that technical validity and accountable use must be designed together in credit-risk applications.

### Portfolio value

As a portfolio project, the completed repository demonstrates practical competence in data reconciliation and preprocessing, supervised classification, first-principles model implementation, hyperparameter optimisation, leakage-safe evaluation, explainable modelling, automated testing, reproducible research, report generation, and Git-based project management. The value lies in the traceable workflow from raw data to verified evidence, not in a single metric.

## Conclusion

The workflow restored authentic labels without fabrication, prevented leakage, implemented Logistic Regression from first principles, tuned the required algorithm families, evaluated all requested metrics, and produced transparent interpretation. The result is suitable for assessment and portfolio demonstration, not production lending.

## Reproducibility

Python 3.13.6; pandas 2.3.0; NumPy 2.3.1; scikit-learn 1.8.0; executed 2026-07-17T16:48:15.914192+00:00. Run the four commands in README.md from a clean environment.

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
