# Train/Test Strategy and Leakage QA

## Verified split

| Partition | Total | Good (0) | Bad/risky (1) | Good | Bad/risky |
|---|---:|---:|---:|---:|---:|
| Training | 800 | 560 | 240 | 70.0% | 30.0% |
| Independent test | 200 | 140 | 60 | 70.0% | 30.0% |

The implementation calls `train_test_split` with `test_size=0.20`, `random_state=42`, and `stratify=y`.

## Verified tuning sequence

1. Split the complete verified dataset.
2. Pass only `X_train` and `y_train` to GridSearchCV.
3. Use shuffled Stratified 5-Fold Cross-Validation within those 800 training rows.
4. Fit preprocessing separately inside each training fold through Pipeline and ColumnTransformer.
5. Select hyperparameters using mean cross-validated ROC-AUC.
6. Let GridSearchCV refit the best pipeline on all 800 training rows.
7. Evaluate once on the untouched 200-row test set.

No test observations participate in model selection, hyperparameter tuning, or threshold optimisation. The classification threshold is fixed at 0.5.

## Verified leakage prevention

Median imputation, categorical constant imputation, one-hot encoding, IQR clipping, and RobustScaler are learned inside the model pipelines. The scratch Logistic Regression uses a separate preprocessor fitted with `fit_transform(X_train, y_train)` and applies only `transform(X_test)` afterward. Full-dataset operations before the split are restricted to descriptive EDA and schema/target separation; no statistics learned there enter model fitting.

## Validation-set justification

With only 1,000 observations, reserving a separate validation set would reduce the data available for learning. Stratified five-fold cross-validation supplies rotating internal validation folds, allows every training observation to serve in validation, and preserves class proportions. The chosen 80% train + 20% test design therefore provides robust hyperparameter evaluation while retaining 800 observations for final model refitting.

**QA outcome: PASS - no data leakage identified and no methodological correction required.**
