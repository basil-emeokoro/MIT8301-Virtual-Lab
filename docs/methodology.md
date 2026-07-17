# Methodology

The workflow uses a stratified 80/20 split and stratified five-fold cross-validation. Numerical features are median-imputed, capped using training-fitted 1.5-IQR fences, and robustly scaled. Nominal variables, including Job, are constant-imputed and one-hot encoded. The test set is untouched until final evaluation.

## Training, Validation and Test Strategy

The split contains 800 training observations (560 good, 240 bad) and 200 test observations (140 good, 60 bad), preserving 70% good and 30% bad in each. GridSearchCV operates only on the training set with shuffled Stratified 5-Fold Cross-Validation and automatically refits the selected model on all training rows. Rotating validation folds remove the need to reserve a separate validation set from this relatively small 1,000-row dataset. All learned preprocessing is inside Pipeline and ColumnTransformer objects; the test set is used once for final evaluation and never for tuning or threshold optimisation.
