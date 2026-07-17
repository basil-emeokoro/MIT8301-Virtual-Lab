# Methodology

The workflow uses a stratified 80/20 split and stratified five-fold cross-validation. Numerical features are median-imputed, capped using training-fitted 1.5-IQR fences, and robustly scaled. Nominal variables, including Job, are constant-imputed and one-hot encoded. The test set is untouched until final evaluation.
