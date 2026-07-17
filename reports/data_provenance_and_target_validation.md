# Data Provenance and Target Validation

- **Outcome:** PASS
- **Rows:** 1,000
- **Mismatched predictor cells:** 0
- **Class balance:** 700 good (70.0%), 300 bad/risky (30.0%)
- **Supplied SHA-256:** `2b129ca0157266a6439be4fdcc908b18da16261625465d22a7f6f7f1a51d3886`
- **Candidate SHA-256:** `2b786d93d243056258f2a96d6b0c0898bc2c459c0a845cdd7f431d22b54a7c36`
- **Candidate source:** Hugging Face mirror of Kaggle German Credit Risk - With Target
- **Source URL:** https://huggingface.co/datasets/AiresPucrs/german-credit-data
- **Download date (UTC):** 2026-07-17

## Method

The export index was removed, harmless whitespace/case differences were normalised, and all nine predictors were compared row by row. The target was appended only after a 100% exact predictor match.

## Per-feature exact match

| Feature | Exact match |
|---|---:|
| Age | 100.0% |
| Sex | 100.0% |
| Job | 100.0% |
| Housing | 100.0% |
| Saving accounts | 100.0% |
| Checking account | 100.0% |
| Credit amount | 100.0% |
| Duration | 100.0% |
| Purpose | 100.0% |

The final target is `credit_risk`: 1 means bad/risky credit and 0 means good credit.