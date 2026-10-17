# Machine Learning Practice

My learning exercises and mini labs using Python, NumPy, Pandas, and scikit-learn. This repository tracks my progression from linear regression to classification, decision trees, random forests, and model evaluation.

These are educational exercises, not production systems. Classification examples use small hand-written or synthetic datasets; their scores do not establish real-world performance. My separate Court Forecast portfolio project is not included here.

## Practice folders

| Folder | Focus | Exercises |
| --- | --- | --- |
| [Day 2](day02_linear_regression/) | Linear regression, train/test splitting, MAE | Student score prediction and lesson notes |
| [Day 3](day03_classification/) | Logistic regression, probabilities, classification metrics | Purchases, student pass/fail, employee promotion |
| [Day 4](day04_decision_trees/) | Tree depth and model complexity | Synthetic loan-default comparison |
| [Day 5](day05_random_forests/) | Tree ensembles and feature importance | Synthetic machine failures; compare 10, 50, 100, and 300 trees |
| [Day 6](day06_model_evaluation/) | Stratified cross-validation and final evaluation | Delivery delays, five candidates, irrelevant-feature investigation |

## Run the exercises

From the repository root, using Python 3:

```sh
python -m venv .venv
```

Activate the environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies and run a self-contained lab:

```sh
python -m pip install -r requirements.txt
python day06_model_evaluation/delivery_delay_lab.py
```

The Day 3, Day 4, Day 5, and Day 6 lab scripts create their data in memory. The Day 2 score lab requires a CSV that is not currently available. The two files ending in `_notes.py` are lesson snippets requiring externally defined `X` and `y`; see their folder READMEs.

## Latest lab: delivery delay evaluation

The Day 6 lab compares three decision trees and two random forests using five-fold stratified cross-validation on 1,440 development rows, with 360 rows reserved for final testing. The selection rule is to maximize validation recall among candidates whose mean validation precision is at least 0.65.

The restricted forest was the only qualifying candidate. Removing the random `receipt_code` feature modestly improved its validation recall and F1 while preserving the precision requirement.

| Positive-class metric | CV with receipt code | CV without receipt code | Final test without receipt code |
| --- | ---: | ---: | ---: |
| Precision | 0.664 | 0.668 | 0.690 |
| Recall | 0.396 | 0.415 | 0.374 |
| F1 | 0.493 | 0.510 | 0.485 |

The final model meets the precision target but misses most late deliveries. The current script preserves the final feature choice; the before/after findings are documented in the Day 6 README.

## Learning progression and limitations

Earlier exercises compare settings on the same holdout set; those scores should be treated as exploratory validation results, not independent final estimates. Day 6 introduces a separate final test set and cross-validation for model selection. Training impurity-based feature importance is exploratory and does not prove that a feature generalizes or causes an outcome.

Scripts retain my practice structure and learning comments. Folder and file names have been standardized, and small note corrections are documented in the relevant folders. Dependencies are unpinned; exact output can vary with library versions.

