# Day 5: Random forests

`machine_failure_comparison.py` generates 2,500 synthetic machine records. Failure probability depends on several risk factors, with a random draw determining the outcome. Pressure is generated but is not used in that risk formula.

The exercise compares a depth-5 decision tree with depth-5 forests containing 10, 50, 100, and 300 trees. It prints classification metrics, tree depths, and impurity-based feature importances.

```sh
python day05_random_forests/machine_failure_comparison.py
```

Learning focus: ensembles, noisy targets, the number of trees, missed failures, and feature importance. Repeated comparisons on the same holdout are exploratory; an independent final estimate would require separate test data. A range comment was corrected to match machine ages 1 through 20.
