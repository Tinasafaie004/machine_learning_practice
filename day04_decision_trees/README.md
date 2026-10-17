# Day 4: Decision trees

`loan_default_depth_comparison.py` generates 300 synthetic observations with noisy labels. It compares depths 1 through 5 and an unrestricted tree, then prints detailed reports for depths 4 and 5.

```sh
python day04_decision_trees/loan_default_depth_comparison.py
```

Learning focus: tree depth, training versus holdout performance, confusion matrices, and recall. The reused holdout acts as validation during these comparisons; it is not an untouched final test. A comment was clarified: `max_depth=None` permits unrestricted depth but does not automatically prove overfitting.
