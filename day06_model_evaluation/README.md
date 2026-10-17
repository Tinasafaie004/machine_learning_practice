# Day 6: Model evaluation

## Files

- `delivery_delay_lab.py`: self-contained lab generating 1,800 deliveries, comparing five candidates, inspecting training feature importances, running cross-validation, and evaluating the selected restricted forest on the final holdout.
- `cross_validation_notes.py`: lesson snippet requiring predefined `X` and `y`. It demonstrates custom scoring labels. The original `percision` typo and result-key mismatches were corrected in this repository copy.

```sh
python day06_model_evaluation/delivery_delay_lab.py
```

## Evaluation design

Reserve 20% (360 deliveries) for final testing. Use five-fold stratified cross-validation on the other 1,440: each round trains on 1,152 and validates on 288. Compare depths 2, 6, and unrestricted for individual trees, plus unrestricted and restricted 150-tree forests.

Select the highest mean validation recall among candidates with mean validation precision >= 0.65. The restricted forest (`max_depth=6`, `min_samples_leaf=8`, `random_state=42`) was the only qualifying candidate.

## Receipt-code investigation

The code originally included random `receipt_code` values. On the selected forest:

| Metric | With receipt code | Without receipt code |
| --- | ---: | ---: |
| Mean training recall | 0.5071 | 0.5152 |
| Mean validation recall | 0.3962 | 0.4152 |
| Mean validation precision | 0.6640 | 0.6675 |
| Mean validation F1 | 0.4934 | 0.5096 |
| Recall gap | 0.1110 | 0.1000 |
| Validation recall SD | 0.0517 | 0.0576 |

I removed the feature because it has no role in the generating formula and removal modestly improved validation performance. Not every candidate improved. Feature importance on training data alone cannot establish usefulness on new data. The committed script reflects the final version without the feature in `X`; it still generates the receipt column to preserve the original random-number sequence and dataset.

## Final evaluation

The selected forest is already fitted on all development rows in the feature-importance loop. Cross-validation fits separate copies, so the original fitted object remains available for final predictions.

Confusion matrix (rows = actual class, columns = predicted class; 1 = late):

| | Predicted on time | Predicted late |
| --- | ---: | ---: |
| Actually on time | 207 | 22 |
| Actually late | 82 | 49 |

Positive-class precision = 49/71 = 0.6901; recall = 49/131 = 0.3740; F1 = 0.4851. Accuracy = 0.7111. The model meets the precision requirement on this test set, but misses 82 of 131 actual late deliveries. Test recall is about 4.1 percentage points below the cross-validation mean. These results were reported without retuning against the final test set.
