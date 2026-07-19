# Concrete Input Normalization

SPICE fits one row-standard scaler on training rows and applies it everywhere else.
Scaling prevents large-valued features from dominating optimization.

```text
normalized = (feature - mean_train) / scale_train
```

The leakage rule is simple: statistics must come only from rows covered by training
windows. Validation, test, evaluation, and serving data use the persisted training
scaler.

## `ScalerStats`

| Field | Meaning |
| --- | --- |
| `means` | Per-feature training means. |
| `scales` | Per-feature training standard deviations, with zero-variance scales stored as `1.0`. |

Zero variance is not an error. It means the feature was constant in fitted rows;
subtracting the mean is enough.

## Row-Standard Fit

`fit_row_standard_scaler()` fits mean and variance over unique feature rows covered
by the selected training windows.

```text
training sample windows
  -> covered row ids
  -> unique rows
  -> unweighted mean/std
```

Each covered row contributes once, even if many training samples include it.
