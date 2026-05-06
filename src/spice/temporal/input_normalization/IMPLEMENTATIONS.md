# Concrete Input Normalizers

Input normalizers fit feature scaling statistics on training data and apply them to train, validation, test, and inference windows. Scaling prevents large-valued features from dominating optimization.

## Mental Model

Neural networks train better when inputs have comparable scale:

```text
normalized = (feature - mean) / scale
```

The important boundary is leakage. Statistics must be fit only from training windows. Validation, test, and evaluation data must use the training-fitted scaler.

## Beginner Theory: Standardization And Leakage

Standardization subtracts a training mean and divides by a training standard deviation:

```text
z = (x - mean_train) / std_train
```

This gives each feature a similar numeric scale. Without it, a large-valued feature can dominate gradients even if it is not more informative.

Leakage happens when information from validation, test, or evaluation data influences training. Fitting a scaler on all rows leaks future distribution information into the model. The safe pattern is:

```text
fit scaler on train rows
transform train rows with that scaler
transform validation/test/evaluation rows with that same scaler
```

## Shared Scaler Behavior

Both current normalizers produce a feature scaler and share the same transform helpers in `temporal.input_normalization.scaling`:

| Field | Meaning |
| --- | --- |
| `mean` | Per-feature training mean. |
| `scale` | Per-feature training standard deviation. |
| `safe_scale` | Zero-variance scales replaced with `1.0`. |

Zero variance is not an error. It means the feature was constant in the fitted rows; subtracting the mean is enough.

## `row_standard`

`row_standard` fits mean and variance over unique rows covered by training windows.

```text
training sample windows
  -> covered row ids
  -> unique rows
  -> unweighted mean/std
```

Each covered row contributes once, even if many training samples include it. This treats the training segment as a set of observed rows.

## `window_weighted_standard`

`window_weighted_standard` fits mean and variance over rows weighted by how often they appear inside training windows.

```text
row weight = number of training windows containing row
```

Rows that appear in many contexts have more influence. This matches the effective exposure of the model during training.

## Comparison

| Normalizer | Row contribution | Use |
| --- | --- | --- |
| `row_standard` | Once per covered row | Stable corpus-level row scaling. |
| `window_weighted_standard` | Once per window occurrence | Scaling matched to repeated sequence exposure. |

Current YAML training default selects `row_standard`.

## Data Flow

```text
train sample indices
  -> find covered context rows
  -> fit scaler
  -> transform problem-store feature_matrix
  -> build model batches
```

The transformed feature matrix keeps the same row order and shape.

## Invariants

| Rule | Why |
| --- | --- |
| Fit on training windows only. | Prevents validation/test/evaluation leakage. |
| Mean and scale length equals feature count. | Model input dimension must match. |
| Zero variance scale becomes `1.0`. | Avoids divide-by-zero. |
| Runtime artifact stores scaler. | Inference reuses training statistics. |

## Extension Pattern

A new normalizer should clearly define which training rows contribute and with what weights. It should return the same scaler shape and keep inference transform independent from evaluation data.

## Theory References

- scikit-learn `StandardScaler`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- scikit-learn data leakage guidance: https://scikit-learn.org/stable/common_pitfalls.html
