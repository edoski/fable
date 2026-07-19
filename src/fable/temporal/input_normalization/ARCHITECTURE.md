# Input Normalization Architecture

## Purpose

`temporal.input_normalization` owns fitted scaler facts and feature-matrix transforms.
SPICE now has one internal policy: row-standard scaling over rows covered by the
training split.

## Flow

```text
training problem store + train sample indices
        |
        v
fit_row_standard_scaler()
        |
        v
ScalerStats
        |
        v
transform_problem_store_features(problem_store, scaler)
```

The saved `ScalerStats` is reused during validation, test, evaluation, and serving so
all non-training data is transformed with training-time statistics.

Scaler fitting uses scikit-learn `StandardScaler` for mean and scale statistics. The
durable boundary remains SPICE-owned `ScalerStats`, and replay uses SPICE's float32
transform helper so stored artifacts do not depend on a live sklearn object.
