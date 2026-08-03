# Feature ablation

Last updated: 2026-08-01, with 96 of 102 validation fits complete.

These observations are provisional. The campaign measures validation Cost over
optimum; it does not yet provide sealed-evaluation or classification results.
Only stable findings supported by matched full-model comparisons belong in the
manuscript.

## Manuscript-relevant observations

### Ethereum LSTM

The full model reaches 4.327% Cost over optimum, compared with 5.134% for the
base-fee-only model. The full feature set therefore improves the validation
metric by 0.807 percentage points.

Gas utilization and transaction count have the largest positive individual
contributions observed so far. Removing them increases Cost over optimum by
0.113 and 0.094 percentage points, respectively. Removing the P90 and P50
priority-fee features increases it by 0.033 and 0.022 percentage points.

Removing hour or day-of-week encoding slightly improves the result. The changes
are small and do not yet support a claim that calendar features are harmful.

### Ethereum Transformer

The full model reaches 4.438% Cost over optimum, compared with 5.309% for the
base-fee-only model. The full feature set improves the validation metric by
0.871 percentage points. Removing transaction count worsens the result to
4.532%, an increase of 0.094 percentage points. The close agreement with the
Ethereum LSTM ablation provides early evidence that transaction count is useful
across both architectures. Removing the P90 and P50 priority-fee features
increases Cost over optimum by 0.030 and 0.025 percentage points, closely
matching their modest contribution to the LSTM. Removing day-of-week encoding
improves the result to 4.363%, while removing hour encoding worsens it to
4.463%. Calendar effects are therefore not interchangeable and should not be
collapsed into one claim.

### Ethereum Transformer--LSTM

The full model reaches 4.412% Cost over optimum, compared with 5.238% for the
base-fee-only model. The full feature set improves the validation metric by
0.826 percentage points, closely matching the gains for the other two Ethereum
architectures. Removing gas utilization worsens the result to 4.579%, an
increase of 0.168 percentage points. This is the largest gas-utilization
contribution among the completed Ethereum architectures. Removing transaction
count worsens the result to 4.491%, an increase of 0.079 percentage points.
Transaction count therefore improves all three Ethereum architectures.
Removing base fee changes the result only slightly, to 4.419%. Removing hour
and day-of-week encoding improves the result to 4.242% and 4.317%, reductions
of 0.169 and 0.095 percentage points. Removing the P50 priority-fee feature
worsens the result to 4.434%, a modest 0.022 percentage-point increase that
matches its contribution to the other Ethereum architectures. In contrast,
removing P90 improves the result to 4.376%, a 0.036 percentage-point reduction.
The two quantiles therefore have opposing effects for this hybrid, as they do
for the Polygon Transformer--LSTM. The calendar effect is substantially larger
here than in the other completed Ethereum architectures.

### Polygon Transformer

The full model reaches 0.839% Cost over optimum, compared with 0.864% for the
base-fee-only model. The full feature set improves the validation metric by
0.025 percentage points. Removing base fee worsens the result to 0.958%, and
removing gas utilization worsens it to 0.922%. Both features therefore appear
economically useful for this architecture.

Removing block interval improves the result to 0.781%, while removing gas limit
improves it to 0.797%. This is evidence that either feature may be redundant or
harmful for the Polygon Transformer, but the current single-feature ablations
cannot identify their joint effect.

### Polygon LSTM

The full model reaches 0.829% Cost over optimum, compared with 0.849% for the
base-fee-only model. Removing base fee worsens the result to 0.923%, an increase
of 0.094 percentage points, while removing gas utilization worsens it to 0.859%,
an increase of 0.030 percentage points. Removing block interval improves the
result to 0.754%, a reduction of 0.075 percentage points. This matches the
direction observed for both Polygon attention-based architectures, making block
interval the first feature whose removal improves all three Polygon families.
Removing gas limit changes the result only slightly, to 0.831%, unlike the
material improvements for both attention-based families. The completed
calendar and priority-fee ablations differ from the full model by at most 0.007
percentage points and do not yet support substantive claims.

### Polygon Transformer--LSTM

The full model reaches 0.945% Cost over optimum, compared with 0.995% for the
base-fee-only model. The full feature set improves the validation metric by
0.050 percentage points. Removing gas limit improves the result to 0.824%, a
reduction of 0.122 percentage points. This agrees with the Polygon Transformer
and strengthens the evidence that gas limit is harmful or redundant for
Polygon's attention-based models. Removing block interval also improves the
result to 0.831%, a reduction of 0.114 percentage points, matching the
Transformer direction. Removing P90 improves the result to 0.905%, while
removing P50 worsens it to 0.957%; the two priority-fee quantiles therefore
have opposing effects for this model. Removing base fee worsens the result to
1.014%, while removing gas utilization currently improves it to 0.925%;
feature effects therefore still differ between the two architectures.

### Avalanche LSTM

The full model reaches 0.670% Cost over optimum, compared with 0.803% for the
base-fee-only model. The full feature set improves the validation metric by
0.134 percentage points. Removing gas utilization worsens the result to 0.773%,
an increase of 0.103 percentage points and the largest completed single-feature
effect. Removing P50, transaction count, base fee, gas limit, or P90 also
worsens the metric, by 0.028, 0.020, 0.017, 0.014, and 0.010 percentage points
respectively. Removing hour encoding worsens the result by 0.013 percentage
points; removing day-of-week encoding or block interval changes it by at most
0.005 percentage points. These smaller effects do not support substantive
claims.

### Avalanche Transformer

The full model reaches 0.719% Cost over optimum, compared with 0.949% for the
base-fee-only model. The full feature set therefore improves the validation
metric by 0.230 percentage points. Removing transaction count is effectively
neutral at 0.720%, while removing block interval worsens the result to 0.744%,
an increase of 0.025 percentage points. The block-interval direction therefore
differs from Polygon, where its removal improves all three completed
architecture comparisons. Removing hour and day-of-week encoding improves the
result to 0.707% and 0.696%, reductions of 0.012 and 0.023 percentage points.
Both calendar encodings therefore appear mildly harmful for this model, though
their joint effect is unknown. Removing gas utilization worsens the result to
0.842%, an increase of 0.123 percentage points and the largest completed
Avalanche Transformer feature effect. Removing P50 and P90 worsens the result
to 0.733% and 0.773%, increases of 0.014 and 0.054 percentage points. Both
priority-fee features therefore appear useful, with a larger contribution from
P90.

### Avalanche Transformer--LSTM

The full model reaches 0.762% Cost over optimum. Removing gas utilization
worsens the result to 0.868%, an increase of 0.106 percentage points. This
matches the direction and similar magnitude observed for the Avalanche LSTM
and Transformer, providing provisional cross-architecture evidence that gas
utilization is useful on Avalanche. Removing gas limit also worsens the result,
to 0.799%, an increase of 0.037 percentage points. Removing transaction count
worsens it similarly, to 0.796%, an increase of 0.034 percentage points.
Removing base fee improves the result to 0.670%; this architecture-specific
direction should not be generalized before the remaining matched ablations
complete.

### Training behavior

Ethereum LSTM candidates usually select epochs 4--8, while the base-fee-only
candidate selects epoch 16 and still performs substantially worse. Ethereum
Transformer candidates currently select epochs 8--18. Polygon Transformer
candidates usually select epochs 3--7, except for the model without base fee,
which selects epoch 14. Early stopping is reducing training substantially below
the 36-epoch cap.

## Follow-ups worth investigating

- Train both Polygon attention-based architectures without block interval and
  gas limit. Compare them with the full models and existing single-feature
  ablations to determine whether the apparent harm is independent, overlapping,
  or interactive. Both removals now improve both architectures.
- Test a Polygon-wide route without block interval. Its individual removal now
  improves all three architectures, but the current evidence remains tied to
  the pre-HPO control configuration.
- Revisit calendar-feature effects after all matched architectures and chains
  finish. Current directions differ by model, so they do not support one shared
  removal rule.
- Test joint removal of hour and day-of-week encoding for the Ethereum
  Transformer--LSTM. Both individual removals improve it materially, but the
  current ablation cannot determine their combined effect.
- Test joint removal of hour and day-of-week encoding for the Avalanche
  Transformer. Both individual removals improve it, but the current ablation
  cannot determine their combined effect.
- Compare feature effects across architectures only after every architecture has
  a completed full-model baseline. Current Ethereum results already suggest that
  feature utility may be architecture-specific.

These follow-ups require a separately approved experiment. They are not part of
the frozen 102-fit campaign.
