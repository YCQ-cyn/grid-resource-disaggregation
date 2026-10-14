# Evaluation

[简体中文](../zh/evaluation.md)

## Metric definitions

PR-AUC is average precision over known labels. Macro F1 averages per-label F1. Generation MAE includes only reference power above the TRAIN activity threshold. Energy bias is cumulative prediction-minus-reference divided by cumulative reference. Minute-level kW is divided by 60 for kWh; interrupted evaluation covers only valid evaluated minutes.

PV evaluation also checks inactive MAE, inactive predicted energy, active 0–2kW MAE, daily absolute energy bias and ramp errors. Correlation or visual overlap alone is insufficient.

## Reproduction and generalization

Explicit date roles precede history-window construction. Normalization, target noise thresholds and power scales use TRAIN only; classification thresholds use Validation. Three seeds retain their best checkpoints independently. Public tests verify numerical behavior, causality and data routing rather than private-data accuracy.

Operational evaluation needs an unseen feeder/batch covering small capacity, sustained generation, load disturbances and acquisition failures. Do not tune coefficients or choose seeds from the evaluated Test set.
