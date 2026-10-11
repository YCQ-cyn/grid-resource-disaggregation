# Evaluation

[简体中文](../zh/evaluation.md)

## Existing experiment summary

These figures come from existing training results; no retraining was performed for this repository. They describe private data protocols. Measurements and weights needed to reproduce these exact figures are not distributed.

Resource identification uses a 114-channel, 120-minute TCN plus forward-only LSTM. Checkpoints and final seed selection use Validation PR-AUC/F1 only.

| Resource | Validation PR-AUC | Validation F1 | Internal Test PR-AUC | Internal Test F1 |
|---|---:|---:|---:|---:|
| PV | 0.9707 | 0.9278 | 0.9448 | 0.8436 |
| Energy-station proxy | 0.8627 | 0.8940 | 0.9093 | 0.9066 |
| Charger operating proxy | 0.7242 | 0.7806 | 0.7986 | 0.7241 |
| Macro | 0.8525 | 0.8675 | 0.8842 | 0.8247 |

Charger reference topology is not independently certified; its metrics do not constitute external acceptance on a verified mixed feeder. Energy labels are power-threshold proxies.

PV uses a 116-channel, 240-minute Small Causal S4D with two heads. Test is excluded from selection.

| Split | MAE kW | Generation MAE kW | RMSE kW | R² | Activity F1 | Energy bias |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.00772 | 0.15748 | 0.04994 | 0.94449 | 0.89835 | −16.00% |
| Chronological Test | 0.03798 | 0.23166 | 0.14519 | 0.98835 | 0.54658 | +1.52% |

Test reuses a previously inspected batch and is not an independent external evaluation. Validation has limited generating-day diversity and many zero targets. Low overall MAE does not resolve active-period errors, declining activity F1 or low-capacity generalization.

## Metric definitions

PR-AUC is average precision over known labels. Macro F1 averages per-label F1. Generation MAE includes only reference power above the TRAIN activity threshold. Energy bias is cumulative prediction-minus-reference divided by cumulative reference. Minute-level kW is divided by 60 for kWh; interrupted evaluation covers only valid evaluated minutes.

PV evaluation also checks inactive MAE, inactive predicted energy, active 0–2kW MAE, daily absolute energy bias and ramp errors. Correlation or visual overlap alone is insufficient.

## Reproduction and generalization

Explicit date roles precede history-window construction. Normalization, target noise thresholds and power scales use TRAIN only; classification thresholds use Validation. Three seeds retain their best checkpoints independently. Public tests verify numerical behavior, causality and data routing rather than private-data accuracy.

Operational evaluation needs an unseen feeder/batch covering small capacity, sustained generation, load disturbances and acquisition failures. Do not tune coefficients or choose seeds from the evaluated Test set.
