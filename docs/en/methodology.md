# Methodology

[简体中文](../zh/methodology.md)

## Tasks and metering boundary

With positive main power representing import, a typical relation is:

```text
P_main = P_load + P_energy_station + P_charger − P_pv
```

Energy-station power may be bidirectional. Meter polarity, topology and timing must be established for a valid reference. Unrelated submeter records cannot be shifted into artificial synchronous truth.

Both outputs describe the completed current minute. Identification detects operating signals; disaggregation estimates generation. Neither consumes the other's output.

## Resource identification

The 120-minute input contains 56 main-meter fields, 56 validity indicators, measurement_mask and gap_indicator. TRAIN-only median, IQR and 0.1%/99.9% bounds normalize the electrical fields. Missing values become neutral normalized zeros while masks preserve their missingness.

The TCN uses widths 16/16/24/32, kernel size 5 and dilations 1/2/4/8. Residual blocks contain two causally trimmed convolutions. A one-layer, 32-unit forward-only LSTM processes the sequence. The final timestep produces three logits with independent sigmoid outputs. The 114-channel configuration contains 38,555 parameters.

Targets are PV activity, energy-station power ≥5kW and charger power ≥1kW. The PV threshold is fitted to TRAIN nighttime noise. Energy and charger labels are operational proxies. Unobserved, low-quality or semantically unknown targets are excluded from supervision. Unpaired charger records supervise only the third label during TRAIN-only auxiliary learning.

Loss is per-class normalized masked BCE, with TRAIN positive weights clipped to 0.25–10. Three seeds undergo auxiliary pretraining and main-meter training. Checkpoints use Validation Macro PR-AUC; thresholds use Validation F1; seed selection uses Validation Macro PR-AUC followed by Macro F1.

## PV disaggregation

The 240-minute input contains 56 electrical fields and their validity indicators, plus measurement_mask, gap_indicator, measurement_age_scaled and sample_age_scaled. Small Causal S4D uses a 116→32 projection, two width-32/state-24 diagonal state-space blocks and two final-timestep heads, totaling 9,314 parameters. LayerNorm acts on channels; convolution is causal, without future pooling or bidirectional recurrence.

Final power is sigmoid(activity)×softplus(magnitude)×TRAIN target scale. The normalized power loss is SmoothL1 with beta=0.02 and active-period weight 2.5, plus 0.20 activity BCE, 0.10 active magnitude supervision and 0.20 inactive zero-output loss. Interpolated targets receive weight 0.25.

Minibatches sample 75% new-domain and 25% historical auxiliary endpoints. New-domain TRAIN determines center/IQR and target scale; pooled TRAIN expands clipping bounds. The TCN-LSTM dual-head comparator shares inputs, loss and evaluation. Selection prioritizes Validation MAE, then Generation MAE, RMSE and daily absolute energy bias. Test is excluded from selection.

## Missingness and causality

Main-meter values can be held for at most three past minutes; future-based input interpolation is forbidden. Offline PV targets may be interpolated only within the same split across bounded gaps of at most three minutes. Primary metrics use observed references. Splits precede window construction, with history reset across roles.

The existing model retains a 00:00–00:14 gap policy from its data protocol. This is not a universal metering assumption. Short histories receive neutral zero padding; input_available must be checked independently of a numeric output.

## References

- Bai et al., *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*, 2018: [paper](https://arxiv.org/abs/1803.01271).
- Gu et al., *On the Parameterization and Initialization of Diagonal State Space Models*, 2022: [paper](https://arxiv.org/abs/2206.11893).
- [Official S4 implementation](https://github.com/state-spaces/s4): diagonal-kernel parameterization; see third-party notices.
