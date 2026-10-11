# Data contract

[简体中文](../zh/data_contract.md)

## Private data configuration

`config/dataset.example.json` contains path placeholders and empty date lists. Paths resolve relative to the configuration file. Supply your own files and dates before training.

Each batch defines id, main, pv, optional energy/charger, async, energy_supervised and dates. Dates use YYYY-MM-DD. Resource roles are train/validation/internal_test/domain_test; PV roles are train/validation/test. A calendar date must not cross TRAIN and evaluation roles or appear in multiple roles in one batch. Set energy_supervised=false for an unverified energy reference.

PV new-domain IDs start with `new`; historical IDs use another prefix. Domain-weighted PV training requires both groups. New-domain dates provide Validation and Test; historical data are TRAIN-only. Auxiliary charger entries define device and path and are excluded from evaluation dates.

## Natural-minute CSV

Records contain unique minute-aligned minute_start, coverage_ratio and the [56 ordered fields](../../config/fields.json). Use a consistent UTC+8 local minute clock. Fields are read by name; fields.json defines encoding order. A complete observation requires coverage_ratio≥0.999. Poor-quality labels must not become OFF labels.

Main, PV, energy-station and charger streams are separate CSVs. Resource labels use exact minute_start joins, without nearest matching or artificial offsets. Verify polarity; the PV reference is max(-TotW_MA,0). Model inputs come only from main measurements.

## Asynchronous records

With async=true, CSVs provide data_time, report_time and the 56 fields. Timezone-aware timestamps retain their stated zone; naive source timestamps are interpreted as UTC and converted to UTC+8. The first arrival for a sample is retained. Main records use report-time minute bins; PV references use sampling-time bins. This is an asynchronous proxy reference, not certified synchronous minute truth. Resource label preparation excludes this format.

## Model channels

| Channels | Resource | PV |
|---|---:|---:|
| Normalized main electrical fields | 56 | 56 |
| Validity indicators | 56 | 56 |
| measurement_mask | 1 | 1 |
| gap_indicator | 1 | 1 |
| measurement_age_scaled | — | 1 |
| sample_age_scaled | — | 1 |

Measurement age is minutes since the last valid total-active-power observation, clipped to four and divided by four. Sample age is seconds from the last valid sample to minute end, clipped to 0–300 and divided by 300. PV holds past valid fields for at most three missing minutes while leaving their validity indicators false. Inputs must not use submeter targets, future records or evaluation-fitted transformations.

## Local artifacts

Resource training creates selected.pt, selection.json, data/normalizer.json and per-label metrics. PV training creates a release directory with a checkpoint, manifest.json, normalizer.json and evaluation tables. These artifacts are generated locally and are not distributed in the repository.

PV output includes pv_power_kw (positive generation in kW), internal activity_probability, magnitude_kw, input_available and gap_indicator. Resource output contains independent operating-proxy probabilities and states. PV disaggregation activity does not replace the resource model's PV identification output.
