# Training and inference

[简体中文](../zh/usage.md)

Run commands from the repository root. Measurements, weights and runtime artifacts are supplied or generated locally.

## Installation and checks

```sh
python -m pip install -e '.[dev]'
python -m grid_disaggregation --help
python -m pytest -q
python -m grid_disaggregation.preprocessing.formula_checks
```

COMTRADE archives can use system 7z, or the optional `.[archives]` dependency.

```sh
python -m grid_disaggregation.preprocessing.extract_features --help
python -m grid_disaggregation.preprocessing.extract_features --input data/recordings --output-dir outputs/features
```

This exports basic quantities, harmonics and 56 minute-level calculated fields. Models consume the calculated fields, not every exported quantity.

## Configure measurements

```sh
cp config/dataset.example.json config/dataset.local.json
```

Specify main/PV/energy/charger paths, identities, metering relationships and explicit date roles. Resource training requires exact natural-minute observations. PV can use configured asynchronous records. New-domain IDs start with new; historical auxiliary data are TRAIN-only. Independent charger records may support auxiliary training but do not establish a main-meter pairing. See the [data contract](data_contract.md).

Replace placeholder paths with your own files. Commands do not download measurements or pretrained weights.

## Resource training and inference

```sh
python -m grid_disaggregation resource-train --plan config/dataset.local.json --output outputs/resource
python -m grid_disaggregation resource-predict --input data/main.csv --model-dir outputs/resource --output outputs/resource_predictions.csv
```

Three seeds are trained. Optional charger records support EV-only pretraining. Each seed retains its best Validation checkpoint; selection and thresholds use Validation only. Evaluation ignores unknown labels and skips absent holdout roles.

## PV preparation, training and inference

```sh
python -m grid_disaggregation pv-prepare --plan config/dataset.local.json --output outputs/pv_data
python -m grid_disaggregation pv-train --table outputs/pv_data/training_table.csv --normalizer outputs/pv_data/normalizer.json --method small_s4d --output outputs/pv
python -m grid_disaggregation pv-predict --input data/main.csv --model-dir outputs/pv/release --output outputs/pv_predictions.csv
```

Use `--method tcn_lstm` and a separate output directory for the comparator. Domain-weighted training requires usable TRAIN endpoints in both new and historical groups. The configured workflow rebuilds data and normalizers; it does not reproduce undistributed private date assignments. The public modules preserve the existing model, encoding and loss mathematics.

## Online integration

```python
from grid_disaggregation.pv.inference import OnlinePV

runner = OnlinePV('outputs/pv/release')
result = runner.update('2026-01-01 10:00:00', record, observed=True)
```

record is a main-meter field dictionary and may include sample_time_main/arrival_time_main. Submit only completed minutes in increasing order. The current runtime requires source sample/availability times before minute end; asynchronous source CSVs use the existing canonical_csv binning rule. Do not submit future records early.

The runtime retains history and normalization in memory and inserts missing-minute records internally. Check input_available independently of a numeric prediction. These Python modules do not implement device hardware deployment.
