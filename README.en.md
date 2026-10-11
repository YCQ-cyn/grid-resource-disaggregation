# Feeder Resource Identification and PV Power Disaggregation

[简体中文](README.md)

Identify photovoltaic, energy-station and EV-charger operating signals from minute-level main-meter measurements, and estimate PV generation for the current minute. The two tasks operate independently and require only main-meter data at inference, without submeter labels, future observations, weather or geographic inputs.

## Business problem

Feeder power combines ordinary consumption, energy-station operation, EV charging and PV generation. PV offsets demand; equipment switching produces abrupt changes; energy-station activity can introduce sustained power changes or reversals. A single power reading cannot reliably identify the source of a change.

This project examines local events and their evolution over a history window to answer two questions: **Which resource operating signals are present now? How much PV power is being generated now?** Outputs support operational monitoring, generation estimation and business verification. They do not establish physical equipment installation.

## Methods

| Task | Model | History | Output |
|---|---|---|---|
| Multi-label resource identification | Causal TCN + forward-only LSTM | 120 minutes | Three independent probabilities and states |
| Real-time PV disaggregation | Small Causal S4D with two heads; TCN-LSTM dual-head comparator | 240 minutes | Current-minute PV power in kW |

**Resource identification.** The TCN extracts local jumps, drops, fluctuations and sustained segments. A unidirectional LSTM summarizes their historical evolution. Independent sigmoid outputs allow multiple resources to be active simultaneously. Submeters provide supervision only; missing labels remain unknown and are masked in loss and evaluation. Unpaired charger records support TRAIN-only EV auxiliary learning without inventing a feeder pairing.

**PV disaggregation.** A lightweight causal state-space backbone models historical dependencies. Activity and magnitude heads are combined as:

```text
PV power (kW) = sigmoid(activity) × softplus(magnitude) × target_scale_kw
```

Training jointly constrains final power, activity, active-period magnitude and inactive-period zero output. The reference is `max(-TotW_MA, 0)` from a paired PV submeter. Inference uses only main-meter features, independently of resource identification. This is estimation of the completed current minute, not a forecast of future power.

**Data and evaluation.** A fixed sequence of 56 electrical fields is augmented with validity, gap and freshness metadata: 114 channels for identification and 116 for PV. Dates are split before windows are generated; normalization is fitted on TRAIN only. Three seeds are trained, with checkpoint and threshold selection restricted to Validation. Short input gaps use past-only holds. Short target interpolation is confined to offline training within a split and receives reduced weight; primary evaluation uses observed references.

## Repository structure

```text
grid-resource-disaggregation/
├── README.md
├── README.en.md
├── pyproject.toml
├── config/
│   ├── fields.json                  Electrical field order
│   └── dataset.example.json         Private paths and date-split template
├── src/grid_disaggregation/
│   ├── cli.py                       Command-line workflows
│   ├── common.py                    Configuration utilities
│   ├── resource/
│   │   ├── models.py                Causal TCN and TCN-LSTM
│   │   ├── data.py                  Exact joins, masks and windows
│   │   ├── training.py              Three-seed validation-only selection
│   │   └── inference.py             Online resource identification
│   ├── pv/
│   │   ├── models.py                TCN-LSTM dual-head comparator
│   │   ├── state_space.py           Small Causal S4D
│   │   ├── data.py                  Pairing, missingness and split isolation
│   │   ├── features.py              Causal feature encoding
│   │   ├── training.py              Dual-head training
│   │   ├── metrics.py               Power, activity and energy evaluation
│   │   └── inference.py             Minute-by-minute PV disaggregation
│   └── preprocessing/
│       ├── extract_features.py      COMTRADE extraction entry point
│       ├── comtrade_io.py           Recording and archive readers
│       ├── basic_items.py           Basic electrical quantities
│       ├── harmonic_items.py        Harmonic quantities
│       ├── calculated_items.py      Minute-level calculated fields
│       └── formula_checks.py        Formula checks
├── docs/
│   ├── zh/
│   └── en/                          Methods, contracts and usage
├── tests/                           Data-independent tests
├── licenses/                        Third-party licenses
└── .github/workflows/               Automated tests
```

## Quick start

Python 3.11 or later. CPU inference is supported; CUDA is optional.

```sh
python -m pip install -e '.[dev]'
python -m grid_disaggregation --help
python -m pytest -q
```

Copy the configuration template to `config/dataset.local.json` and specify your own files and disjoint dates. See [usage](docs/en/usage.md) for training and inference. Measurement data, trained weights and fitted normalizers are not distributed.

## Evaluation and scope

Identification uses PR-AUC, Macro F1 and per-resource Precision, Recall and F1. PV evaluation includes MAE, Generation MAE, RMSE, R², Activity F1 and energy bias, with separate checks for inactive false energy, low-power errors and daily bias. The [experiment summary](docs/en/evaluation.md) distinguishes Validation, internal Test and reused chronological holdout results.

The energy-station label is an operating-power proxy, not independently verified storage charge/discharge truth. Some charger topology is unverified, so associated metrics are exploratory. Power decomposition from a main meter is not unique. Generalization across feeders, capacities and seasons requires independent evaluation.

- [Methodology](docs/en/methodology.md)
- [Data contract](docs/en/data_contract.md)
- [Training and inference](docs/en/usage.md)
- [Evaluation](docs/en/evaluation.md)
- [Third-party notices](NOTICE.md)
