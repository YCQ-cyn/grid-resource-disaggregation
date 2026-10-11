#!/usr/bin/env python3
"""Fast deterministic checks for the electrical and minute calculations."""

from __future__ import annotations

import math

import numpy as np

from .basic_items import calculate_basic_items
from .calculated_items import (
    CALCULATED_FIELDS,
    MinuteCalculationConfig,
    calculate_minute_items,
)
from .harmonic_items import analyze_harmonics


def assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: actual={actual}, expected={expected}, tolerance={tolerance}")


def main() -> int:
    sample_rate = 12_800.0
    seconds = 1
    time = np.arange(int(sample_rate * seconds), dtype=np.float64) / sample_rate
    voltage_rms = 220.0
    current_rms = 10.0
    lag = np.deg2rad(30.0)
    phase_angles = np.asarray([0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0])
    voltage = np.column_stack(
        [np.sqrt(2.0) * voltage_rms * np.sin(2.0 * np.pi * 50.0 * time + angle) for angle in phase_angles]
    )
    current = np.column_stack(
        [
            np.sqrt(2.0) * current_rms * np.sin(2.0 * np.pi * 50.0 * time + angle - lag)
            for angle in phase_angles
        ]
    )
    waveform = np.column_stack([voltage, current])
    harmonic = analyze_harmonics(waveform, sample_rate, 50.0)
    basic = calculate_basic_items(waveform, harmonic, rated_capacity_kva=100.0)

    expected_phase_p = voltage_rms * current_rms * math.cos(lag)
    expected_phase_q = voltage_rms * current_rms * math.sin(lag)
    assert_close(basic.fields["PhV_phsA"], voltage_rms, 1e-9, "phase voltage RMS")
    assert_close(basic.fields["A_phsA"], current_rms, 1e-9, "phase current RMS")
    assert_close(basic.fields["PhW_phsA"], expected_phase_p, 1e-6, "phase active power")
    assert_close(basic.fields["PhVar_phsA"], expected_phase_q, 1.0, "phase reactive power")
    assert_close(basic.fields["TotPF"], math.cos(lag), 1e-9, "total power factor")
    assert_close(harmonic.fields["HphV1_phsA"], voltage_rms, 0.1, "voltage fundamental")
    assert_close(harmonic.fields["HA1_phsA"], current_rms, 0.01, "current fundamental")
    assert_close(basic.fields["ImbNgV"], 0.0, 1e-6, "voltage imbalance")
    assert_close(basic.fields["ImbNgA"], 0.0, 1e-6, "current imbalance")

    minute_rows = []
    for second_id in range(60):
        row = {
            "second_id": second_id,
            "sample_count": int(sample_rate),
            "expected_samples": int(sample_rate),
            "_valid_duration_s": 1.0,
            "is_complete_second": 1,
            "source_recordings": "synthetic",
        }
        row.update(basic.fields)
        row.update(basic.internal)
        row.update(harmonic.fields)
        minute_rows.append(row)
    calculated = calculate_minute_items(minute_rows)
    if len(CALCULATED_FIELDS) != 56 or CALCULATED_FIELDS[-1] != "TotVar_MTS":
        raise AssertionError("Excel 计算字段必须为前56项，并以 TotVar_MTS 结束")
    assert_close(calculated["TotW_MA"], 3.0 * expected_phase_p / 1000.0, 1e-9, "minute active mean")
    assert_close(calculated["TotVar_MA"], 3.0 * expected_phase_q / 1000.0, 0.005, "minute reactive mean")
    assert_close(calculated["TotW_PSD"], 0.0, 1e-12, "active population std")
    assert_close(calculated["TotW_MTS"], 0.0, 1e-12, "active trend")
    assert_close(calculated["TotPF_AA"], math.cos(lag), 1e-9, "minute arithmetic PF")

    dynamic_rows = []
    for second_id, original in enumerate(minute_rows):
        row = dict(original)
        in_step = 20 <= second_id < 40
        row["TotW"] = 3000.0 if in_step else 0.0
        row["PhW_phsA"] = 3000.0 if in_step else 0.0
        row["PhW_phsB"] = 0.0
        row["PhW_phsC"] = 0.0
        row["TotVar"] = 6000.0 if in_step else 0.0
        row["PhVar_phsA"] = 6000.0 if in_step else 0.0
        row["PhVar_phsB"] = 0.0
        row["PhVar_phsC"] = 0.0
        for phase in "ABC":
            for order in range(1, 26):
                row[f"HA{order}_phs{phase}"] = (
                    10.0 if order == 1 else 1.0 if order == 2 else 0.0
                )
        dynamic_rows.append(row)
    config = MinuteCalculationConfig(
        event_threshold_kw=0.5,
        event_threshold_kvar=0.5,
    )
    dynamic = calculate_minute_items(dynamic_rows, config)
    assert_close(dynamic["TotW_MCU"], 3.0, 1e-12, "three-second rise")
    assert_close(dynamic["TotW_MCD"], 3.0, 1e-12, "three-second drop")
    assert_close(dynamic["TotVar_MCU"], 6.0, 1e-12, "Q at P rise")
    assert_close(dynamic["TotVar_MCD"], -6.0, 1e-12, "Q at P drop")
    assert_close(dynamic["A_SCC"], 0.1, 1e-12, "three-phase harmonic ratio")
    assert_close(dynamic["ImbNg_TotW"], 1.0, 1e-9, "phase-change imbalance")
    assert_close(dynamic["TotW_MEC"], 2.0, 1e-12, "merged P events")

    missing_last5 = [dict(row) for row in minute_rows]
    for row in missing_last5:
        if int(row["second_id"]) >= 55:
            row["_valid_duration_s"] = 0.0
            row["sample_count"] = 0
            row["is_complete_second"] = 0
    missing_result = calculate_minute_items(missing_last5)
    if math.isfinite(missing_result["TotW_MEA"]):
        raise AssertionError("固定第55～59秒缺失时，MEA 必须缺失，不能向前补值")
    print("self_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
