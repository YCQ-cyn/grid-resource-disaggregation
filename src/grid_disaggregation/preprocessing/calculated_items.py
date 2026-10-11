#!/usr/bin/env python3
"""Calculate the 56 minute-level fields required by the Excel data-item block.

The Excel workbook defines the output schema. The formulas for matching fields
follow the supplied minute-interface Markdown documents. The final three Excel
items (TotPF_AC, TotW_TotVA and A_EC) are intentionally excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


CALCULATED_FIELDS = [
    "TotW_MA", "PhW_phsA_MA", "PhW_phsB_MA", "PhW_phsC_MA",
    "TotVar_MA", "PhVar_phsA_MA", "PhVar_phsB_MA", "PhVar_phsC_MA",
    "PhV_phsA", "PhV_phsB", "PhV_phsC", "A_phsA", "A_phsB", "A_phsC",
    "TotPF_AA",
    "TotW_PSD", "PhW_phsA_PSD", "PhW_phsB_PSD", "PhW_phsC_PSD",
    "TotVar_PSD", "PhVar_phsA_PSD", "PhVar_phsB_PSD", "PhVar_phsC_PSD",
    "TotW_MCU", "PhW_phsA_MCU", "PhW_phsB_MCU", "PhW_phsC_MCU",
    "TotW_MCD", "PhW_phsA_MCD", "PhW_phsB_MCD", "PhW_phsC_MCD",
    "TotW_MEA", "PhW_phsA_MEA", "PhW_phsB_MEA", "PhW_phsC_MEA",
    "A_SCC", "TotVar_MCU", "TotVar_MCD",
    "TotVar_MEA", "PhVar_phsA_MEA", "PhVar_phsB_MEA", "PhVar_phsC_MEA",
    "TotW_AAC", "PhW_phsA_AAC", "PhW_phsB_AAC", "PhW_phsC_AAC",
    "TotVar_AAC", "PhVar_phsA_AAC", "PhVar_phsB_AAC", "PhVar_phsC_AAC",
    "ImbNg_TotW", "ImbNg_TotVar", "TotW_MEC", "TotVar_MEC",
    "TotW_MTS", "TotVar_MTS",
]


@dataclass(frozen=True)
class MinuteCalculationConfig:
    """Configurable thresholds which the Markdown deliberately leaves configurable."""

    minute_min_coverage: float = 0.999
    last5_min_coverage: float = 0.999
    min_valid_seconds: int = 1
    min_valid_adjacent_pairs: int = 1
    min_harmonic_windows: int = 1
    fundamental_current_min_a: float = 1e-6
    event_threshold_kw: float = 0.0
    event_threshold_kvar: float = 0.0
    imbalance_epsilon: float = 1e-12
    imbalance_stable_threshold: float = 1e-9


def _number(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")


def _duration(row: Mapping[str, object]) -> float:
    """Return the real valid duration represented by a natural-second row."""

    direct = _number(row.get("_valid_duration_s"))
    if np.isfinite(direct):
        return float(np.clip(direct, 0.0, 1.0))
    sample_count = _number(row.get("sample_count"))
    expected = _number(row.get("expected_samples"))
    if np.isfinite(sample_count) and np.isfinite(expected) and expected > 0:
        return float(np.clip(sample_count / expected, 0.0, 1.0))
    return 1.0 if int(row.get("is_complete_second", 0)) == 1 else 0.0


def _is_complete(row: Mapping[str, object]) -> bool:
    try:
        return int(row.get("is_complete_second", 0)) == 1
    except (TypeError, ValueError):
        return False


def _weighted_mean(
    rows: Sequence[Mapping[str, object]],
    key: str,
    scale: float,
    minimum_coverage: float,
    window_seconds: float,
) -> float:
    numerator = 0.0
    valid_duration = 0.0
    for row in rows:
        value = _number(row.get(key))
        duration = _duration(row)
        if np.isfinite(value) and duration > 0:
            numerator += value * duration / scale
            valid_duration += duration
    if valid_duration <= 0 or valid_duration / window_seconds < minimum_coverage:
        return float("nan")
    return float(numerator / valid_duration)


def _weighted_quadratic_mean(
    rows: Sequence[Mapping[str, object]],
    key: str,
    minimum_coverage: float,
    window_seconds: float,
) -> float:
    numerator = 0.0
    valid_duration = 0.0
    for row in rows:
        value = _number(row.get(key))
        duration = _duration(row)
        if np.isfinite(value) and duration > 0:
            numerator += value * value * duration
            valid_duration += duration
    if valid_duration <= 0 or valid_duration / window_seconds < minimum_coverage:
        return float("nan")
    return float(np.sqrt(numerator / valid_duration))


def _complete_series(
    rows: Sequence[Mapping[str, object]],
    key: str,
    scale: float = 1.0,
) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for row in rows:
        if not _is_complete(row):
            continue
        second = int(row["second_id"])
        value = _number(row.get(key))
        if np.isfinite(value):
            result[second] = float(value / scale)
    return result


def _population_std(series: Mapping[int, float], minimum_count: int) -> float:
    values = np.asarray(list(series.values()), dtype=np.float64)
    if values.size < minimum_count:
        return float("nan")
    return float(np.std(values, ddof=0))


def _three_second_changes(
    series: Mapping[int, float],
    first_end_second: int,
    last_end_second: int,
) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for second in range(first_end_second, last_end_second + 1):
        required = range(second - 5, second + 1)
        if not all(item in series for item in required):
            continue
        recent = series[second] + series[second - 1] + series[second - 2]
        previous = series[second - 3] + series[second - 4] + series[second - 5]
        result[second] = float((recent - previous) / 3.0)
    return result


def _rise_and_drop(
    changes: Mapping[int, float],
) -> Tuple[float, Optional[int], float, Optional[int]]:
    if not changes:
        missing = float("nan")
        return missing, None, missing, None
    ordered = sorted(changes.items())
    maximum = max(value for _, value in ordered)
    minimum = min(value for _, value in ordered)
    rise_second = next(second for second, value in ordered if value == maximum)
    drop_second = next(second for second, value in ordered if value == minimum)
    if maximum <= 0:
        rise_value, rise_second = 0.0, None
    else:
        rise_value = float(maximum)
    if minimum >= 0:
        drop_value, drop_second = 0.0, None
    else:
        drop_value = float(-minimum)
    return rise_value, rise_second, drop_value, drop_second


def _adjacent_absolute_changes(
    series: Mapping[int, float],
    minute_start: int,
) -> List[float]:
    result: List[float] = []
    for second in range(minute_start, minute_start + 60):
        if second in series and second - 1 in series:
            result.append(abs(series[second] - series[second - 1]))
    return result


def _average_absolute_change(
    series: Mapping[int, float],
    minute_start: int,
    minimum_pairs: int,
) -> float:
    changes = _adjacent_absolute_changes(series, minute_start)
    if len(changes) < minimum_pairs:
        return float("nan")
    return float(np.mean(changes))


def _common_phase_imbalance(
    phase_series: Sequence[Mapping[int, float]],
    minute_start: int,
    config: MinuteCalculationConfig,
) -> float:
    pair_ends = [
        second
        for second in range(minute_start, minute_start + 60)
        if all(second in series and second - 1 in series for series in phase_series)
    ]
    if len(pair_ends) < config.min_valid_adjacent_pairs:
        return float("nan")
    phase_changes = np.asarray(
        [
            np.mean([abs(series[second] - series[second - 1]) for second in pair_ends])
            for series in phase_series
        ],
        dtype=np.float64,
    )
    total = float(np.sum(phase_changes))
    if total < config.imbalance_stable_threshold:
        return 0.0
    return float(
        (np.max(phase_changes) - np.min(phase_changes))
        / (total + config.imbalance_epsilon)
    )


def _event_count(
    changes: Mapping[int, float],
    minute_start: int,
    threshold: float,
) -> int:
    """Merge continuous, same-direction threshold positions into one event."""

    previous_second = minute_start - 1
    previous_value = changes.get(previous_second)
    previous_direction = (
        int(np.sign(previous_value))
        if previous_value is not None and abs(previous_value) > threshold
        else 0
    )
    count = 0
    for second in range(minute_start, minute_start + 60):
        value = changes.get(second)
        direction = (
            int(np.sign(value))
            if value is not None and abs(value) > threshold
            else 0
        )
        if direction != 0 and not (
            previous_direction == direction and previous_second == second - 1
        ):
            count += 1
        previous_direction = direction
        previous_second = second
    return count


def _trend_slope(series: Mapping[int, float], minimum_count: int) -> float:
    if len(series) < max(2, minimum_count):
        return float("nan")
    ordered = sorted(series.items())
    x = np.asarray([second for second, _ in ordered], dtype=np.float64)
    y = np.asarray([value for _, value in ordered], dtype=np.float64)
    x -= x[0]
    centered_x = x - np.mean(x)
    denominator = float(np.sum(centered_x * centered_x))
    if denominator <= 0:
        return 0.0
    return float(np.sum(centered_x * (y - np.mean(y))) / denominator)


def _fixed_last_five_mean(
    rows: Sequence[Mapping[str, object]],
    key: str,
    scale: float,
    minute_start: int,
    minimum_coverage: float,
) -> float:
    fixed_rows = [
        row
        for row in rows
        if minute_start + 55 <= int(row["second_id"]) < minute_start + 60
    ]
    return _weighted_mean(fixed_rows, key, scale, minimum_coverage, 5.0)


def _arithmetic_power_factor(
    rows: Sequence[Mapping[str, object]],
    minimum_coverage: float,
) -> float:
    valid_rows: List[Tuple[Mapping[str, object], float]] = []
    required = ["TotW"] + [f"PhV_phs{phase}" for phase in "ABC"] + [
        f"A_phs{phase}" for phase in "ABC"
    ]
    for row in rows:
        duration = _duration(row)
        if duration <= 0:
            continue
        if all(np.isfinite(_number(row.get(key))) for key in required):
            valid_rows.append((row, duration))
    total_duration = sum(duration for _, duration in valid_rows)
    if total_duration <= 0 or total_duration / 60.0 < minimum_coverage:
        return float("nan")
    total_active = sum(_number(row["TotW"]) * duration for row, duration in valid_rows)
    total_active /= total_duration
    apparent = 0.0
    for phase in "ABC":
        voltage = np.sqrt(
            sum(_number(row[f"PhV_phs{phase}"]) ** 2 * duration for row, duration in valid_rows)
            / total_duration
        )
        current = np.sqrt(
            sum(_number(row[f"A_phs{phase}"]) ** 2 * duration for row, duration in valid_rows)
            / total_duration
        )
        apparent += voltage * current
    if apparent <= 1e-15:
        return float("nan")
    return float(np.clip(total_active / apparent, -1.0, 1.0))


def _current_harmonic_ratio(
    rows: Sequence[Mapping[str, object]],
    config: MinuteCalculationConfig,
) -> float:
    ratios: List[float] = []
    for row in rows:
        if not _is_complete(row):
            continue
        fundamental = np.asarray(
            [_number(row.get(f"HA1_phs{phase}")) for phase in "ABC"],
            dtype=np.float64,
        )
        higher = np.asarray(
            [
                _number(row.get(f"HA{order}_phs{phase}"))
                for phase in "ABC"
                for order in range(2, 26)
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(fundamental)) or not np.all(np.isfinite(higher)):
            continue
        denominator = float(np.sqrt(np.sum(fundamental * fundamental)))
        if denominator < config.fundamental_current_min_a:
            continue
        ratios.append(float(np.sqrt(np.sum(higher * higher)) / denominator))
    if len(ratios) < config.min_harmonic_windows:
        return float("nan")
    return float(np.median(np.asarray(ratios, dtype=np.float64)))


def calculate_minute_items(
    rows: Sequence[Dict[str, object]],
    config: MinuteCalculationConfig = MinuteCalculationConfig(),
    history_rows: Sequence[Dict[str, object]] = (),
) -> Dict[str, float]:
    """Calculate one natural minute from current rows and six history rows."""

    if not rows:
        return {field: float("nan") for field in CALCULATED_FIELDS}
    current_rows = sorted(rows, key=lambda row: int(row["second_id"]))
    minute_start = int(current_rows[0]["second_id"]) // 60 * 60
    if any(int(row["second_id"]) // 60 * 60 != minute_start for row in current_rows):
        raise ValueError("calculate_minute_items 的 rows 必须全部属于同一自然分钟")

    merged_by_second: Dict[int, Dict[str, object]] = {
        int(row["second_id"]): row for row in history_rows
    }
    merged_by_second.update({int(row["second_id"]): row for row in current_rows})
    merged_rows = [merged_by_second[key] for key in sorted(merged_by_second)]

    result: Dict[str, float] = {}
    mean_fields = [
        ("TotW_MA", "TotW", 1000.0),
        ("PhW_phsA_MA", "PhW_phsA", 1000.0),
        ("PhW_phsB_MA", "PhW_phsB", 1000.0),
        ("PhW_phsC_MA", "PhW_phsC", 1000.0),
        ("TotVar_MA", "TotVar", 1000.0),
        ("PhVar_phsA_MA", "PhVar_phsA", 1000.0),
        ("PhVar_phsB_MA", "PhVar_phsB", 1000.0),
        ("PhVar_phsC_MA", "PhVar_phsC", 1000.0),
    ]
    for output, source, scale in mean_fields:
        result[output] = _weighted_mean(
            current_rows, source, scale, config.minute_min_coverage, 60.0
        )

    for phase in "ABC":
        result[f"PhV_phs{phase}"] = _weighted_quadratic_mean(
            current_rows, f"PhV_phs{phase}", config.minute_min_coverage, 60.0
        )
        result[f"A_phs{phase}"] = _weighted_quadratic_mean(
            current_rows, f"A_phs{phase}", config.minute_min_coverage, 60.0
        )
    result["TotPF_AA"] = _arithmetic_power_factor(
        current_rows, config.minute_min_coverage
    )

    total_p = _complete_series(merged_rows, "TotW", 1000.0)
    total_q = _complete_series(merged_rows, "TotVar", 1000.0)
    phase_p = [_complete_series(merged_rows, f"PhW_phs{p}", 1000.0) for p in "ABC"]
    phase_q = [_complete_series(merged_rows, f"PhVar_phs{p}", 1000.0) for p in "ABC"]

    def current_only(series: Mapping[int, float]) -> Dict[int, float]:
        return {
            second: value
            for second, value in series.items()
            if minute_start <= second < minute_start + 60
        }

    result["TotW_PSD"] = _population_std(current_only(total_p), config.min_valid_seconds)
    result["TotVar_PSD"] = _population_std(current_only(total_q), config.min_valid_seconds)
    for index, phase in enumerate("ABC"):
        result[f"PhW_phs{phase}_PSD"] = _population_std(
            current_only(phase_p[index]), config.min_valid_seconds
        )
        result[f"PhVar_phs{phase}_PSD"] = _population_std(
            current_only(phase_q[index]), config.min_valid_seconds
        )

    total_p_changes = _three_second_changes(total_p, minute_start - 1, minute_start + 59)
    total_q_changes = _three_second_changes(total_q, minute_start - 1, minute_start + 59)
    current_p_changes = {
        second: value for second, value in total_p_changes.items()
        if minute_start <= second < minute_start + 60
    }
    rise, rise_second, drop, drop_second = _rise_and_drop(current_p_changes)
    result["TotW_MCU"] = rise
    result["TotW_MCD"] = drop
    if not current_p_changes:
        result["TotVar_MCU"] = float("nan")
        result["TotVar_MCD"] = float("nan")
    else:
        result["TotVar_MCU"] = 0.0 if rise_second is None else float(
            total_q_changes.get(rise_second, float("nan"))
        )
        result["TotVar_MCD"] = 0.0 if drop_second is None else float(
            total_q_changes.get(drop_second, float("nan"))
        )

    for index, phase in enumerate("ABC"):
        changes = _three_second_changes(phase_p[index], minute_start, minute_start + 59)
        phase_rise, _, phase_drop, _ = _rise_and_drop(changes)
        result[f"PhW_phs{phase}_MCU"] = phase_rise
        result[f"PhW_phs{phase}_MCD"] = phase_drop

    result["TotW_MEA"] = _fixed_last_five_mean(
        current_rows, "TotW", 1000.0, minute_start, config.last5_min_coverage
    )
    result["TotVar_MEA"] = _fixed_last_five_mean(
        current_rows, "TotVar", 1000.0, minute_start, config.last5_min_coverage
    )
    for phase in "ABC":
        result[f"PhW_phs{phase}_MEA"] = _fixed_last_five_mean(
            current_rows, f"PhW_phs{phase}", 1000.0,
            minute_start, config.last5_min_coverage
        )
        result[f"PhVar_phs{phase}_MEA"] = _fixed_last_five_mean(
            current_rows, f"PhVar_phs{phase}", 1000.0,
            minute_start, config.last5_min_coverage
        )

    result["A_SCC"] = _current_harmonic_ratio(current_rows, config)
    result["TotW_AAC"] = _average_absolute_change(
        total_p, minute_start, config.min_valid_adjacent_pairs
    )
    result["TotVar_AAC"] = _average_absolute_change(
        total_q, minute_start, config.min_valid_adjacent_pairs
    )
    for index, phase in enumerate("ABC"):
        result[f"PhW_phs{phase}_AAC"] = _average_absolute_change(
            phase_p[index], minute_start, config.min_valid_adjacent_pairs
        )
        result[f"PhVar_phs{phase}_AAC"] = _average_absolute_change(
            phase_q[index], minute_start, config.min_valid_adjacent_pairs
        )

    result["ImbNg_TotW"] = _common_phase_imbalance(phase_p, minute_start, config)
    result["ImbNg_TotVar"] = _common_phase_imbalance(phase_q, minute_start, config)
    result["TotW_MEC"] = float(
        _event_count(total_p_changes, minute_start, config.event_threshold_kw)
    )
    result["TotVar_MEC"] = float(
        _event_count(total_q_changes, minute_start, config.event_threshold_kvar)
    )
    result["TotW_MTS"] = _trend_slope(current_only(total_p), config.min_valid_seconds)
    result["TotVar_MTS"] = _trend_slope(current_only(total_q), config.min_valid_seconds)

    return {field: float(result.get(field, float("nan"))) for field in CALCULATED_FIELDS}
