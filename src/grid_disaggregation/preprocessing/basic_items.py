#!/usr/bin/env python3
"""Calculate the 44 basic acquisition fields listed in the reference Excel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .harmonic_items import HarmonicResult


PHASES = "ABC"
BASIC_FIELDS = [
    "PhV_phsA", "PhV_phsB", "PhV_phsC",
    "A_phsA", "A_phsB", "A_phsC",
    "Hz", "Hz_phsA", "Hz_phsB", "Hz_phsC",
    "TotHzOfs", "HzOfs_phsA", "HzOfs_phsB", "HzOfs_phsC",
    "TotPF", "PhPF_phsA", "PhPF_phsB", "PhPF_phsC",
    "TotVA", "PhVA_phsA", "PhVA_phsB", "PhVA_phsC",
    "TotVar", "PhVar_phsA", "PhVar_phsB", "PhVar_phsC",
    "TotW", "PhW_phsA", "PhW_phsB", "PhW_phsC",
    "PhVOfs_phsA", "PhVOfs_phsB", "PhVOfs_phsC",
    "ImbNgA", "ImbNgV", "LoadRate", "SeqA_c0", "SeqV_c0",
    "ThdA_phsA", "ThdA_phsB", "ThdA_phsC",
    "ThdPhV_phsA", "ThdPhV_phsB", "ThdPhV_phsC",
]


@dataclass(frozen=True)
class BasicResult:
    fields: Dict[str, float]
    internal: Dict[str, float]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-15:
        return float("nan")
    return float(numerator / denominator)


def _finite_sum_or_nan(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.sum(finite)) if finite.size else float("nan")


def _sequence_components(
    phasors: np.ndarray,
    phase_sequence: str,
) -> Tuple[complex, complex, complex]:
    if phasors.shape != (3,) or not np.all(np.isfinite(phasors)):
        missing = complex(float("nan"), float("nan"))
        return missing, missing, missing
    alpha = np.exp(1j * 2.0 * np.pi / 3.0)
    phase_a, phase_b, phase_c = phasors
    zero = (phase_a + phase_b + phase_c) / 3.0
    abc_component = (phase_a + alpha * phase_b + alpha**2 * phase_c) / 3.0
    acb_component = (phase_a + alpha**2 * phase_b + alpha * phase_c) / 3.0
    if phase_sequence == "ABC":
        positive, negative = abc_component, acb_component
    elif phase_sequence == "ACB":
        positive, negative = acb_component, abc_component
    elif phase_sequence == "auto":
        if abs(abc_component) >= abs(acb_component):
            positive, negative = abc_component, acb_component
        else:
            positive, negative = acb_component, abc_component
    else:
        raise ValueError("phase_sequence 只能是 auto、ABC 或 ACB")
    return zero, positive, negative


def _detect_phase_sequence(phasors: np.ndarray) -> str:
    if phasors.shape != (3,) or not np.all(np.isfinite(phasors)):
        return "ABC"
    alpha = np.exp(1j * 2.0 * np.pi / 3.0)
    phase_a, phase_b, phase_c = phasors
    abc_component = (phase_a + alpha * phase_b + alpha**2 * phase_c) / 3.0
    acb_component = (phase_a + alpha**2 * phase_b + alpha * phase_c) / 3.0
    return "ABC" if abs(abc_component) >= abs(acb_component) else "ACB"


def _thd_percent(harmonic_rms: np.ndarray) -> np.ndarray:
    result = np.full(3, np.nan)
    for phase in range(3):
        fundamental = harmonic_rms[phase, 0]
        if np.isfinite(fundamental) and fundamental > 1e-15:
            higher = harmonic_rms[phase, 1:]
            result[phase] = 100.0 * np.sqrt(np.nansum(higher * higher)) / fundamental
    return result


def calculate_basic_items(
    waveform: np.ndarray,
    harmonic: HarmonicResult,
    nominal_frequency: float = 50.0,
    nominal_voltage: float = 220.0,
    rated_capacity_kva: Optional[float] = None,
    power_sign: int = 1,
    phase_sequence: str = "auto",
) -> BasicResult:
    """Return all official basic fields plus internal minute-calculation drivers.

    Power values are W/var/VA, harmonic distortion and imbalance are percentages,
    and frequency deviation is in Hz.
    """

    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.ndim != 2 or waveform.shape[1] != 6 or waveform.shape[0] == 0:
        raise ValueError("waveform 必须是非空 N×6 数组")
    if power_sign not in (-1, 1):
        raise ValueError("power_sign 只能为 -1 或 1")

    voltage = waveform[:, :3]
    current = waveform[:, 3:]
    voltage_rms = np.sqrt(np.mean(voltage * voltage, axis=0))
    current_rms = np.sqrt(np.mean(current * current, axis=0))
    active_power = power_sign * np.mean(voltage * current, axis=0)
    apparent_power = voltage_rms * current_rms
    phase_pf = np.divide(
        active_power,
        apparent_power,
        out=np.full(3, np.nan),
        where=apparent_power > 1e-15,
    )
    phase_pf = np.clip(phase_pf, -1.0, 1.0)

    harmonic_complex_power = (
        power_sign
        * harmonic.voltage_phasors
        * np.conjugate(harmonic.current_phasors)
    )
    fundamental_complex_power = harmonic_complex_power[:, 0]
    harmonic_reactive = np.imag(harmonic_complex_power)
    reactive_power = np.asarray(
        [
            _finite_sum_or_nan(harmonic_reactive[phase, :])
            for phase in range(3)
        ],
        dtype=np.float64,
    )
    total_active = float(np.sum(active_power))
    total_reactive = _finite_sum_or_nan(reactive_power)
    total_apparent = float(np.sum(apparent_power))
    total_pf = _safe_ratio(total_active, total_apparent)

    resolved_phase_sequence = (
        _detect_phase_sequence(harmonic.voltage_fundamental)
        if phase_sequence == "auto"
        else phase_sequence
    )
    voltage_zero, voltage_positive, voltage_negative = _sequence_components(
        harmonic.voltage_fundamental,
        resolved_phase_sequence,
    )
    current_zero, current_positive, current_negative = _sequence_components(
        harmonic.current_fundamental,
        resolved_phase_sequence,
    )
    voltage_imbalance = 100.0 * _safe_ratio(abs(voltage_negative), abs(voltage_positive))
    current_imbalance = 100.0 * _safe_ratio(abs(current_negative), abs(current_positive))

    voltage_thd = _thd_percent(harmonic.voltage_rms)
    current_thd = _thd_percent(harmonic.current_rms)
    phase_frequencies = harmonic.phase_frequencies
    network_frequency = harmonic.network_frequency

    fields: Dict[str, float] = {}
    for index, phase in enumerate(PHASES):
        fields[f"PhV_phs{phase}"] = float(voltage_rms[index])
        fields[f"A_phs{phase}"] = float(current_rms[index])
        fields[f"Hz_phs{phase}"] = float(phase_frequencies[index])
        fields[f"HzOfs_phs{phase}"] = float(phase_frequencies[index] - nominal_frequency)
        fields[f"PhPF_phs{phase}"] = float(phase_pf[index])
        fields[f"PhVA_phs{phase}"] = float(apparent_power[index])
        fields[f"PhVar_phs{phase}"] = float(reactive_power[index])
        fields[f"PhW_phs{phase}"] = float(active_power[index])
        fields[f"PhVOfs_phs{phase}"] = (
            100.0 * (float(voltage_rms[index]) - nominal_voltage) / nominal_voltage
            if nominal_voltage > 0
            else float("nan")
        )
        fields[f"ThdA_phs{phase}"] = float(current_thd[index])
        fields[f"ThdPhV_phs{phase}"] = float(voltage_thd[index])

    fields.update(
        Hz=float(network_frequency),
        TotHzOfs=float(network_frequency - nominal_frequency),
        TotPF=float(total_pf),
        TotVA=float(total_apparent),
        TotVar=float(total_reactive),
        TotW=float(total_active),
        ImbNgA=float(current_imbalance),
        ImbNgV=float(voltage_imbalance),
        LoadRate=(
            100.0 * total_apparent / (rated_capacity_kva * 1000.0)
            if rated_capacity_kva is not None and rated_capacity_kva > 0
            else float("nan")
        ),
        SeqA_c0=float(abs(current_zero)),
        SeqV_c0=float(abs(voltage_zero)),
    )

    internal = {
        "_phase_sequence_acb": 1.0 if resolved_phase_sequence == "ACB" else 0.0,
    }
    return BasicResult(fields=fields, internal=internal)
