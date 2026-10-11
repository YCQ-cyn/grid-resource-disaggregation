#!/usr/bin/env python3
"""One-second frequency and tracked 1st-25th harmonic calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


PHASES = "ABC"
HARMONIC_ORDERS = tuple(range(1, 26))
VOLTAGE_HARMONIC_FIELDS = [
    f"HphV{order}_phs{phase}"
    for phase in PHASES
    for order in HARMONIC_ORDERS
]
CURRENT_HARMONIC_FIELDS = [
    f"HA{order}_phs{phase}"
    for phase in PHASES
    for order in HARMONIC_ORDERS
]
HARMONIC_FIELDS = VOLTAGE_HARMONIC_FIELDS + CURRENT_HARMONIC_FIELDS


@dataclass(frozen=True)
class HarmonicResult:
    fields: Dict[str, float]
    voltage_rms: np.ndarray  # shape (3, 25)
    current_rms: np.ndarray  # shape (3, 25)
    voltage_phasors: np.ndarray  # complex RMS phasors, shape (3, 25)
    current_phasors: np.ndarray  # complex RMS phasors, shape (3, 25)
    voltage_fundamental: np.ndarray  # complex RMS phasors, shape (3,)
    current_fundamental: np.ndarray  # complex RMS phasors, shape (3,)
    phase_frequencies: np.ndarray  # Hz, shape (3,)
    network_frequency: float


def _positive_zero_crossing_frequency(
    signal: np.ndarray,
    sample_rate: float,
    nominal_frequency: float,
) -> float:
    centered = np.asarray(signal, dtype=np.float64) - float(np.mean(signal))
    if centered.size < max(16, int(sample_rate / nominal_frequency * 3)):
        return float("nan")
    left = centered[:-1]
    right = centered[1:]
    indexes = np.flatnonzero((left <= 0.0) & (right > 0.0))
    if indexes.size < 3:
        return float("nan")
    denominators = right[indexes] - left[indexes]
    fractions = np.divide(
        -left[indexes],
        denominators,
        out=np.zeros_like(denominators),
        where=np.abs(denominators) > 1e-15,
    )
    crossings = indexes.astype(np.float64) + fractions
    value = (crossings.size - 1) * sample_rate / (crossings[-1] - crossings[0])
    if not (0.8 * nominal_frequency <= value <= 1.2 * nominal_frequency):
        return float("nan")
    return float(value)


def _empty_result() -> HarmonicResult:
    nan_fields = {field: float("nan") for field in HARMONIC_FIELDS}
    return HarmonicResult(
        fields=nan_fields,
        voltage_rms=np.full((3, 25), np.nan),
        current_rms=np.full((3, 25), np.nan),
        voltage_phasors=np.full((3, 25), np.nan + 1j * np.nan, dtype=np.complex128),
        current_phasors=np.full((3, 25), np.nan + 1j * np.nan, dtype=np.complex128),
        voltage_fundamental=np.full(3, np.nan + 1j * np.nan, dtype=np.complex128),
        current_fundamental=np.full(3, np.nan + 1j * np.nan, dtype=np.complex128),
        phase_frequencies=np.full(3, np.nan),
        network_frequency=float("nan"),
    )


def analyze_harmonics(
    waveform: np.ndarray,
    sample_rate: float,
    nominal_frequency: float = 50.0,
    minimum_cycles: int = 5,
) -> HarmonicResult:
    """Calculate harmonic RMS phasors at multiples of the tracked frequency.

    ``waveform`` columns must be Ua, Ub, Uc, Ia, Ib, Ic. A minimum of five
    fundamental cycles is required; shorter partial seconds return blank values.
    """

    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.ndim != 2 or waveform.shape[1] != 6:
        raise ValueError("waveform 必须是 N×6，列顺序为 Ua,Ub,Uc,Ia,Ib,Ic")
    sample_count = waveform.shape[0]
    minimum_samples = int(round(minimum_cycles * sample_rate / nominal_frequency))
    if sample_count < minimum_samples:
        return _empty_result()

    phase_frequencies = np.asarray(
        [
            _positive_zero_crossing_frequency(waveform[:, phase], sample_rate, nominal_frequency)
            for phase in range(3)
        ],
        dtype=np.float64,
    )
    finite_frequencies = phase_frequencies[np.isfinite(phase_frequencies)]
    network_frequency = (
        float(np.mean(finite_frequencies))
        if finite_frequencies.size
        else float(nominal_frequency)
    )
    phase_frequencies = np.where(
        np.isfinite(phase_frequencies),
        phase_frequencies,
        network_frequency,
    )

    centered = waveform - np.mean(waveform, axis=0, keepdims=True)
    window = np.hanning(sample_count).astype(np.float64)
    coherent_gain = float(np.sum(window))
    if coherent_gain <= 0:
        return _empty_result()
    # Direct Hann-window projections avoid rounding h*f1 to an FFT bin.  This
    # follows the Markdown requirement to locate harmonics from the tracked
    # actual fundamental rather than fixed 50.000 Hz multiples.
    sample_indexes = np.arange(sample_count, dtype=np.float64)
    orders = np.asarray(HARMONIC_ORDERS, dtype=np.float64)
    frequencies = orders * network_frequency
    valid_orders = frequencies < sample_rate / 2.0
    phasors_by_order = np.full(
        (len(HARMONIC_ORDERS), 6),
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    if np.any(valid_orders):
        kernels = np.exp(
            -2j
            * np.pi
            * frequencies[valid_orders, None]
            * sample_indexes[None, :]
            / sample_rate
        )
        phasors_by_order[valid_orders, :] = (
            np.sqrt(2.0)
            * (kernels @ (centered * window[:, None]))
            / coherent_gain
        )

    voltage_phasors = np.asarray(phasors_by_order[:, :3].T, dtype=np.complex128)
    current_phasors = np.asarray(phasors_by_order[:, 3:].T, dtype=np.complex128)
    voltage_rms = np.abs(voltage_phasors)
    current_rms = np.abs(current_phasors)

    fields: Dict[str, float] = {}
    for phase_index, phase in enumerate(PHASES):
        for order_index, order in enumerate(HARMONIC_ORDERS):
            fields[f"HphV{order}_phs{phase}"] = float(voltage_rms[phase_index, order_index])
            fields[f"HA{order}_phs{phase}"] = float(current_rms[phase_index, order_index])

    return HarmonicResult(
        fields=fields,
        voltage_rms=voltage_rms,
        current_rms=current_rms,
        voltage_phasors=voltage_phasors,
        current_phasors=current_phasors,
        voltage_fundamental=np.asarray(voltage_phasors[:, 0], dtype=np.complex128),
        current_fundamental=np.asarray(current_phasors[:, 0], dtype=np.complex128),
        phase_frequencies=phase_frequencies,
        network_frequency=network_frequency,
    )
