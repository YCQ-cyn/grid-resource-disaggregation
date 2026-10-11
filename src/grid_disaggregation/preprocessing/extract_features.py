#!/usr/bin/env python3
"""Generate the Excel-defined basic, harmonic and calculated data items.

Outputs:
  - 基础项_1秒.csv: 44 official basic fields
  - 谐波项_1秒.csv: 150 official 1st-25th harmonic fields
  - 计算项_1分钟.csv: 56 required minute fields
  - 处理日志.csv: per-recording diagnostics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np

from .basic_items import BASIC_FIELDS, calculate_basic_items
from .calculated_items import (
    CALCULATED_FIELDS,
    MinuteCalculationConfig,
    calculate_minute_items,
)
from .comtrade_io import EPOCH, discover_recordings, open_recording
from .harmonic_items import HARMONIC_FIELDS, analyze_harmonics


SECOND_METADATA_FIELDS = [
    "second_start",
    "sample_count",
    "expected_samples",
    "coverage_ratio",
    "is_complete",
    "source_recordings",
]
MINUTE_METADATA_FIELDS = [
    "minute_start",
    "second_count",
    "complete_second_count",
    "waveform_sample_count",
    "expected_samples",
    "coverage_ratio",
    "is_complete",
    "source_recordings",
]
LOG_FIELDS = [
    "source",
    "kind",
    "source_size_mib",
    "status",
    "message",
    "cfg",
    "dat",
    "sample_rate_hz",
    "record_count",
    "kept_samples",
    "overlap_dropped",
    "trailing_bytes",
    "timestamp_mode",
]
FORMULA_ID = "minute_electrical_features"


class ProcessingLimitReached(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 COMTRADE CFG/DAT、ZIP 或 7z 生成 Excel 中定义的数据项"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="输入目录、单个 ZIP/7z，或已解压 CFG/DAT 路径",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="三个结果 CSV 和处理日志的输出目录",
    )
    parser.add_argument("--nominal-voltage", type=float, default=220.0, help="相电压标称值，默认 220 V")
    parser.add_argument("--nominal-frequency", type=float, default=50.0, help="标称频率，默认 50 Hz")
    parser.add_argument(
        "--rated-capacity-kva",
        type=float,
        default=None,
        help="配变额定容量；不提供时 LoadRate 留空",
    )
    parser.add_argument(
        "--power-sign",
        type=int,
        choices=(-1, 1),
        default=1,
        help="确认电流方向后选择 1 或 -1，默认 1",
    )
    parser.add_argument(
        "--phase-sequence",
        choices=("auto", "ABC", "ACB"),
        default="auto",
        help="对称分量相序；默认 auto 自动选择主序，样例数据识别为 ACB",
    )
    parser.add_argument(
        "--time-source",
        choices=("auto", "cfg", "filename"),
        default="auto",
        help="绝对起始时间来源，默认 auto",
    )
    parser.add_argument(
        "--relative-time-source",
        choices=("sample_index", "auto", "dat_timestamp"),
        default="sample_index",
        help="DAT 文件内时间轴，当前设备建议 sample_index",
    )
    parser.add_argument("--chunk-records", type=int, default=256_000, help="每批读取记录数")
    parser.add_argument(
        "--min-leading-mb",
        type=float,
        default=30.0,
        help="只在首个正常压缩包之前跳过小于该大小的开头包；设0禁用",
    )
    parser.add_argument(
        "--complete-threshold",
        type=float,
        default=0.999,
        help="秒/分钟完整率阈值，默认 0.999",
    )
    parser.add_argument(
        "--minute-min-coverage",
        type=float,
        default=None,
        help="分钟平均/RMS/PF 最低有效覆盖率；默认沿用 --complete-threshold",
    )
    parser.add_argument(
        "--second-min-coverage",
        type=float,
        default=None,
        help="动态计算所用自然秒最低覆盖率；默认沿用 --complete-threshold",
    )
    parser.add_argument(
        "--last5-min-coverage",
        type=float,
        default=None,
        help="固定第55～59秒窗口最低覆盖率；默认沿用 --complete-threshold",
    )
    parser.add_argument("--min-valid-seconds", type=int, default=1, help="标准差/趋势最少有效秒数")
    parser.add_argument(
        "--min-valid-adjacent-pairs",
        type=int,
        default=1,
        help="AAC 和变化不均衡度最少连续相邻秒对数",
    )
    parser.add_argument(
        "--min-harmonic-windows",
        type=int,
        default=1,
        help="A_SCC 分钟中位数最少有效谐波短窗数",
    )
    parser.add_argument(
        "--fundamental-current-min-a",
        type=float,
        default=1e-6,
        help="A_SCC 基波三相合成电流最低有效阈值，A",
    )
    parser.add_argument(
        "--event-threshold-kw", "--event-min-kw",
        dest="event_threshold_kw", type=float, default=0.0,
        help="显著有功事件固定阈值，kW（旧参数 --event-min-kw 仍可用）",
    )
    parser.add_argument(
        "--event-threshold-kvar", "--event-min-kvar",
        dest="event_threshold_kvar", type=float, default=0.0,
        help="显著无功事件固定阈值，kvar（旧参数 --event-min-kvar 仍可用）",
    )
    parser.add_argument(
        "--imbalance-epsilon", type=float, default=1e-12,
        help="三相变化不均衡度分母保护量",
    )
    parser.add_argument(
        "--imbalance-stable-threshold", type=float, default=1e-9,
        help="三相平均绝对变化量之和低于此值时不均衡度取0",
    )
    parser.add_argument("--keep-temp", action="store_true", help="保留压缩包临时解压目录")
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=None,
        help="仅用于快速验证：最多输出多少个秒记录",
    )
    return parser.parse_args()


def epoch_second_text(second_id: int) -> str:
    return (EPOCH + timedelta(seconds=int(second_id))).strftime("%Y-%m-%d %H:%M:%S")


def minute_text(minute_id: int) -> str:
    return epoch_second_text(minute_id * 60)


def csv_value(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return ""
        return f"{numeric:.12g}"
    if isinstance(value, (int, np.integer)):
        return int(value)
    if value is None:
        return ""
    return value


def clean_row(row: Dict[str, object], fields: Sequence[str]) -> Dict[str, object]:
    return {field: csv_value(row.get(field, "")) for field in fields}


@dataclass
class SecondAssembler:
    emit: Callable[[int, np.ndarray, Set[str]], None]
    current_second: Optional[int] = None
    waveform_parts: List[np.ndarray] = field(default_factory=list)
    sources: Set[str] = field(default_factory=set)

    def add(self, second_ids: np.ndarray, waveform: np.ndarray, source_name: str) -> None:
        if second_ids.size == 0:
            return
        split_points = np.flatnonzero(np.diff(second_ids) != 0) + 1
        boundaries = np.concatenate(([0], split_points, [second_ids.size]))
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            second_id = int(second_ids[left])
            if self.current_second is None:
                self.current_second = second_id
            elif second_id != self.current_second:
                if second_id < self.current_second:
                    raise ValueError("输入采样时间未按递增顺序排列")
                self.flush()
                self.current_second = second_id
            self.waveform_parts.append(np.asarray(waveform[left:right], dtype=np.float64))
            self.sources.add(source_name)

    def flush(self) -> None:
        if self.current_second is None or not self.waveform_parts:
            return
        waveform = (
            self.waveform_parts[0]
            if len(self.waveform_parts) == 1
            else np.concatenate(self.waveform_parts, axis=0)
        )
        current = self.current_second
        sources = set(self.sources)
        self.current_second = None
        self.waveform_parts = []
        self.sources = set()
        self.emit(current, waveform, sources)


def run(args: argparse.Namespace) -> int:
    if args.chunk_records <= 0:
        raise ValueError("--chunk-records 必须大于 0")
    if not (0 < args.complete_threshold <= 1.1):
        raise ValueError("--complete-threshold 必须大于 0 且不应超过 1.1")
    if args.nominal_frequency <= 0 or args.nominal_voltage <= 0:
        raise ValueError("标称频率和电压必须大于 0")
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise ValueError("--max-seconds 必须大于 0")
    if args.min_leading_mb < 0:
        raise ValueError("--min-leading-mb 不能为负数")
    coverage_values = [
        value
        for value in (
            args.minute_min_coverage,
            args.second_min_coverage,
            args.last5_min_coverage,
        )
        if value is not None
    ]
    if any(value <= 0 or value > 1.1 for value in coverage_values):
        raise ValueError("各覆盖率阈值必须大于0且不应超过1.1")
    if args.min_valid_seconds <= 0 or args.min_valid_adjacent_pairs <= 0:
        raise ValueError("最少有效秒数和最少相邻秒对数必须大于0")
    if args.min_harmonic_windows <= 0:
        raise ValueError("--min-harmonic-windows 必须大于0")
    if args.fundamental_current_min_a < 0:
        raise ValueError("--fundamental-current-min-a 不能为负数")
    if args.event_threshold_kw < 0 or args.event_threshold_kvar < 0:
        raise ValueError("事件阈值不能为负数")
    if args.imbalance_epsilon < 0 or args.imbalance_stable_threshold < 0:
        raise ValueError("不均衡度保护量和稳定阈值不能为负数")

    minute_min_coverage = (
        args.complete_threshold
        if args.minute_min_coverage is None
        else args.minute_min_coverage
    )
    second_min_coverage = (
        args.complete_threshold
        if args.second_min_coverage is None
        else args.second_min_coverage
    )
    last5_min_coverage = (
        args.complete_threshold
        if args.last5_min_coverage is None
        else args.last5_min_coverage
    )
    if args.event_threshold_kw == 0.0 or args.event_threshold_kvar == 0.0:
        print(
            "警告：事件阈值仍为0；正式生产请用 --event-threshold-kw/"
            "--event-threshold-kvar 传入项目已冻结阈值。",
            file=sys.stderr,
            flush=True,
        )

    sources = discover_recordings(args.input)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    basic_path = output_dir / "基础项_1秒.csv"
    harmonic_path = output_dir / "谐波项_1秒.csv"
    calculated_path = output_dir / "计算项_1分钟.csv"
    log_path = output_dir / "处理日志.csv"
    config_path = output_dir / "处理配置.json"

    basic_fields = SECOND_METADATA_FIELDS + BASIC_FIELDS
    harmonic_fields = SECOND_METADATA_FIELDS + HARMONIC_FIELDS
    minute_fields = MINUTE_METADATA_FIELDS + CALCULATED_FIELDS
    minute_config = MinuteCalculationConfig(
        minute_min_coverage=minute_min_coverage,
        last5_min_coverage=last5_min_coverage,
        min_valid_seconds=args.min_valid_seconds,
        min_valid_adjacent_pairs=args.min_valid_adjacent_pairs,
        min_harmonic_windows=args.min_harmonic_windows,
        fundamental_current_min_a=args.fundamental_current_min_a,
        event_threshold_kw=args.event_threshold_kw,
        event_threshold_kvar=args.event_threshold_kvar,
        imbalance_epsilon=args.imbalance_epsilon,
        imbalance_stable_threshold=args.imbalance_stable_threshold,
    )

    log_rows: List[Dict[str, object]] = []
    minute_rows: List[Dict[str, object]] = []
    calculation_history: List[Dict[str, object]] = []
    active_minute_id: Optional[int] = None
    seconds_written = 0
    minutes_written = 0
    global_sample_rate: Optional[float] = None
    last_global_us: Optional[int] = None
    normal_archive_started = False

    with (
        basic_path.open("w", newline="", encoding="utf-8-sig") as basic_file,
        harmonic_path.open("w", newline="", encoding="utf-8-sig") as harmonic_file,
        calculated_path.open("w", newline="", encoding="utf-8-sig") as calculated_file,
    ):
        basic_writer = csv.DictWriter(basic_file, fieldnames=basic_fields)
        harmonic_writer = csv.DictWriter(harmonic_file, fieldnames=harmonic_fields)
        calculated_writer = csv.DictWriter(calculated_file, fieldnames=minute_fields)
        basic_writer.writeheader()
        harmonic_writer.writeheader()
        calculated_writer.writeheader()

        def emit_minute() -> None:
            nonlocal minute_rows, calculation_history, active_minute_id, minutes_written
            if active_minute_id is None or not minute_rows or global_sample_rate is None:
                return
            values = calculate_minute_items(
                minute_rows,
                minute_config,
                history_rows=calculation_history,
            )
            sample_count = sum(int(row["sample_count"]) for row in minute_rows)
            expected = int(round(global_sample_rate * 60.0))
            coverage = sample_count / expected if expected else float("nan")
            source_names = sorted(
                {
                    name
                    for row in minute_rows
                    for name in str(row["source_recordings"]).split(";")
                    if name
                }
            )
            output: Dict[str, object] = {
                "minute_start": minute_text(active_minute_id),
                "second_count": len(minute_rows),
                "complete_second_count": sum(
                    int(row["is_complete_second"]) for row in minute_rows
                ),
                "waveform_sample_count": sample_count,
                "expected_samples": expected,
                "coverage_ratio": coverage,
                "is_complete": 1 if coverage >= minute_min_coverage else 0,
                "source_recordings": ";".join(source_names),
            }
            output.update(values)
            calculated_writer.writerow(clean_row(output, minute_fields))
            minutes_written += 1
            combined_history = {
                int(row["second_id"]): row
                for row in calculation_history + minute_rows
            }
            calculation_history = [
                combined_history[second]
                for second in sorted(combined_history)[-6:]
            ]
            minute_rows = []

        def emit_second(second_id: int, waveform: np.ndarray, source_names: Set[str]) -> None:
            nonlocal seconds_written, minute_rows, active_minute_id
            if global_sample_rate is None:
                raise RuntimeError("尚未确定采样率")
            expected = int(round(global_sample_rate))
            sample_count = int(waveform.shape[0])
            coverage = sample_count / expected if expected else float("nan")
            is_complete = 1 if coverage >= second_min_coverage else 0
            harmonic = analyze_harmonics(
                waveform,
                global_sample_rate,
                nominal_frequency=args.nominal_frequency,
            )
            basic = calculate_basic_items(
                waveform,
                harmonic,
                nominal_frequency=args.nominal_frequency,
                nominal_voltage=args.nominal_voltage,
                rated_capacity_kva=args.rated_capacity_kva,
                power_sign=args.power_sign,
                phase_sequence=args.phase_sequence,
            )
            metadata: Dict[str, object] = {
                "second_start": epoch_second_text(second_id),
                "sample_count": sample_count,
                "expected_samples": expected,
                "coverage_ratio": coverage,
                "is_complete": is_complete,
                "source_recordings": ";".join(sorted(source_names)),
            }
            basic_output = dict(metadata)
            basic_output.update(basic.fields)
            harmonic_output = dict(metadata)
            harmonic_output.update(harmonic.fields)
            basic_writer.writerow(clean_row(basic_output, basic_fields))
            harmonic_writer.writerow(clean_row(harmonic_output, harmonic_fields))

            minute_id = second_id // 60
            if active_minute_id is None:
                active_minute_id = minute_id
            elif minute_id != active_minute_id:
                emit_minute()
                active_minute_id = minute_id

            internal_row: Dict[str, object] = {
                "second_id": second_id,
                "sample_count": sample_count,
                "expected_samples": expected,
                "_valid_duration_s": min(1.0, sample_count / global_sample_rate),
                "is_complete_second": is_complete,
                "source_recordings": metadata["source_recordings"],
            }
            internal_row.update(basic.fields)
            internal_row.update(basic.internal)
            internal_row.update(harmonic.fields)
            minute_rows.append(internal_row)
            seconds_written += 1
            if args.max_seconds is not None and seconds_written >= args.max_seconds:
                raise ProcessingLimitReached

        assembler = SecondAssembler(emit=emit_second)
        limit_reached = False

        for source_index, source in enumerate(sources, start=1):
            log_row: Dict[str, object] = {
                "source": source.source_name,
                "kind": source.kind,
                "status": "error",
            }
            source_size_mib = (
                source.path.stat().st_size / (1024.0 * 1024.0)
                if source.path.is_file()
                else float("nan")
            )
            log_row["source_size_mib"] = source_size_mib
            if (
                not normal_archive_started
                and source.kind in ("zip", "7z")
                and args.min_leading_mb > 0
                and source_size_mib < args.min_leading_mb
            ):
                log_row.update(
                    status="skipped_small_leading",
                    message=f"开头压缩包小于 {args.min_leading_mb:.3f} MiB",
                    kept_samples=0,
                    overlap_dropped=0,
                )
                log_rows.append(log_row)
                print(
                    f"[{source_index}/{len(sources)}] 跳过小型开头包 "
                    f"{source.source_name} ({source_size_mib:.1f} MiB)",
                    flush=True,
                )
                continue
            normal_archive_started = True
            kept_samples = 0
            overlap_dropped = 0
            print(f"[{source_index}/{len(sources)}] {source.source_name} ({source.kind})", flush=True)
            try:
                with open_recording(
                    source,
                    time_source=args.time_source,
                    relative_time_source=args.relative_time_source,
                    keep_temp=args.keep_temp,
                ) as recording:
                    if global_sample_rate is None:
                        global_sample_rate = recording.sample_rate
                    elif not math.isclose(
                        global_sample_rate,
                        recording.sample_rate,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    ):
                        raise ValueError(
                            f"采样率从 {global_sample_rate} 变为 {recording.sample_rate}"
                        )

                    log_row.update(
                        cfg=str(recording.cfg.cfg_path),
                        dat=str(recording.dat_path),
                        sample_rate_hz=recording.sample_rate,
                        record_count=recording.record_count,
                        trailing_bytes=recording.trailing_bytes,
                        timestamp_mode=recording.timestamp_mode,
                    )

                    for absolute_us, waveform in recording.iter_chunks(args.chunk_records):
                        if last_global_us is not None:
                            keep = absolute_us > last_global_us
                            overlap_dropped += int(np.count_nonzero(~keep))
                            if not np.any(keep):
                                continue
                            absolute_us = absolute_us[keep]
                            waveform = waveform[keep]
                        kept_samples += int(absolute_us.size)
                        last_global_us = int(absolute_us[-1])
                        second_ids = absolute_us // 1_000_000
                        assembler.add(second_ids, waveform, source.source_name)

                    log_row.update(status="ok", message="")
            except ProcessingLimitReached:
                log_row.update(status="limited", message=f"达到 --max-seconds={args.max_seconds}")
                limit_reached = True
            except Exception as exc:
                log_row.update(status="error", message=str(exc))
                print(f"    ERROR: {exc}", file=sys.stderr, flush=True)
            finally:
                log_row.update(
                    kept_samples=kept_samples,
                    overlap_dropped=overlap_dropped,
                )
                log_rows.append(log_row)
            if limit_reached:
                break

        if not limit_reached:
            assembler.flush()
        emit_minute()

    with log_path.open("w", newline="", encoding="utf-8-sig") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for row in log_rows:
            writer.writerow(clean_row(row, LOG_FIELDS))

    configuration = {
        "formula_version": FORMULA_ID,
        "schema_source": "终端采集APP采集项、数据项.xlsx",
        "calculation_source": "资源辨识与光伏分离分钟级数据接口需求说明(1)/(3).md",
        "calculated_field_count": len(CALCULATED_FIELDS),
        "excluded_excel_tail_fields": ["TotPF_AC", "TotW_TotVA", "A_EC"],
        "reactive_power_method": "sum_h1_to_h25_imaginary_complex_power",
        "nominal_voltage_v": args.nominal_voltage,
        "nominal_frequency_hz": args.nominal_frequency,
        "rated_capacity_kva": args.rated_capacity_kva,
        "power_sign": args.power_sign,
        "phase_sequence": args.phase_sequence,
        "min_leading_mb": args.min_leading_mb,
        "minute_min_coverage": minute_min_coverage,
        "second_min_coverage": second_min_coverage,
        "last5_min_coverage": last5_min_coverage,
        "min_valid_seconds": args.min_valid_seconds,
        "min_valid_adjacent_pairs": args.min_valid_adjacent_pairs,
        "min_harmonic_windows": args.min_harmonic_windows,
        "fundamental_current_min_a": args.fundamental_current_min_a,
        "event_threshold_kw": args.event_threshold_kw,
        "event_threshold_kvar": args.event_threshold_kvar,
        "imbalance_epsilon": args.imbalance_epsilon,
        "imbalance_stable_threshold": args.imbalance_stable_threshold,
        "time_source": args.time_source,
        "relative_time_source": args.relative_time_source,
    }
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(configuration, config_file, ensure_ascii=False, indent=2)
        config_file.write("\n")

    print(f"完成：{seconds_written} 个秒记录，{minutes_written} 个分钟记录")
    print(f"基础项：{basic_path}")
    print(f"谐波项：{harmonic_path}")
    print(f"计算项：{calculated_path}")
    print(f"日志：{log_path}")
    print(f"配置：{config_path}")
    return 0 if seconds_written > 0 else 2


def main() -> int:
    try:
        return run(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
