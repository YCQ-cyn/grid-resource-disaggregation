#!/usr/bin/env python3
"""COMTRADE input discovery, archive extraction, CFG parsing and DAT reading.

The module intentionally depends only on NumPy plus the Python standard library.
ZIP files are supported directly. 7z files use an installed 7z/7zz/7za command
or the optional ``py7zr`` package.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

import numpy as np


EPOCH = datetime(1970, 1, 1)
SOURCE_TIME_RE = re.compile(
    r"_(\d{8})_(\d{6})_(\d+)(?:\.(?:zip|7z))?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AnalogChannel:
    index: int
    name: str
    unit: str
    a: float
    b: float
    primary: float
    secondary: float
    ps: str


@dataclass(frozen=True)
class ComtradeConfig:
    cfg_path: Path
    station_name: str
    device_id: str
    revision: str
    total_channels: int
    analog_count: int
    digital_count: int
    analog_channels: Tuple[AnalogChannel, ...]
    nominal_frequency: float
    sample_rates: Tuple[Tuple[float, int], ...]
    start_time: datetime
    trigger_time: datetime
    data_format: str
    time_multiplier: float

    @property
    def sample_rate(self) -> float:
        if len(self.sample_rates) != 1:
            raise ValueError(
                f"仅支持单采样率 CFG，{self.cfg_path} 中发现 {len(self.sample_rates)} 段采样率"
            )
        return self.sample_rates[0][0]

    @property
    def declared_last_sample(self) -> int:
        return self.sample_rates[-1][1]


@dataclass(frozen=True)
class RecordingSource:
    path: Path
    kind: str  # extracted | zip | 7z
    source_name: str


@dataclass
class OpenedRecording:
    source: RecordingSource
    cfg: ComtradeConfig
    dat_path: Path
    data: np.memmap
    channel_indexes: Tuple[int, int, int, int, int, int]
    channel_a: np.ndarray
    channel_b: np.ndarray
    start_us: int
    record_count: int
    trailing_bytes: int
    timestamp_mode: str
    first_dat_timestamp: int

    @property
    def sample_rate(self) -> float:
        return self.cfg.sample_rate

    def iter_chunks(
        self,
        chunk_records: int,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(absolute_time_us, waveform)`` arrays.

        Waveform column order is always Ua, Ub, Uc, Ia, Ib, Ic and engineering
        units are V, V, V, A, A, A.
        """

        if chunk_records <= 0:
            raise ValueError("chunk_records 必须大于 0")
        sample_period_us = 1_000_000.0 / self.sample_rate
        use_dat_timestamps = self.timestamp_mode.startswith("dat_timestamp")

        for begin in range(0, self.record_count, chunk_records):
            end = min(begin + chunk_records, self.record_count)
            chunk = self.data[begin:end]
            if use_dat_timestamps:
                relative = (
                    chunk["timestamp"].astype(np.float64)
                    - float(self.first_dat_timestamp)
                )
                rel_us = np.rint(
                    relative * self.cfg.time_multiplier
                ).astype(np.int64)
            else:
                positions = begin + np.arange(end - begin, dtype=np.int64)
                rel_us = np.rint(
                    positions.astype(np.float64) * sample_period_us
                ).astype(np.int64)

            raw = np.asarray(chunk["analog"][:, self.channel_indexes], dtype=np.float64)
            waveform = raw * self.channel_a + self.channel_b
            yield rel_us + self.start_us, waveform


def parse_comtrade_datetime(text: str) -> datetime:
    text = text.strip()
    formats = (
        "%d/%m/%Y,%H:%M:%S.%f",
        "%d/%m/%Y,%H:%M:%S",
        "%m/%d/%Y,%H:%M:%S.%f",
        "%m/%d/%Y,%H:%M:%S",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法识别 COMTRADE 时间：{text!r}")


def read_cfg(cfg_path: Path) -> ComtradeConfig:
    raw = cfg_path.read_bytes()
    text: Optional[str] = None
    for encoding in ("utf-8-sig", "gb18030", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"无法解码 CFG：{cfg_path}")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        raise ValueError(f"CFG 内容过短：{cfg_path}")

    header = [part.strip() for part in lines[0].split(",")]
    channel_header = [part.strip() for part in lines[1].split(",")]
    total_channels = int(channel_header[0])
    analog_count = int(channel_header[1].upper().rstrip("A"))
    digital_count = int(channel_header[2].upper().rstrip("D"))

    cursor = 2
    analog_channels = []
    for _ in range(analog_count):
        parts = [part.strip() for part in lines[cursor].split(",")]
        if len(parts) < 13:
            raise ValueError(f"模拟通道行格式错误：{lines[cursor]}")
        analog_channels.append(
            AnalogChannel(
                index=int(parts[0]),
                name=parts[1],
                unit=parts[4],
                a=float(parts[5]),
                b=float(parts[6]),
                primary=float(parts[10]),
                secondary=float(parts[11]),
                ps=parts[12].upper(),
            )
        )
        cursor += 1

    cursor += digital_count
    nominal_frequency = float(lines[cursor])
    cursor += 1
    rate_count = int(lines[cursor])
    cursor += 1
    sample_rates = []
    for _ in range(rate_count):
        parts = [part.strip() for part in lines[cursor].split(",")]
        sample_rates.append((float(parts[0]), int(parts[1])))
        cursor += 1

    start_time = parse_comtrade_datetime(lines[cursor])
    cursor += 1
    trigger_time = parse_comtrade_datetime(lines[cursor])
    cursor += 1
    data_format = lines[cursor].upper()
    cursor += 1
    time_multiplier = float(lines[cursor]) if cursor < len(lines) else 1.0

    return ComtradeConfig(
        cfg_path=cfg_path,
        station_name=header[0] if header else "",
        device_id=header[1] if len(header) > 1 else "",
        revision=header[2] if len(header) > 2 else "1991",
        total_channels=total_channels,
        analog_count=analog_count,
        digital_count=digital_count,
        analog_channels=tuple(analog_channels),
        nominal_frequency=nominal_frequency,
        sample_rates=tuple(sample_rates),
        start_time=start_time,
        trigger_time=trigger_time,
        data_format=data_format,
        time_multiplier=time_multiplier,
    )


def normalize_channel_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resolve_waveform_channels(cfg: ComtradeConfig) -> Tuple[int, int, int, int, int, int]:
    aliases: Dict[str, Sequence[str]] = {
        "ua": ("ua", "uaph", "va", "van", "voltagea"),
        "ub": ("ub", "ubph", "vb", "vbn", "voltageb"),
        "uc": ("uc", "ucph", "vc", "vcn", "voltagec"),
        "ia": ("ia", "iaph", "currenta"),
        "ib": ("ib", "ibph", "currentb"),
        "ic": ("ic", "icph", "currentc"),
    }
    normalized = [normalize_channel_name(ch.name) for ch in cfg.analog_channels]
    result = []
    for target in ("ua", "ub", "uc", "ia", "ib", "ic"):
        matches = [i for i, value in enumerate(normalized) if value in aliases[target]]
        if len(matches) != 1:
            available = ", ".join(ch.name for ch in cfg.analog_channels)
            raise ValueError(f"无法唯一匹配 {target} 通道；现有通道：{available}")
        result.append(matches[0])
    return tuple(result)  # type: ignore[return-value]


def build_record_dtype(cfg: ComtradeConfig) -> np.dtype:
    if cfg.data_format == "BINARY":
        analog_dtype = "<i2"
    elif cfg.data_format == "BINARY32":
        analog_dtype = "<i4"
    elif cfg.data_format == "FLOAT32":
        analog_dtype = "<f4"
    else:
        raise ValueError(
            f"暂不支持 DAT 格式 {cfg.data_format}；仅支持 BINARY/BINARY32/FLOAT32"
        )

    fields = [
        ("sample_no", "<u4"),
        ("timestamp", "<u4"),
        ("analog", analog_dtype, (cfg.analog_count,)),
    ]
    digital_words = math.ceil(cfg.digital_count / 16)
    if digital_words:
        fields.append(("digital", "<u2", (digital_words,)))
    return np.dtype(fields, align=False)


def datetime_to_epoch_us(value: datetime) -> int:
    return int(round((value - EPOCH).total_seconds() * 1_000_000))


def parse_source_time(name: str) -> Optional[datetime]:
    match = SOURCE_TIME_RE.search(Path(name).name)
    if not match:
        return None
    date_text, time_text, _suffix = match.groups()
    return datetime.strptime(date_text + time_text, "%Y%m%d%H%M%S")


def _source_sort_key(source: RecordingSource) -> Tuple[int, datetime, int, str]:
    """Sort by filename timestamp, then by the numeric trailing sequence.

    This mirrors the original batch program and keeps adjacent archives in their
    physical acquisition order even if a vendor stops zero-padding the suffix.
    """

    match = SOURCE_TIME_RE.search(Path(source.source_name).name)
    if not match:
        return (1, datetime.max, 0, source.source_name)
    date_text, time_text, suffix = match.groups()
    parsed = datetime.strptime(date_text + time_text, "%Y%m%d%H%M%S")
    return (0, parsed, int(suffix), source.source_name)


def discover_recordings(input_path: Path) -> list[RecordingSource]:
    """Find unique recordings and avoid processing ZIP + extracted duplicates.

    When an extracted CFG and an archive share the same stem, the extracted
    recording is preferred because it avoids a redundant decompression step.
    """

    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在：{input_path}")

    if input_path.is_file():
        suffix = input_path.suffix.lower()
        if suffix in (".zip", ".7z"):
            return [
                RecordingSource(
                    path=input_path,
                    kind=suffix.lstrip("."),
                    source_name=input_path.stem,
                )
            ]
        if suffix in (".cfg", ".dat"):
            return [
                RecordingSource(
                    path=input_path.parent,
                    kind="extracted",
                    source_name=input_path.stem,
                )
            ]
        raise ValueError(f"不支持的输入文件：{input_path}")

    by_stem: Dict[str, RecordingSource] = {}
    for archive in sorted(
        [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in (".zip", ".7z")]
    ):
        by_stem.setdefault(
            archive.stem.lower(),
            RecordingSource(archive, archive.suffix.lower().lstrip("."), archive.stem),
        )

    for cfg_path in sorted(input_path.rglob("*.cfg")):
        parent_dats = sorted(cfg_path.parent.rglob("*.dat"))
        same_stem_dats = [
            path for path in parent_dats
            if path.stem.lower() == cfg_path.stem.lower()
        ]
        # Ignore orphan CFG remnants.  The original remote workflow processes
        # archives; a stray extracted CFG without a uniquely usable DAT must not
        # turn an otherwise complete archive sequence into an error.
        if len(same_stem_dats) != 1 and len(parent_dats) != 1:
            continue
        by_stem[cfg_path.stem.lower()] = RecordingSource(
            path=cfg_path.parent,
            kind="extracted",
            source_name=cfg_path.stem,
        )

    sources = sorted(by_stem.values(), key=_source_sort_key)
    if not sources:
        raise FileNotFoundError(f"在 {input_path} 下没有发现 ZIP、7z 或已解压 CFG/DAT")
    return sources


def find_cfg_dat(root: Path, preferred_stem: Optional[str] = None) -> Tuple[Path, Path]:
    cfgs = sorted(root.rglob("*.cfg"))
    dats = sorted(root.rglob("*.dat"))
    if preferred_stem:
        matching_cfgs = [p for p in cfgs if p.stem.lower() == preferred_stem.lower()]
        if len(matching_cfgs) == 1:
            cfgs = matching_cfgs
    if len(cfgs) != 1:
        raise ValueError(f"{root} 中应有且仅有一个 CFG，实际找到 {len(cfgs)} 个")
    cfg_path = cfgs[0]
    same_stem = [p for p in dats if p.stem.lower() == cfg_path.stem.lower()]
    if len(same_stem) == 1:
        return cfg_path, same_stem[0]
    if len(dats) == 1:
        return cfg_path, dats[0]
    raise ValueError(f"无法为 {cfg_path.name} 唯一匹配 DAT")


@contextmanager
def materialize_source(source: RecordingSource, keep_temp: bool = False) -> Iterator[Path]:
    if source.kind == "extracted":
        yield source.path
        return

    if keep_temp:
        destination = source.path.parent / f"{source.path.stem}_extracted"
        destination.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        temp = tempfile.TemporaryDirectory(prefix="comtrade_items_")
        destination = Path(temp.name)
        cleanup = True

    try:
        if source.kind == "zip":
            with zipfile.ZipFile(source.path) as archive:
                archive.extractall(destination)
        elif source.kind == "7z":
            seven_zip = shutil.which("7z") or shutil.which("7zz") or shutil.which("7za")
            if seven_zip:
                result = subprocess.run(
                    [seven_zip, "x", "-y", f"-o{destination}", str(source.path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                if result.returncode != 0:
                    raise RuntimeError(f"7z 解压失败：{result.stdout[-2000:]}")
            else:
                try:
                    import py7zr  # type: ignore
                except ImportError as exc:
                    raise RuntimeError("处理 7z 需要 7z 命令或 py7zr 包") from exc
                with py7zr.SevenZipFile(source.path, mode="r") as archive:
                    archive.extractall(path=destination)
        else:
            raise ValueError(f"未知输入类型：{source.kind}")
        yield destination
    finally:
        if cleanup:
            temp.cleanup()


def _choose_start_time(cfg: ComtradeConfig, source: RecordingSource, mode: str) -> datetime:
    filename_time = parse_source_time(source.source_name)
    if mode == "cfg" or filename_time is None:
        return cfg.start_time
    if mode == "filename":
        return filename_time
    difference = abs((cfg.start_time - filename_time).total_seconds())
    return cfg.start_time if difference <= 2.0 else filename_time


def _choose_timestamp_mode(
    data: np.memmap,
    cfg: ComtradeConfig,
    record_count: int,
    requested: str,
) -> str:
    if requested == "sample_index":
        return "sample_index_and_rate"
    if requested == "dat_timestamp":
        return "dat_timestamp_forced"

    probe_count = min(record_count, 4096)
    head = np.asarray(data[:probe_count]["timestamp"], dtype=np.int64)
    tail = np.asarray(data[-probe_count:]["timestamp"], dtype=np.int64)
    first = int(data[0]["timestamp"])
    last = int(data[record_count - 1]["timestamp"])
    monotonic = (
        (head.size < 2 or bool(np.all(np.diff(head) >= 0)))
        and (tail.size < 2 or bool(np.all(np.diff(tail) >= 0)))
    )
    dat_span_us = (last - first) * cfg.time_multiplier
    expected_span_us = (record_count - 1) * 1_000_000.0 / cfg.sample_rate
    duration_ok = abs(dat_span_us - expected_span_us) <= max(10_000.0, 0.02 * expected_span_us)
    if last > first and monotonic and duration_ok:
        return "dat_timestamp_validated"
    return "sample_index_and_rate"


@contextmanager
def open_recording(
    source: RecordingSource,
    time_source: str = "auto",
    relative_time_source: str = "sample_index",
    keep_temp: bool = False,
) -> Iterator[OpenedRecording]:
    with materialize_source(source, keep_temp=keep_temp) as root:
        cfg_path, dat_path = find_cfg_dat(root, source.source_name)
        cfg = read_cfg(cfg_path)
        dtype = build_record_dtype(cfg)
        file_size = dat_path.stat().st_size
        record_count = file_size // dtype.itemsize
        trailing_bytes = file_size % dtype.itemsize
        if record_count <= 0:
            raise ValueError(f"DAT 没有完整记录：{dat_path}")

        data = np.memmap(dat_path, dtype=dtype, mode="r", shape=(record_count,))
        channel_indexes = resolve_waveform_channels(cfg)
        channels = [cfg.analog_channels[index] for index in channel_indexes]
        scale_a = np.asarray([channel.a for channel in channels], dtype=np.float64)
        scale_b = np.asarray([channel.b for channel in channels], dtype=np.float64)
        start_time = _choose_start_time(cfg, source, time_source)
        timestamp_mode = _choose_timestamp_mode(
            data,
            cfg,
            record_count,
            relative_time_source,
        )

        opened = OpenedRecording(
            source=source,
            cfg=cfg,
            dat_path=dat_path,
            data=data,
            channel_indexes=channel_indexes,
            channel_a=scale_a,
            channel_b=scale_b,
            start_us=datetime_to_epoch_us(start_time),
            record_count=record_count,
            trailing_bytes=trailing_bytes,
            timestamp_mode=timestamp_mode,
            first_dat_timestamp=int(data[0]["timestamp"]),
        )
        try:
            yield opened
        finally:
            mmap_handle = getattr(data, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
