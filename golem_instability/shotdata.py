"""Shot-level view of the database: metadata, active diagnostics, signals."""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import config
from .fetch import GolemFetcher, ShotNotFound
from .config import SignalSpec

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """A single time trace on the shot's common time axis.

    Attributes:
        key: Registry key (``"U_loop"``, ``"Ip"``, ...).
        t: Time in milliseconds, ``t=0`` at the discharge trigger.
        y: Values in :attr:`unit`.
        unit: Physical unit of ``y``.
        label: Legend label.
        group: Category from the registry (``basic``, ``mhd``, ...).
    """

    key: str
    t: np.ndarray
    y: np.ndarray
    unit: str = ""
    label: str = ""
    group: str = ""

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.t.shape != self.y.shape:
            raise ValueError(f"{self.key}: time/value length mismatch "
                             f"({self.t.shape} vs {self.y.shape})")
        if not self.label:
            self.label = self.key

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def dt(self) -> float:
        """Median sample spacing in ms (the grid is not always exactly uniform)."""
        return float(np.median(np.diff(self.t))) if self.t.size > 1 else float("nan")

    @property
    def fs_khz(self) -> float:
        """Sampling rate in kHz."""
        return 1.0 / self.dt if self.dt else float("nan")

    def window(self, t0: float, t1: float) -> "Signal":
        """Restrict to ``t0 <= t <= t1`` (ms)."""
        m = (self.t >= t0) & (self.t <= t1)
        return Signal(self.key, self.t[m], self.y[m], self.unit, self.label, self.group)

    def at(self, times: Sequence[float]) -> np.ndarray:
        """Linear interpolation onto arbitrary times (NaN outside the range)."""
        times = np.asarray(times, dtype=float)
        out = np.interp(times, self.t, self.y, left=np.nan, right=np.nan)
        return out

    def baseline_corrected(self, t_ref: float = -1.0) -> "Signal":
        """Subtract the mean of the pre-trigger samples (``t < t_ref``)."""
        pre = self.y[self.t < t_ref]
        offset = float(np.mean(pre)) if pre.size > 5 else 0.0
        return Signal(self.key, self.t, self.y - offset, self.unit, self.label, self.group)

    def to_series(self) -> pd.Series:
        return pd.Series(self.y, index=pd.Index(self.t, name="t_ms"), name=self.key)


def parse_signal_csv(raw: bytes, spec: Optional[SignalSpec] = None, key: str = "") -> Signal:
    """Parse the two-column headerless ``time_ms,value`` convention.

    Some files carry a stray header line or a trailing comment; anything that
    does not parse as two floats is dropped rather than aborting the load.
    """
    frame = pd.read_csv(
        io.BytesIO(raw),
        header=None,
        names=["t", "y"],
        usecols=[0, 1],
        comment="#",
        on_bad_lines="skip",
    )
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    if frame.empty:
        raise ValueError("no numeric rows in signal CSV")
    if spec is not None:
        return Signal(spec.key, frame["t"].to_numpy(), frame["y"].to_numpy(),
                      spec.unit, spec.label, spec.group)
    return Signal(key or "signal", frame["t"].to_numpy(), frame["y"].to_numpy())


#: Header keys of the Tektronix/Rigol multi-channel export used for raw dumps.
_DAS_HEADER_KEYS = ("model", "channel", "label", "waveform type", "digital type",
                    "horizontal units", "sample interval", "record length",
                    "vertical units", "vertical scale", "vertical offset", "note")


def parse_das_csv(raw: bytes, time_scale_to_ms: Optional[float] = None) -> Dict[str, Signal]:
    """Parse a raw digitiser dump (``DAS_raw_data_dir/<Instrument>_ALL.csv``).

    These are vendor exports rather than the two-column convention: a header
    block of ``key,value,value,...`` lines carrying the per-channel labels and
    the sample interval, followed by the samples. The data rows appear either
    as one ``time,value`` pair per channel or as a single shared time column
    followed by one column per channel; both are handled, and blank separator
    columns are dropped.

    Args:
        raw: File content.
        time_scale_to_ms: Multiplier from the file's time unit to ms; deduced
            from the ``Horizontal Units`` header when omitted (``s`` -> 1e3).

    Returns:
        Mapping of channel label to :class:`Signal`. Labels are taken from the
        ``Label`` header row, falling back to ``Channel`` names.
    """
    text = raw.decode("utf-8", errors="replace")
    header: Dict[str, List[str]] = {}
    data_start = 0
    lines = text.splitlines()
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.split(",")]
        key = cells[0].lower()
        if key in _DAS_HEADER_KEYS:
            # The row repeats "<key>,<value>,..." once per channel group; keep
            # the first non-empty cell after each occurrence of the key.
            values: List[str] = []
            expecting = False
            for cell in cells:
                if cell.lower() == key:
                    expecting = True
                elif expecting and cell:
                    values.append(cell)
                    expecting = False
            header.setdefault(key, []).extend(values)
            data_start = i + 1
        elif header and line.strip() and re.match(r"^\s*[-+0-9.eE]", line):
            data_start = i
            break
    if not header:
        raise ValueError("no recognisable digitiser header")

    labels = [c for c in header.get("label", []) if c]
    if not labels:
        labels = [c for c in header.get("channel", []) if c]
    if not labels:
        raise ValueError("no channel labels in digitiser header")

    frame = pd.read_csv(io.StringIO("\n".join(lines[data_start:])), header=None,
                        on_bad_lines="skip")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(axis=1, how="all").dropna(how="all")
    if frame.empty:
        raise ValueError("no numeric samples in digitiser dump")

    if time_scale_to_ms is None:
        unit = (header.get("horizontal units") or ["s"])[0].strip().lower()
        time_scale_to_ms = {"s": 1e3, "ms": 1.0, "us": 1e-3}.get(unit, 1e3)

    columns = list(frame.columns)
    out: Dict[str, Signal] = {}
    if len(columns) >= 2 * len(labels):
        for i, label in enumerate(labels):
            t = frame[columns[2 * i]].to_numpy() * time_scale_to_ms
            y = frame[columns[2 * i + 1]].to_numpy()
            out[label] = Signal(label, t, y, "", label, "raw")
    else:
        t = frame[columns[0]].to_numpy() * time_scale_to_ms
        for i, label in enumerate(labels[: len(columns) - 1]):
            out[label] = Signal(label, t, frame[columns[i + 1]].to_numpy(), "", label, "raw")
    return out


@dataclass
class Shot:
    """Lazy accessor for everything belonging to one discharge.

    Args:
        shot_no: Concrete shot number.
        fetcher: Configured :class:`~golem_instability.fetch.GolemFetcher`.
    """

    shot_no: int
    fetcher: GolemFetcher
    _signals: Dict[str, Optional[Signal]] = field(default_factory=dict, repr=False)
    _meta: Optional[Dict[str, str]] = field(default=None, repr=False)
    _diagnostics: Optional[Dict[str, List[str]]] = field(default=None, repr=False)

    # ------------------------------------------------------------- metadata
    @property
    def meta(self) -> Dict[str, str]:
        """Small text metadata files (date, time, session, comment)."""
        if self._meta is None:
            meta: Dict[str, str] = {}
            for name in config.META_FILES:
                text = self.fetcher.get_text(self.shot_no, name)
                if text is not None:
                    meta[name] = text.strip()
            if "shot_no" not in meta:
                raise ShotNotFound(f"shot {self.shot_no} has no shot_no file")
            self._meta = meta
        return self._meta

    def exists(self) -> bool:
        try:
            return bool(self.meta)
        except ShotNotFound:
            return False

    @property
    def diagnostics(self) -> Dict[str, List[str]]:
        """Diagnostics that actually ran, split into basic/on-stage/off-stage.

        The ``Production/Diagnostics_*`` files hold ``Name/Variant`` tokens;
        only the ``Name`` part (the folder under ``Diagnostics/``) is kept.
        """
        if self._diagnostics is None:
            out: Dict[str, List[str]] = {}
            for group, path in config.PRODUCTION_DIAGNOSTIC_FILES.items():
                text = self.fetcher.get_text(self.shot_no, path) or ""
                out[group] = sorted({tok.split("/")[0] for tok in text.split() if tok})
            self._diagnostics = out
        return self._diagnostics

    @property
    def active_diagnostics(self) -> List[str]:
        """Flat list of diagnostic folder names available for this shot."""
        return sorted({name for names in self.diagnostics.values() for name in names})

    def has_plasma(self) -> bool:
        """``Diagnostics/PlasmaDetection/Results/b_plasma`` as a bool."""
        flag = self.fetcher.get_float(self.shot_no, config.PLASMA_FLAG_PATH)
        return bool(flag and flag > 0.5)

    def scalars(self) -> Dict[str, Optional[float]]:
        """Pre-computed headline numbers (max/mean Ip, Bt, plasma timing)."""
        return {
            name: self.fetcher.get_float(self.shot_no, path)
            for name, path in config.SCALAR_FILES.items()
        }

    def plasma_window(self, pad_ms: float = 0.0) -> Optional[tuple]:
        """``(t_start, t_end)`` in ms from PlasmaDetection, or ``None``."""
        t0 = self.fetcher.get_float(self.shot_no, config.SCALAR_FILES["t_plasma_start"])
        t1 = self.fetcher.get_float(self.shot_no, config.SCALAR_FILES["t_plasma_end"])
        if t0 is None or t1 is None or not np.isfinite([t0, t1]).all() or t1 <= t0:
            return None
        return (t0 - pad_ms, t1 + pad_ms)

    def setup_parameters(self) -> Dict[str, str]:
        """Requested discharge parameters parsed out of the ``*.setup`` files.

        The setup files are ``key=value`` (or ``key value``) lines; the exact
        key set varies with the session, so everything found is returned and
        callers pick what they need (``U_CD``, ``U_Bt``, pressure, gas, ...).
        """
        params: Dict[str, str] = {}
        listing = self.fetcher.get_text(self.shot_no, "") or ""
        names = set(re.findall(r'href="([^"]+\.setup)"', listing))
        names |= {"session.setup", "whole.setup", "mission.setup"}
        for name in sorted(names):
            text = self.fetcher.get_text(self.shot_no, name)
            if not text:
                continue
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                match = re.match(r"^([A-Za-z_][\w.\-]*)\s*[=:\t ]\s*(.+)$", line)
                if match:
                    params.setdefault(match.group(1), match.group(2).strip())
        return params

    # -------------------------------------------------------------- signals
    def signal(self, key: str) -> Optional[Signal]:
        """Load one registry signal, or ``None`` if it is absent for this shot."""
        if key in self._signals:
            return self._signals[key]
        spec = config.signal(key)
        result: Optional[Signal] = None
        if spec.diagnostic and spec.diagnostic not in self.active_diagnostics:
            logger.debug("shot %d: %s did not run", self.shot_no, spec.diagnostic)
        else:
            raw = self.fetcher.get_bytes(self.shot_no, spec.path)
            if raw is None:
                logger.debug("shot %d: %s missing", self.shot_no, spec.path)
            else:
                try:
                    result = parse_signal_csv(raw, spec)
                except (ValueError, pd.errors.ParserError) as exc:
                    logger.warning("shot %d: cannot parse %s (%s)", self.shot_no, spec.path, exc)
        self._signals[key] = result
        return result

    def signals(self, keys: Iterable[str]) -> Dict[str, Signal]:
        """Load several signals, silently skipping the ones that are absent."""
        out: Dict[str, Signal] = {}
        for key in keys:
            sig = self.signal(key)
            if sig is not None and len(sig) > 1:
                out[key] = sig
        return out

    def raw_das(self, diagnostic: str = "BasicDiagnostics",
                filename: Optional[str] = None) -> Dict[str, Signal]:
        """Load a raw digitiser dump for a diagnostic, keyed by channel label.

        Use this when the processed ``Results/*.csv`` are not enough - for
        instance to check whether a feature is present in the *uncorrected*
        Rogowski channel, or whether the loop-voltage channel was clipped.
        Returns an empty dict when no dump is available.
        """
        base = f"Diagnostics/{diagnostic}/DAS_raw_data_dir/"
        candidates = [filename] if filename else []
        if not candidates:
            listing = self.fetcher.get_text(self.shot_no, base) or ""
            candidates = re.findall(r'href="([^"]*_ALL\.csv)"', listing)
        for name in candidates:
            raw = self.fetcher.get_bytes(self.shot_no, base + name.lstrip("/"))
            if raw is None:
                continue
            try:
                return parse_das_csv(raw)
            except (ValueError, pd.errors.ParserError) as exc:
                logger.warning("shot %d: cannot parse %s (%s)", self.shot_no, name, exc)
        return {}

    def available_signals(self) -> List[str]:
        """Registry keys whose diagnostic ran on this shot."""
        active = set(self.active_diagnostics)
        return [k for k, s in config.SIGNALS.items() if not s.diagnostic or s.diagnostic in active]

    def inventory(self) -> pd.DataFrame:
        """Table of every signal available for this shot, with basic statistics.

        Useful as the first step of a phenomenological survey: it says which
        diagnostics ran, at what sampling rate, over what time span and with
        what dynamic range, without having to plot anything.
        """
        rows: List[Dict[str, object]] = []
        for key in self.available_signals():
            sig = self.signal(key)
            if sig is None or len(sig) < 2:
                continue
            spec = config.signal(key)
            rows.append({
                "key": key,
                "diagnostic": spec.diagnostic or "-",
                "unit": spec.unit,
                "n_samples": len(sig),
                "dt_ms": sig.dt,
                "fs_kHz": sig.fs_khz,
                "t_start_ms": float(sig.t[0]),
                "t_end_ms": float(sig.t[-1]),
                "min": float(np.nanmin(sig.y)),
                "max": float(np.nanmax(sig.y)),
                "mean": float(np.nanmean(sig.y)),
            })
        return pd.DataFrame(rows)

    def describe(self) -> str:
        meta = self.meta
        return (f"shot #{self.shot_no}  {meta.get('shot_date', '?')} {meta.get('shot_time', '')}  "
                f"session {meta.get('session_id', '?')}\n"
                f"  comment    : {meta.get('comment', '')}\n"
                f"  diagnostics: {', '.join(self.active_diagnostics) or 'none listed'}\n"
                f"  plasma     : {self.has_plasma()}")


def load_shot(shot_no: int, fetcher: Optional[GolemFetcher] = None) -> Shot:
    """Convenience constructor with a default fetcher."""
    return Shot(shot_no, fetcher or GolemFetcher())
