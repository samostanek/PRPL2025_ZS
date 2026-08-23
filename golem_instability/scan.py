"""Cross-shot search for the repetitive loop-voltage collapse.

The database holds thousands of discharges, so the scan is built to be cheap
per shot: the tiny ``shot_date`` / ``b_plasma`` files are checked before any
waveform is downloaded, everything that is fetched lands in the cache, and
shots are processed in parallel.
"""

from __future__ import annotations

import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

import numpy as np

from .config import MINOR_RADIUS_M
from .detect import DetectionParams, detect_in_shot, result_to_row
from .fetch import GolemFetcher
from .shotdata import Shot, ShotNotFound, waveform_value

logger = logging.getLogger(__name__)

#: Generous upper bound on the GOLEM shot rate, used to bracket date searches.
SHOTS_PER_DAY: int = 25


def parse_shot_date(text: str) -> Optional[dt.date]:
    """Parse the ``YY-MM-DD`` content of ``shotdir/<N>/shot_date``."""
    text = (text or "").strip()
    for fmt in ("%y-%m-%d", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def shot_date(fetcher: GolemFetcher, shot: int) -> Optional[dt.date]:
    """Date of a shot, or ``None`` when the shot does not exist."""
    text = fetcher.get_text(shot, "shot_date")
    return parse_shot_date(text) if text else None


def find_shot_near_date(fetcher: GolemFetcher, target: dt.date,
                        hi_shot: Optional[int] = None, max_probes: int = 60) -> int:
    """Binary-search the shot number closest to ``target``.

    Missing shot numbers are stepped over with exponential backoff, and the low
    bound is derived from a generous shots-per-day rate rather than from 1, to
    stay out of the dead zone of purged low shot numbers.
    """
    hi = hi_shot or fetcher.latest_shot()
    hi_date = shot_date(fetcher, hi)
    if hi_date is None:
        raise ShotNotFound(f"shot {hi} has no date")
    days_back = max((hi_date - target).days, 0)
    lo = max(hi - days_back * SHOTS_PER_DAY - 500, 1)

    probes = 0
    while lo < hi and probes < max_probes:
        mid = (lo + hi) // 2
        date = shot_date(fetcher, mid)
        step = 1
        while date is None and mid + step < hi and probes < max_probes:
            date = shot_date(fetcher, mid + step)
            probes += 1
            if date is not None:
                mid = mid + step
                break
            step *= 2
        probes += 1
        if date is None:
            hi = mid
            continue
        if date < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def scan_shots(fetcher: GolemFetcher, shots: Sequence[int],
               params: Optional[DetectionParams] = None,
               require_plasma: bool = True, workers: int = 8,
               progress_every: int = 25) -> pd.DataFrame:
    """Run the collapse detector over many shots.

    Args:
        fetcher: Configured fetcher (its cache is shared by all workers).
        shots: Shot numbers to examine.
        params: Detector tunables.
        require_plasma: Skip shots whose ``b_plasma`` flag is not set.
        workers: Thread pool size.
        progress_every: Log a progress line every N shots.

    Returns:
        One row per examined shot with the detector summary, sorted by
        ``instability_score`` descending.
    """
    params = params or DetectionParams()
    rows: List[Dict[str, object]] = []

    def work(shot_no: int) -> Optional[Dict[str, object]]:
        shot = Shot(shot_no, fetcher)
        try:
            if not shot.exists():
                return None
            if require_plasma and not shot.has_plasma():
                return {"shot_no": shot_no, "skipped": "no_plasma"}
            result = detect_in_shot(shot, params)
            return result_to_row(shot, result)
        except Exception as exc:  # a broken shot must not stop the scan
            logger.warning("shot %d failed: %s", shot_no, exc)
            return {"shot_no": shot_no, "skipped": f"error: {exc}"}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, s): s for s in shots}
        for i, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            if row is not None:
                rows.append(row)
            if progress_every and i % progress_every == 0:
                logger.info("scanned %d/%d shots (%d requests, %d cache hits)",
                            i, len(futures), fetcher.n_requests, fetcher.n_cache_hits)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if "instability_score" in frame:
        frame = frame.sort_values("instability_score", ascending=False, na_position="last")
    return frame.reset_index(drop=True)


def scan_range(fetcher: GolemFetcher, first: int, last: int, step: int = 1,
               **kwargs) -> pd.DataFrame:
    """Convenience wrapper: scan ``range(first, last + 1, step)``."""
    return scan_shots(fetcher, list(range(first, last + 1, step)), **kwargs)


def candidates(scan: pd.DataFrame, min_score: float = 0.5,
               min_events: int = 3) -> pd.DataFrame:
    """Subset of a scan table that looks like the instability of interest."""
    if scan.empty or "instability_score" not in scan:
        return scan
    mask = (scan["instability_score"] >= min_score) & (scan.get("n_events", 0) >= min_events)
    if "is_candidate" in scan:
        mask &= scan["is_candidate"].fillna(0) > 0
    return scan[mask].reset_index(drop=True)


def summarise_scan(scan: pd.DataFrame) -> str:
    """Short human-readable digest of a scan table."""
    if scan.empty:
        return "no shots scanned"
    examined = len(scan)
    skipped = int(scan["skipped"].notna().sum()) if "skipped" in scan else 0
    with_events = int((scan.get("n_events", pd.Series(dtype=float)).fillna(0) > 0).sum())
    cand = len(candidates(scan))
    lines = [f"scanned {examined} shots ({skipped} skipped), "
             f"{with_events} with at least one collapse, {cand} candidates"]
    top = candidates(scan).head(15)
    if not top.empty:
        cols = [c for c in ("shot_no", "shot_date", "n_events", "t_onset", "period_mean",
                            "depth_mean", "d_ip_mean", "instability_score") if c in top]
        lines.append(top[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return "\n".join(lines)


#: Discharge settings worth carrying into a cross-shot table.
DEFAULT_SETTINGS_KEYS: Tuple[str, ...] = (
    "bt_ecd.U_cd", "bt_ecd.U_Bt", "bt_ecd.t_cd", "bt_ecd.t_Bt",
    "workinggas.S_gas", "workinggas.p_H", "workinggas.p_He",
    "positionstabilization.main_switch",
    "positionstabilization.radial_switch", "positionstabilization.radial_waveform",
    "positionstabilization.vertical_switch", "positionstabilization.vertical_waveform",
    "currentstabilization.main_switch", "currentstabilization.waveform",
)

#: Waveform settings evaluated at the onset time of the first collapse.
_WAVEFORM_KEYS: Tuple[str, ...] = (
    "positionstabilization.radial_waveform",
    "positionstabilization.vertical_waveform",
    "currentstabilization.waveform",
)


def greenwald_density(ip_kA: float, minor_radius_m: float = MINOR_RADIUS_M) -> float:
    """Greenwald density limit in m^-3 for a plasma current in kA."""
    if not np.isfinite(ip_kA) or ip_kA <= 0:
        return float("nan")
    return float((ip_kA * 1e-3) / (np.pi * minor_radius_m ** 2) * 1e20)


def enrich_scan(fetcher: GolemFetcher, scan: pd.DataFrame,
                settings_keys: Sequence[str] = DEFAULT_SETTINGS_KEYS,
                signal_keys: Sequence[str] = ("ne_lav",),
                only_candidates: bool = False) -> pd.DataFrame:
    """Add machine settings and plasma conditions at onset to a scan table.

    For every shot with a detected onset this fetches the requested discharge
    settings, evaluates the control waveforms *at the onset time*, and samples
    the requested signals there. That is what makes the cross-shot questions
    answerable: does the instability start at a fixed time, at a fixed
    stabilisation drive, or at a fixed density?

    Args:
        fetcher: Fetcher (cached).
        scan: Table from :func:`scan_shots`.
        settings_keys: Discharge-setting keys to carry over.
        signal_keys: Signals to sample at onset and average over the shot.
        only_candidates: Restrict the (network-costly) enrichment to shots that
            the detector flagged.

    Returns:
        A copy of ``scan`` with the extra columns.
    """
    if scan.empty:
        return scan
    frame = scan.copy()
    rows = frame[frame.get("t_onset").notna()] if "t_onset" in frame else frame.iloc[0:0]
    if only_candidates and "is_candidate" in frame:
        rows = rows[rows["is_candidate"] > 0]

    extra: Dict[int, Dict[str, object]] = {}
    for _, row in rows.iterrows():
        shot_no = int(row["shot_no"])
        onset = float(row["t_onset"])
        values: Dict[str, object] = {}
        try:
            shot = Shot(shot_no, fetcher)
            settings = shot.discharge_settings()
            for key in settings_keys:
                if key in settings:
                    values[key.split(".")[-1] if key.count(".") else key] = settings[key]
            for key in _WAVEFORM_KEYS:
                if key in settings:
                    label = key.split(".")[-1].replace("waveform", "") or "current"
                    values[f"{label.rstrip('_')}_at_onset"] = waveform_value(settings[key], onset)
            for key in signal_keys:
                sig = shot.signal(key)
                if sig is None:
                    continue
                window = sig.y[(sig.t >= onset - 0.2) & (sig.t <= onset + 0.2)]
                values[f"{key}_at_onset"] = float(np.nanmedian(window)) if window.size else np.nan
            ip_max = fetcher.get_float(shot_no, "Diagnostics/BasicDiagnostics/Results/Ip_max")
            if ip_max:
                values["Ip_max"] = ip_max
                n_gw = greenwald_density(ip_max)
                ne = values.get("ne_lav_at_onset")
                values["n_greenwald"] = n_gw
                if isinstance(ne, float) and np.isfinite(ne) and np.isfinite(n_gw):
                    values["f_greenwald_at_onset"] = ne / n_gw
        except Exception as exc:  # enrichment must never break a scan
            logger.warning("shot %d: enrichment failed (%s)", shot_no, exc)
        extra[shot_no] = values

    for column in sorted({k for v in extra.values() for k in v}):
        frame[column] = frame["shot_no"].map(
            lambda s: extra.get(int(s), {}).get(column, np.nan))
    return frame


#: Physical constants for the edge safety factor.
_MU0 = 4e-7 * np.pi


def onset_conditions(fetcher: GolemFetcher, scan: pd.DataFrame) -> pd.DataFrame:
    """Plasma conditions at the moment the instability starts, per shot.

    Columns: the onset time, the plasma current there both absolutely and as a
    fraction of that shot's peak, the edge safety factor, the toroidal field,
    the flux already delivered, and the line-averaged density.
    """
    if scan.empty or "t_onset" not in scan:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for _, row in scan.dropna(subset=["t_onset"]).iterrows():
        shot_no, onset = int(row["shot_no"]), float(row["t_onset"])
        try:
            shot = Shot(shot_no, fetcher)
            ip, bt, u = shot.signal("Ip"), shot.signal("Bt"), shot.signal("U_loop")
            if ip is None or u is None:
                continue
            ip_on = float(np.interp(onset, ip.t, ip.y))
            ip_max = float(np.nanmax(ip.y))
            bt_on = float(np.interp(onset, bt.t, bt.y)) if bt is not None else np.nan
            q_a = (2 * np.pi * MINOR_RADIUS_M ** 2 * bt_on
                   / (_MU0 * 0.40 * ip_on * 1e3)) if ip_on > 0 else np.nan
            window = shot.plasma_window()
            t0 = window[0] if window else float(u.t[0])
            mask = (u.t >= t0) & (u.t <= onset)
            trapz = getattr(np, "trapezoid", None) or np.trapz
            flux = float(trapz(u.y[mask], u.t[mask])) if mask.sum() > 2 else np.nan
            ne = shot.signal("ne_lav")
            ne_on = (float(np.interp(onset, ne.t, ne.y)) if ne is not None else np.nan)
            rows.append({
                "shot_no": shot_no, "t_onset_ms": onset, "Ip_at_onset_kA": ip_on,
                "Ip_max_kA": ip_max,
                "Ip_frac_of_max": ip_on / ip_max if ip_max else np.nan,
                "q_a_at_onset": q_a, "Bt_at_onset_T": bt_on,
                "flux_at_onset_mWb": flux, "ne_at_onset": ne_on,
                "t_Ip_max_ms": float(ip.t[int(np.nanargmax(ip.y))]),
            })
        except Exception as exc:
            logger.warning("shot %d: onset conditions failed (%s)", shot_no, exc)
    return pd.DataFrame(rows)


def onset_criterion_test(conditions: pd.DataFrame) -> pd.DataFrame:
    """Which quantity is the instability's trigger?

    Every candidate trigger is measured at the onset in every shot; the one
    that is genuinely the threshold should be the one that varies *least* from
    shot to shot, even as the others range widely. Scatter is reported as the
    interquartile range over the median, which is insensitive to the handful
    of shots whose onset is caught early or late.

    Returns:
        One row per candidate criterion, tightest first.
    """
    if conditions.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for column in ("Ip_frac_of_max", "t_onset_ms", "flux_at_onset_mWb", "Ip_at_onset_kA",
                   "q_a_at_onset", "Bt_at_onset_T", "ne_at_onset"):
        if column not in conditions:
            continue
        values = pd.to_numeric(conditions[column], errors="coerce").dropna()
        if len(values) < 3:
            continue
        median = float(values.median())
        iqr = float(values.quantile(0.75) - values.quantile(0.25))
        rows.append({"criterion": column, "n": len(values), "median": median,
                     "iqr": iqr,
                     "relative_spread": abs(iqr / median) if median else np.nan})
    frame = pd.DataFrame(rows)
    return frame.sort_values("relative_spread").reset_index(drop=True) if not frame.empty else frame
