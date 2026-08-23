"""Detection of repetitive loop-voltage collapses ("U_loop dropouts").

Phenomenology being targeted (as first seen on shot #53740): some tens of
milliseconds into the discharge the loop voltage starts *repeatedly* collapsing
towards zero and recovering, and the plasma current steps down a little at
every collapse, producing a staircase in :math:`I_p(t)`.

The detector is deliberately envelope-relative rather than absolute: ``U_loop``
decays throughout a GOLEM discharge, so a fixed voltage threshold would either
miss early events or fire on the natural end-of-discharge tail. A collapse is
defined as an excursion of ``U_loop`` below a fraction of its own local upper
envelope, with hysteresis so that one physical event is not split in two.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .shotdata import Shot, Signal

#: numpy>=2 renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz

logger = logging.getLogger(__name__)


@dataclass
class DetectionParams:
    """Tunables of the collapse detector.

    Attributes:
        env_window_ms: Width of the running window used for the upper envelope.
        env_quantile: Quantile taken inside that window (0.9 ≈ "typical high").
        enter_ratio: ``U/envelope`` below which a collapse starts.
        exit_ratio: ``U/envelope`` above which it ends (hysteresis).
        min_duration_ms: Reject excursions shorter than this (noise). The real
            events are fast - on GOLEM's 1 MHz basic diagnostics they last
            ~10 us - so both this and ``smooth_ms`` are set in microseconds.
        min_gap_ms: Merge two events separated by less than this.
        smooth_ms: Width of the running median applied before thresholding.
        env_floor_frac: Ignore times where the envelope has fallen below this
            fraction of its shot maximum, i.e. the natural discharge tail.
        edge_guard_ms: Ignore events touching the ends of the analysis window.
        ip_level_ms: Averaging width for the pre/post plasma-current levels.
        min_events: Events required before a shot counts as a candidate.
        min_depth: Mean relative depth required for a candidate.
    """

    env_window_ms: float = 2.0
    env_quantile: float = 0.90
    enter_ratio: float = 0.40
    exit_ratio: float = 0.65
    min_duration_ms: float = 0.004
    min_gap_ms: float = 0.02
    smooth_ms: float = 0.005
    env_floor_frac: float = 0.15
    edge_guard_ms: float = 0.30
    ip_level_ms: float = 0.30
    min_events: int = 3
    min_depth: float = 0.55


EVENT_COLUMNS: Tuple[str, ...] = (
    "t_on", "t_off", "duration_ms", "u_min", "u_env", "depth",
    "flux_deficit_mWb", "ip_pre", "ip_post", "d_ip", "d_ip_rel",
    "dipdt_in", "tau_LR_ms", "u_overshoot", "period_ms",
)


def _uniform_grid(sig: Signal) -> Tuple[np.ndarray, np.ndarray]:
    """Return the signal on a strictly uniform time grid (interpolating if needed)."""
    t, y = sig.t, sig.y
    order = np.argsort(t)
    t, y = t[order], y[order]
    steps = np.diff(t)
    if steps.size == 0:
        return t, y
    dt = float(np.median(steps))
    if dt <= 0:
        raise ValueError("non-monotonic time axis")
    if np.allclose(steps, dt, rtol=0.02, atol=0.0):
        return t, y
    grid = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    return grid, np.interp(grid, t, y)


def _running(values: np.ndarray, width_samples: int, how: str, q: float = 0.9) -> np.ndarray:
    """Centred running median/quantile that keeps the array length."""
    width_samples = max(1, int(width_samples) | 1)  # force odd
    series = pd.Series(values)
    roll = series.rolling(width_samples, center=True, min_periods=1)
    out = roll.median() if how == "median" else roll.quantile(q)
    return out.to_numpy()


def _hysteresis_intervals(below_enter: np.ndarray, below_exit: np.ndarray) -> List[Tuple[int, int]]:
    """Grow every ``below_enter`` core outwards while ``below_exit`` holds.

    Returns index intervals ``[i0, i1)``.
    """
    intervals: List[Tuple[int, int]] = []
    n = below_enter.size
    i = 0
    while i < n:
        if not below_enter[i]:
            i += 1
            continue
        left = i
        while left > 0 and below_exit[left - 1]:
            left -= 1
        right = i
        while right + 1 < n and below_exit[right + 1]:
            right += 1
        intervals.append((left, right + 1))
        i = right + 1
    # Merge overlaps produced by adjacent cores.
    merged: List[Tuple[int, int]] = []
    for start, stop in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


@dataclass
class DetectionResult:
    """Everything the detector produced for one shot.

    Attributes:
        shot_no: Shot number.
        events: One row per collapse, columns as in :data:`EVENT_COLUMNS`.
        summary: Shot-level metrics (see :func:`summarise`).
        t: Time axis of the analysed window (ms).
        u: Loop voltage on that axis, sign-corrected (V).
        u_env: Running upper envelope of ``u`` (V).
        ratio: ``u / u_env``.
        polarity: ``+1`` or ``-1``; the sign applied to the raw loop voltage.
        params: The :class:`DetectionParams` used.
    """

    shot_no: int
    events: pd.DataFrame
    summary: Dict[str, float]
    t: np.ndarray
    u: np.ndarray
    u_env: np.ndarray
    ratio: np.ndarray
    polarity: float
    params: DetectionParams

    @property
    def is_candidate(self) -> bool:
        return bool(self.summary.get("is_candidate", 0.0))

    @property
    def n_events(self) -> int:
        return int(len(self.events))


def analysis_window(shot: Shot, u_loop: Signal, params: DetectionParams) -> Tuple[float, float]:
    """Pick the time span to analyse: the plasma window when it is available."""
    window = shot.plasma_window()
    if window is not None:
        t0, t1 = window
        if t1 - t0 > 1.0:
            return max(t0, float(u_loop.t[0])), min(t1, float(u_loop.t[-1]))
    # Fall back to "loop voltage is appreciable".
    y = np.abs(u_loop.y)
    strong = np.flatnonzero(y > 0.2 * np.nanmax(y))
    if strong.size < 2:
        return float(u_loop.t[0]), float(u_loop.t[-1])
    return float(u_loop.t[strong[0]]), float(u_loop.t[strong[-1]])


def detect_collapses(
    u_loop: Signal,
    ip: Optional[Signal] = None,
    window: Optional[Tuple[float, float]] = None,
    params: Optional[DetectionParams] = None,
    shot_no: int = -1,
) -> DetectionResult:
    """Find repetitive loop-voltage collapses in one discharge.

    Args:
        u_loop: Loop voltage trace.
        ip: Plasma current trace, used for the per-event current step.
        window: ``(t0, t1)`` in ms; defaults to the whole ``u_loop`` trace.
        params: Detector tunables.
        shot_no: Shot number, carried through into the result.

    Returns:
        A :class:`DetectionResult`.
    """
    params = params or DetectionParams()
    if window is not None:
        u_loop = u_loop.window(*window)
    if len(u_loop) < 32:
        empty = pd.DataFrame(columns=list(EVENT_COLUMNS))
        return DetectionResult(shot_no, empty, {"n_events": 0.0, "is_candidate": 0.0},
                               u_loop.t, u_loop.y, u_loop.y, np.ones_like(u_loop.y),
                               1.0, params)

    t, u_raw = _uniform_grid(u_loop)
    dt = float(np.median(np.diff(t)))
    polarity = 1.0 if np.nanmedian(u_raw) >= 0 else -1.0
    u = polarity * u_raw

    smooth_n = max(1, int(round(params.smooth_ms / dt)))
    u_s = _running(u, smooth_n, "median")
    env_n = max(3, int(round(params.env_window_ms / dt)))
    u_env = _running(u_s, env_n, "quantile", params.env_quantile)
    u_env = np.maximum(u_env, 1e-9)

    ratio = np.clip(u_s / u_env, -5.0, 5.0)
    valid = u_env > params.env_floor_frac * np.nanmax(u_env)
    below_enter = (ratio < params.enter_ratio) & valid
    below_exit = (ratio < params.exit_ratio) & valid

    intervals = _hysteresis_intervals(below_enter, below_exit)
    intervals = _merge_close(intervals, gap_samples=max(1, int(round(params.min_gap_ms / dt))))
    intervals = [(a, b) for a, b in intervals if (b - a) * dt >= params.min_duration_ms]
    guard = params.edge_guard_ms
    intervals = [(a, b) for a, b in intervals
                 if t[a] > t[0] + guard and t[min(b, t.size - 1)] < t[-1] - guard]

    events = _event_table(t, u_s, u_env, intervals, ip, polarity, params)
    summary = summarise(events, t, u_s, params)
    summary["polarity"] = polarity
    return DetectionResult(shot_no, events, summary, t, u_s, u_env, ratio, polarity, params)


def _merge_close(intervals: List[Tuple[int, int]], gap_samples: int) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for start, stop in intervals:
        if merged and start - merged[-1][1] <= gap_samples:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    return merged


def _event_table(
    t: np.ndarray,
    u: np.ndarray,
    u_env: np.ndarray,
    intervals: List[Tuple[int, int]],
    ip: Optional[Signal],
    polarity: float,
    params: DetectionParams,
) -> pd.DataFrame:
    """Build the per-event metric table."""
    dt = float(np.median(np.diff(t))) if t.size > 1 else float("nan")
    rows: List[Dict[str, float]] = []
    for start, stop in intervals:
        stop = min(stop, t.size)
        t_on, t_off = float(t[start]), float(t[stop - 1])
        seg_u = u[start:stop]
        seg_env = u_env[start:stop]
        u_min = float(np.nanmin(seg_u))
        env_level = float(np.nanmedian(seg_env))
        # Volt-seconds the plasma did not receive during the collapse.
        flux_deficit = float(np.nansum(np.clip(seg_env - seg_u, 0, None)) * dt)
        # Recovery overshoot: peak of U just after the event, relative to envelope.
        post = u[stop:stop + max(1, int(round(1.0 / dt)))]
        overshoot = float(np.nanmax(post) / env_level) if post.size and env_level > 0 else np.nan

        row: Dict[str, float] = {
            "t_on": t_on,
            "t_off": t_off,
            "duration_ms": t_off - t_on,
            "u_min": u_min,
            "u_env": env_level,
            "depth": 1.0 - u_min / env_level if env_level > 0 else np.nan,
            "flux_deficit_mWb": flux_deficit,
            "u_overshoot": overshoot,
        }
        row.update(_ip_metrics(ip, t_on, t_off, params))
        rows.append(row)

    table = pd.DataFrame(rows, columns=list(EVENT_COLUMNS))
    if not table.empty:
        table["period_ms"] = table["t_on"].diff()
    return table


def _ip_metrics(ip: Optional[Signal], t_on: float, t_off: float,
                params: DetectionParams) -> Dict[str, float]:
    """Plasma-current step across one collapse and the implied L/R time."""
    nan = {"ip_pre": np.nan, "ip_post": np.nan, "d_ip": np.nan,
           "d_ip_rel": np.nan, "dipdt_in": np.nan, "tau_LR_ms": np.nan}
    if ip is None or len(ip) < 4:
        return nan
    w = params.ip_level_ms
    pre_mask = (ip.t >= t_on - w) & (ip.t <= t_on)
    post_mask = (ip.t >= t_off) & (ip.t <= t_off + w)
    in_mask = (ip.t >= t_on) & (ip.t <= t_off)
    if pre_mask.sum() < 2 or post_mask.sum() < 2:
        return nan
    ip_pre = float(np.nanmedian(ip.y[pre_mask]))
    ip_post = float(np.nanmedian(ip.y[post_mask]))
    d_ip = ip_post - ip_pre
    dipdt = np.nan
    if in_mask.sum() >= 3:
        slope = np.polyfit(ip.t[in_mask], ip.y[in_mask], 1)[0]
        dipdt = float(slope)
    tau = float(-ip_pre / dipdt) if dipdt and np.isfinite(dipdt) and dipdt < 0 else np.nan
    return {
        "ip_pre": ip_pre,
        "ip_post": ip_post,
        "d_ip": d_ip,
        "d_ip_rel": d_ip / ip_pre if ip_pre else np.nan,
        "dipdt_in": dipdt,
        "tau_LR_ms": tau,
    }


def summarise(events: pd.DataFrame, t: np.ndarray, u: np.ndarray,
              params: DetectionParams) -> Dict[str, float]:
    """Condense the event table into shot-level metrics.

    Returns a flat dict (JSON/CSV friendly) with, among others:

    ``n_events``            number of detected collapses
    ``t_onset``             time of the first collapse (ms)
    ``period_mean/cv``      mean repetition period and its coefficient of variation
    ``duty_cycle``          fraction of the burst window spent collapsed
    ``depth_mean``          mean relative depth (1 = down to zero)
    ``d_ip_sum``            total plasma current lost inside collapses (kA)
    ``flux_before_onset``   volt-seconds delivered before the first collapse (mWb)
    ``instability_score``   0..1 heuristic used to rank shots
    """
    summary: Dict[str, float] = {
        "n_events": float(len(events)),
        "t_onset": float(events["t_on"].iloc[0]) if len(events) else np.nan,
        "t_last": float(events["t_off"].iloc[-1]) if len(events) else np.nan,
        "period_mean": np.nan,
        "period_std": np.nan,
        "period_cv": np.nan,
        "duty_cycle": np.nan,
        "depth_mean": np.nan,
        "duration_mean": np.nan,
        "d_ip_sum": np.nan,
        "d_ip_mean": np.nan,
        "tau_LR_median": np.nan,
        "flux_before_onset": np.nan,
        "flux_deficit_sum": np.nan,
        "u_env_at_onset": np.nan,
        "instability_score": 0.0,
        "is_candidate": 0.0,
    }
    if events.empty:
        return summary

    periods = events["period_ms"].dropna().to_numpy()
    if periods.size:
        summary["period_mean"] = float(np.mean(periods))
        summary["period_std"] = float(np.std(periods))
        summary["period_cv"] = float(np.std(periods) / np.mean(periods)) if np.mean(periods) else np.nan

    burst_span = float(events["t_off"].iloc[-1] - events["t_on"].iloc[0])
    if burst_span > 0:
        summary["duty_cycle"] = float(events["duration_ms"].sum() / burst_span)
    summary["depth_mean"] = float(events["depth"].mean())
    summary["duration_mean"] = float(events["duration_ms"].mean())
    summary["d_ip_sum"] = float(events["d_ip"].sum(skipna=True))
    summary["d_ip_mean"] = float(events["d_ip"].mean(skipna=True))
    summary["tau_LR_median"] = float(events["tau_LR_ms"].median(skipna=True))
    summary["flux_deficit_sum"] = float(events["flux_deficit_mWb"].sum())
    summary["u_env_at_onset"] = float(events["u_env"].iloc[0])

    t_onset = summary["t_onset"]
    pre = t <= t_onset
    if pre.sum() > 2:
        summary["flux_before_onset"] = float(_trapz(u[pre], t[pre]))

    summary["instability_score"] = _score(summary, events, params)
    summary["is_candidate"] = float(
        len(events) >= params.min_events
        and summary["depth_mean"] >= params.min_depth
        and (np.isnan(summary["d_ip_mean"]) or summary["d_ip_mean"] <= 0.0)
    )
    return summary


def _score(summary: Dict[str, float], events: pd.DataFrame,
           params: DetectionParams) -> float:
    """Heuristic 0..1 ranking: many, deep, regular collapses that eat current."""
    n = len(events)
    count_term = min(1.0, n / 8.0)
    depth_term = float(np.clip(summary["depth_mean"], 0.0, 1.0))
    cv = summary["period_cv"]
    regular_term = float(np.clip(1.0 - cv, 0.0, 1.0)) if np.isfinite(cv) else 0.3
    d_ip_mean = summary["d_ip_mean"]
    current_term = 1.0 if (np.isfinite(d_ip_mean) and d_ip_mean < 0) else 0.3
    return float(np.clip(
        0.35 * count_term + 0.30 * depth_term + 0.20 * regular_term + 0.15 * current_term,
        0.0, 1.0))


def detect_in_shot(shot: Shot, params: Optional[DetectionParams] = None) -> Optional[DetectionResult]:
    """Run the detector on a :class:`~golem_instability.shotdata.Shot`.

    Returns ``None`` when the shot has no usable loop voltage trace.
    """
    params = params or DetectionParams()
    u_loop = shot.signal("U_loop")
    if u_loop is None or len(u_loop) < 32:
        logger.info("shot %d: no U_loop trace", shot.shot_no)
        return None
    ip = shot.signal("Ip")
    window = analysis_window(shot, u_loop, params)
    result = detect_collapses(u_loop, ip, window, params, shot.shot_no)
    return result


def result_to_row(shot: Shot, result: Optional[DetectionResult]) -> Dict[str, object]:
    """One flat record per shot, for the cross-shot scan table."""
    row: Dict[str, object] = {"shot_no": shot.shot_no}
    try:
        row.update({k: v for k, v in shot.meta.items() if k != "shot_no"})
    except Exception:  # pragma: no cover - metadata is best effort
        pass
    row["diagnostics"] = ",".join(shot.active_diagnostics)
    if result is None:
        row.update({k: np.nan for k in ("n_events", "instability_score", "is_candidate")})
        return row
    row.update(result.summary)
    return row
