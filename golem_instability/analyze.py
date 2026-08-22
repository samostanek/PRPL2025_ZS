"""Phenomenological analysis of the loop-voltage collapses.

Two layers:

* **Metrics** - model-free numbers extracted from whatever diagnostics ran:
  event-triggered averages, MHD precursor growth, recycling flashes, the
  effective plasma resistance during the collapse, machine-circuit coherence,
  digitiser clipping, Mirnov mode numbers.
* **Hypothesis scoring** - each candidate mechanism is expressed as a set of
  predictions about those metrics and scored ``+1`` (supported) to ``-1``
  (contradicted), or left inconclusive when the deciding diagnostic did not
  run. The point is not that the computer decides the physics, but that every
  mechanism is made falsifiable by a specific measured number.

Response metrics use robust statistics throughout: a "jump" is measured both
relative to the pre-event level and in units of the pre-event scatter
(median absolute deviation), so that a quiet, noisy channel cannot fake a
large relative response.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sps

from .detect import DetectionResult
from .shotdata import Shot, Signal

logger = logging.getLogger(__name__)

#: Plasma self-inductance assumed when converting dIp/dt into a resistance.
#: GOLEM: R0 = 0.4 m, a = 0.085 m -> L_p = mu0*R0*(ln(8R0/a) - 1.75 + li/2) ~ 1.2 uH.
L_PLASMA_H: float = 1.2e-6

#: Resistance of the GOLEM vacuum vessel to the toroidal current. The published
#: plasma current is derived from the Rogowski signal by subtracting the chamber
#: current, estimated resistively as ``I_ch = U_loop / R_chamber``, so U_loop and
#: Ip are *not* independent measurements - see :func:`chamber_correction_test`.
R_CHAMBER_OHM: float = 9.7e-3

#: Order-of-magnitude inductance of that same chamber current path. With
#: ``R_CHAMBER_OHM`` it sets the time below which the purely resistive chamber
#: correction stops being valid (``L/R ~ 0.12 ms``).
L_CHAMBER_H: float = 1.2e-6

#: Poloidal angles assumed for the 16 Mirnov ring coils (equispaced).
RING_ANGLES: np.ndarray = 2 * np.pi * np.arange(16) / 16.0

_trapz = getattr(np, "trapezoid", None) or np.trapz


def _mad(values: np.ndarray) -> float:
    """Median absolute deviation scaled to a Gaussian standard deviation."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return float("nan")
    return float(1.4826 * np.median(np.abs(finite - np.median(finite))))


# --------------------------------------------------------------------------
# Event triggered averaging
# --------------------------------------------------------------------------
@dataclass
class ConditionalAverage:
    """Event-triggered average of one signal.

    Attributes:
        key: Signal key.
        lag_ms: Lag axis relative to the collapse onset.
        mean: Mean over events.
        std: Standard deviation over events.
        n: Number of events that contributed.
        unit: Physical unit.
        label: Legend label.
    """

    key: str
    lag_ms: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n: int
    unit: str = ""
    label: str = ""


def event_triggered_average(sig: Optional[Signal], event_times: Sequence[float],
                            pre_ms: float = 0.6, post_ms: float = 1.2,
                            n_lags: int = 601,
                            normalise: bool = False) -> Optional[ConditionalAverage]:
    """Average ``sig`` in a window locked to each collapse onset.

    Args:
        sig: Signal to average.
        event_times: Collapse onset times [ms].
        pre_ms / post_ms: Window around each onset.
        n_lags: Number of lag samples.
        normalise: Subtract each realisation's own pre-event mean, so the
            shape rather than the absolute level is averaged.
    """
    if sig is None or len(sig) < 4 or len(event_times) == 0:
        return None
    lag = np.linspace(-pre_ms, post_ms, n_lags)
    stack: List[np.ndarray] = []
    for t_on in event_times:
        piece = np.interp(t_on + lag, sig.t, sig.y, left=np.nan, right=np.nan)
        if np.isnan(piece).mean() > 0.2:
            continue
        if normalise:
            pre = piece[lag < 0]
            if np.isfinite(pre).any():
                piece = piece - np.nanmean(pre)
        stack.append(piece)
    if not stack:
        return None
    arr = np.vstack(stack)
    return ConditionalAverage(sig.key, lag, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0),
                              arr.shape[0], sig.unit, sig.label)


def conditional_averages(signals: Dict[str, Signal], event_times: Sequence[float],
                         **kwargs) -> Dict[str, ConditionalAverage]:
    """Event-triggered averages for a whole set of signals."""
    out: Dict[str, ConditionalAverage] = {}
    for key, sig in signals.items():
        avg = event_triggered_average(sig, event_times, **kwargs)
        if avg is not None:
            out[key] = avg
    return out


def _event_windows(events: pd.DataFrame, default_gap_ms: float = 2.0,
                   lead_ms: float = 1.0, quiet_frac: float = 0.5
                   ) -> List[Tuple[float, float, float, float, float]]:
    """Per-event quiet/lead windows that never reach into the previous event.

    Repetitive collapses can sit closer together than a fixed analysis window,
    in which case a naive baseline would be contaminated by the tail of the
    previous event and every signal would appear to "lead" the collapse. For
    each event the gap since the previous one is measured and both the quiet
    (baseline) window and the lead (precursor) window are shrunk to fit inside
    it.

    Returns:
        ``(quiet_t0, quiet_t1, lead_t0, t_on, t_off)`` per event.
    """
    out: List[Tuple[float, float, float, float, float]] = []
    prev_off = None
    for _, ev in events.iterrows():
        t_on, t_off = float(ev["t_on"]), float(ev["t_off"])
        gap_start = prev_off if prev_off is not None else t_on - default_gap_ms
        gap = max(t_on - gap_start, 1e-3)
        lead = float(np.clip(0.5 * gap, 0.02, lead_ms))
        quiet_t0 = gap_start
        quiet_t1 = gap_start + max(quiet_frac * (gap - lead), 1e-3)
        out.append((quiet_t0, quiet_t1, t_on - lead, t_on, t_off))
        prev_off = t_off
    return out


# --------------------------------------------------------------------------
# Generic robust response of one signal to the collapses
# --------------------------------------------------------------------------
@dataclass
class Response:
    """How one signal behaves around the collapses.

    Attributes:
        rel: Median relative excursion ``(peak - baseline)/|baseline|``.
        z: Median excursion in units of the pre-event scatter (robust sigma).
        peak_lag_ms: Median time of the peak relative to the collapse onset.
        onset_lag_ms: Median time of the first 3-sigma crossing relative to the
            collapse onset. **Negative means the signal leads the collapse**,
            which is the key cause-versus-consequence discriminator.
        n: Number of events used.
    """

    rel: float = float("nan")
    z: float = float("nan")
    peak_lag_ms: float = float("nan")
    onset_lag_ms: float = float("nan")
    n: int = 0

    def as_dict(self, prefix: str) -> Dict[str, float]:
        return {f"{prefix}_rel": self.rel, f"{prefix}_z": self.z,
                f"{prefix}_peak_lag_ms": self.peak_lag_ms,
                f"{prefix}_onset_lag_ms": self.onset_lag_ms}


def signal_response(sig: Optional[Signal], events: pd.DataFrame,
                    post_ms: float = 0.6, lead_ms: float = 1.0) -> Response:
    """Robust per-event excursion of ``sig`` around the collapses.

    Baseline and scatter come from a quiet window inside the gap that precedes
    the event (see :func:`_event_windows`), so a genuine *precursor* rise is
    measured against undisturbed plasma instead of being absorbed into the
    baseline, and a neighbouring event cannot masquerade as one.
    """
    if sig is None or len(sig) < 8 or events.empty:
        return Response()
    rels, zs, peak_lags, onset_lags = [], [], [], []
    for quiet_t0, quiet_t1, lead_t0, t_on, t_off in _event_windows(events, lead_ms=lead_ms):
        quiet = sig.y[(sig.t >= quiet_t0) & (sig.t < quiet_t1)]
        window = (sig.t >= lead_t0) & (sig.t <= t_off + post_ms)
        seg, seg_t = sig.y[window], sig.t[window]
        if quiet.size < 5 or seg.size < 5:
            continue
        base = float(np.median(quiet))
        scatter = _mad(quiet)
        peak_idx = int(np.nanargmax(np.abs(seg - base)))
        peak = float(seg[peak_idx])
        if abs(base) > 1e-12:
            rels.append((peak - base) / abs(base))
        if np.isfinite(scatter) and scatter > 0:
            zs.append((peak - base) / scatter)
            crossings = np.flatnonzero(np.abs(seg - base) > 3 * scatter)
            if crossings.size:
                onset_lags.append(float(seg_t[crossings[0]] - t_on))
        peak_lags.append(float(seg_t[peak_idx] - t_on))
    if not peak_lags:
        return Response()
    return Response(
        rel=float(np.nanmedian(rels)) if rels else float("nan"),
        z=float(np.nanmedian(zs)) if zs else float("nan"),
        peak_lag_ms=float(np.nanmedian(peak_lags)),
        onset_lag_ms=float(np.nanmedian(onset_lags)) if onset_lags else float("nan"),
        n=len(peak_lags),
    )


def responses(signals: Dict[str, Signal], events: pd.DataFrame,
              keys: Sequence[str]) -> Dict[str, Response]:
    """Robust responses for a list of signal keys that are present."""
    return {k: signal_response(signals[k], events) for k in keys if k in signals}


# --------------------------------------------------------------------------
# Individual physics metrics
# --------------------------------------------------------------------------
def _band_envelope(sig: Signal, f_lo_khz: float = 2.0, f_hi_khz: float = 100.0) -> Signal:
    """Band-pass the signal and return its analytic-signal envelope."""
    fs = sig.fs_khz
    nyq = 0.5 * fs
    lo = max(f_lo_khz / nyq, 1e-4)
    hi = min(f_hi_khz / nyq, 0.95)
    if not (0 < lo < hi < 1) or len(sig) < 64:
        return Signal(sig.key, sig.t, np.abs(sig.y - np.mean(sig.y)),
                      sig.unit, sig.label, sig.group)
    b, a = sps.butter(4, [lo, hi], btype="band")
    filtered = sps.filtfilt(b, a, sig.y - np.mean(sig.y))
    return Signal(sig.key + "_env", sig.t, np.abs(sps.hilbert(filtered)),
                  sig.unit, sig.label + " env", sig.group)


def _ring_signals(signals: Dict[str, Signal]) -> List[Tuple[int, Signal]]:
    """Mirnov ring coils present, as ``(index, signal)`` sorted by coil number."""
    out = []
    for key, sig in signals.items():
        if key.startswith("mhd_"):
            try:
                out.append((int(key.split("_")[1]) - 1, sig))
            except ValueError:
                continue
    return sorted(out)


def mhd_precursor(signals: Dict[str, Signal], events: pd.DataFrame,
                  pre_ms: float = 0.5, quiet_ms: float = 2.0) -> Dict[str, float]:
    """Growth of Mirnov activity in the window just before each collapse.

    Returns ``precursor_ratio`` (band-pass envelope in ``[-pre_ms, 0)`` over a
    quiet reference window further back), its spread, and the coil count.
    A tearing-mode-driven crash must show ``precursor_ratio`` well above 1.
    """
    coils = _ring_signals(signals)
    if not coils or events.empty:
        return {"precursor_ratio": np.nan, "precursor_ratio_std": np.nan, "n_coils": 0.0}
    ratios: List[float] = []
    windows = _event_windows(events, lead_ms=pre_ms, default_gap_ms=quiet_ms + pre_ms)
    for _, sig in coils:
        env = _band_envelope(sig)
        for quiet_t0, quiet_t1, lead_t0, t_on, _ in windows:
            pre = env.y[(env.t >= lead_t0) & (env.t < t_on)]
            quiet = env.y[(env.t >= quiet_t0) & (env.t < quiet_t1)]
            if pre.size > 3 and quiet.size > 3 and np.nanmedian(quiet) > 0:
                ratios.append(float(np.nanmedian(pre) / np.nanmedian(quiet)))
    if not ratios:
        return {"precursor_ratio": np.nan, "precursor_ratio_std": np.nan,
                "n_coils": float(len(coils))}
    return {"precursor_ratio": float(np.nanmedian(ratios)),
            "precursor_ratio_std": float(np.nanstd(ratios)),
            "n_coils": float(len(coils))}


def resistance_test(result: DetectionResult, ip: Optional[Signal],
                    l_plasma: float = L_PLASMA_H, core_frac: float = 0.6) -> Dict[str, float]:
    """Compare the plasma resistance during and before the collapse.

    Inside the collapse the circuit equation for the plasma ring is
    ``U_in = R_in Ip + L dIp/dt``, so with the measured ``U_in`` and the fitted
    ``dIp/dt`` (taken over the middle ``core_frac`` of the event, away from the
    edges) the in-event resistance follows as
    ``R_in = (U_in - L dIp/dt) / Ip``. Before the event the quasi-steady ohmic
    balance gives ``R_pre = U_env / Ip``.

    The ratio separates the two families of explanation:

    * ``R_in / R_pre ~ 1`` - the plasma is unchanged and merely lost its drive
      (circuit-side cause; the current then freewheels through ``L/R``).
    * ``R_in / R_pre >> 1`` - the plasma cooled or disrupted (plasma-side cause).
    * ``R_in / R_pre << 1`` - the resistance collapsed, e.g. a runaway-carried
      current.

    ``R_in`` scales linearly with the assumed ``l_plasma``; ``R_pre`` does not.
    """
    events = result.events
    nan = {"R_in_mOhm": np.nan, "R_pre_mOhm": np.nan, "R_ratio": np.nan,
           "tau_LR_est_ms": np.nan}
    if ip is None or events.empty:
        return nan
    r_in, r_pre = [], []
    for _, ev in events.iterrows():
        t_on, t_off = float(ev["t_on"]), float(ev["t_off"])
        span = t_off - t_on
        margin = 0.5 * (1.0 - core_frac) * span
        core = (ip.t >= t_on + margin) & (ip.t <= t_off - margin)
        if core.sum() < 3:
            continue
        slope_kA_per_ms = float(np.polyfit(ip.t[core], ip.y[core], 1)[0])
        ip_mid_A = float(np.median(ip.y[core])) * 1e3
        if ip_mid_A <= 0:
            continue
        u_core = result.u[(result.t >= t_on + margin) & (result.t <= t_off - margin)]
        u_in = float(np.median(u_core)) if u_core.size else 0.0
        didt_A_per_s = slope_kA_per_ms * 1e6      # kA/ms -> A/s
        r_in.append((u_in - l_plasma * didt_A_per_s) / ip_mid_A)
        ip_pre_A = float(ev["ip_pre"]) * 1e3
        if np.isfinite(ip_pre_A) and ip_pre_A > 0 and np.isfinite(ev["u_env"]):
            r_pre.append(float(ev["u_env"]) / ip_pre_A)
    if not r_in or not r_pre:
        return nan
    r_in_m = float(np.nanmedian(r_in) * 1e3)
    r_pre_m = float(np.nanmedian(r_pre) * 1e3)
    tau = l_plasma / (np.nanmedian(r_in)) * 1e3 if np.nanmedian(r_in) > 0 else np.nan
    return {"R_in_mOhm": r_in_m, "R_pre_mOhm": r_pre_m,
            "R_ratio": r_in_m / r_pre_m if r_pre_m else np.nan,
            "tau_LR_est_ms": float(tau)}


def chamber_current_model(t_ms: np.ndarray, u_loop: np.ndarray,
                          r_chamber: float = R_CHAMBER_OHM,
                          l_chamber: float = L_CHAMBER_H) -> np.ndarray:
    """Chamber current from an L/R circuit driven by the loop voltage.

    The standard GOLEM correction assumes ``I_ch = U_loop / R_chamber``, which
    is only valid for changes slow compared with ``L/R ~ 0.12 ms``. This
    integrates ``L dI/dt = U - R I`` instead, exactly over each sample step, and
    returns the current in kA.
    """
    tau_ms = l_chamber / r_chamber * 1e3
    out = np.zeros_like(t_ms, dtype=float)
    current = 0.0
    steps = np.diff(t_ms, prepend=t_ms[0])
    for i in range(1, t_ms.size):
        if not np.isfinite(u_loop[i]):
            out[i] = current
            continue
        decay = np.exp(-max(steps[i], 0.0) / tau_ms)
        current = current * decay + (u_loop[i] / r_chamber) * (1.0 - decay)
        out[i] = current
    return out / 1e3


def _persistent_step(sig: Optional[Signal], events: pd.DataFrame,
                     t_end: Optional[float] = None) -> Dict[str, float]:
    """Separate the transient excursion at an event from the step that remains.

    For each event the deepest excursion just after the onset (``dip``) is
    compared with the level late in the gap before the next event (``step``).
    A processing transient recovers, so ``recovery`` approaches 1; a real loss
    of current persists, so ``recovery`` approaches 0.
    """
    nan = {"dip": np.nan, "step": np.nan, "recovery": np.nan}
    if sig is None or len(sig) < 16 or events.empty:
        return nan
    t_last = t_end if t_end is not None else float(sig.t[-1])
    starts = list(events["t_on"]) + [t_last]
    dips, steps = [], []
    for i, (t_on, t_off) in enumerate(zip(events["t_on"], events["t_off"])):
        pre = sig.y[(sig.t >= t_on - 0.3) & (sig.t < t_on)]
        during = sig.y[(sig.t >= t_on) & (sig.t <= t_off + 0.3)]
        next_on = starts[i + 1]
        gap = max(next_on - t_off, 1e-3)
        settle = sig.y[(sig.t >= next_on - 0.35 * gap) & (sig.t < next_on - 0.05 * gap)]
        if pre.size < 3 or during.size < 3:
            continue
        base = float(np.nanmedian(pre))
        dips.append(float(np.nanmin(during) - base))
        if settle.size >= 3:
            steps.append(float(np.nanmedian(settle) - base))
    if not dips:
        return nan
    dip = float(np.nanmedian(dips))
    step = float(np.nanmedian(steps)) if steps else np.nan
    recovery = float(1.0 - step / dip) if (np.isfinite(step) and dip != 0) else np.nan
    return {"dip": dip, "step": step, "recovery": recovery}


def chamber_correction_test(result: DetectionResult, signals: Dict[str, Signal]
                            ) -> Dict[str, float]:
    """Is the current staircase real, or an artefact of the chamber correction?

    On GOLEM the Rogowski coil around the vessel sees the plasma current *plus*
    the current induced in the stainless-steel chamber, and the published
    ``Ip`` is obtained by subtracting the latter, estimated resistively as
    ``I_ch = U_loop / R_chamber`` with ``R_chamber = 9.7 mOhm``. Two
    consequences matter here:

    1. ``U_loop`` and ``Ip`` are not independent, so "the loop voltage collapsed
       *and* the current stepped down" is not automatically two measurements.
    2. The chamber path has its own ``L/R ~ 0.12 ms``. On transients faster than
       that the real chamber current cannot follow ``U_loop/R_chamber``, so the
       correction over- or under-shoots and writes a step into ``Ip`` that the
       plasma never had.

    Reported:
        ``chamber_swing_kA``            size of the correction change per collapse
        ``dip_over_chamber_swing``      current step divided by that swing; ~1 is
                                        the danger zone
        ``tau_chamber_ms``              L/R of the chamber path
        ``tau_chamber_over_duration``   >1 means the resistive correction is
                                        invalid for these transients
        ``d_irog_mean``                 step of the *reconstructed uncorrected*
                                        total toroidal current
        ``R_chamber_implied_mOhm``      resistance implied by the shot's own
                                        ``Ich`` trace, to confirm the correction
                                        really is purely resistive
        ``ip_dip`` / ``ip_step``        transient and persistent parts of the
                                        current excursion
        ``ip_recovery``                 1 = fully recovered transient, 0 = a step
                                        that stays
        ``irog_dip`` / ``irog_step``    the same for the reconstructed
                                        uncorrected current
        ``ip_dyn_*``                    the same after *re-doing* the chamber
                                        correction with an inductive model
                                        (:func:`chamber_current_model`)
        ``dip_ratio_dyn_pub``           excursion with the inductive correction
                                        divided by the published one; much below
                                        1 means the staircase is written by the
                                        resistive correction, near 1 that it is
                                        real plasma current
    """
    events = result.events
    out: Dict[str, float] = {"chamber_swing_kA": np.nan, "dip_over_chamber_swing": np.nan,
                             "tau_chamber_ms": L_CHAMBER_H / R_CHAMBER_OHM * 1e3,
                             "tau_chamber_over_duration": np.nan,
                             "d_irog_mean": np.nan, "R_chamber_implied_mOhm": np.nan,
                             "ip_dip": np.nan, "ip_step": np.nan, "ip_recovery": np.nan,
                             "irog_dip": np.nan, "irog_step": np.nan,
                             "irog_recovery": np.nan, "dip_ratio_irog_ip": np.nan,
                             "ip_dyn_dip": np.nan, "ip_dyn_step": np.nan,
                             "ip_dyn_recovery": np.nan, "dip_ratio_dyn_pub": np.nan}
    if events.empty:
        return out

    ich = signals.get("Ich")
    if ich is not None and len(ich) > 16:
        u_on_ich = np.interp(ich.t, result.t, result.u, left=np.nan, right=np.nan)
        mask = np.isfinite(u_on_ich) & (np.abs(ich.y) > 0.05 * np.nanmax(np.abs(ich.y)))
        if mask.sum() > 10:
            out["R_chamber_implied_mOhm"] = float(
                np.nanmedian(u_on_ich[mask] / ich.y[mask]) * 1e3)

    swing_kA = (events["u_env"] - events["u_min"]) / R_CHAMBER_OHM / 1e3
    out["chamber_swing_kA"] = float(np.nanmedian(swing_kA))
    d_ip = events["d_ip"].abs()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["dip_over_chamber_swing"] = float(np.nanmedian(d_ip / swing_kA))
    duration = float(np.nanmean(events["duration_ms"]))
    if duration > 0:
        out["tau_chamber_over_duration"] = float(out["tau_chamber_ms"] / duration)

    ip = signals.get("Ip")
    if ip is not None and len(ip) > 16:
        u_on_ip = np.interp(ip.t, result.t, result.u, left=np.nan, right=np.nan)
        # Undo the resistive correction to recover what the Rogowski coil saw,
        # then re-apply a correction that respects the chamber's own L/R.
        i_ch_resistive = np.clip(u_on_ip, 0, None) / R_CHAMBER_OHM / 1e3
        i_ch_dynamic = chamber_current_model(ip.t, np.nan_to_num(u_on_ip))
        i_rog = Signal("I_rog", ip.t, ip.y + i_ch_resistive, "kA",
                       "reconstructed total toroidal current", "derived")
        ip_dyn = Signal("Ip_dyn", ip.t, ip.y + i_ch_resistive - i_ch_dynamic, "kA",
                        "plasma current with an inductive chamber correction", "derived")
        steps = []
        for t_on, t_off in zip(events["t_on"], events["t_off"]):
            pre = i_rog.y[(i_rog.t >= t_on - 0.3) & (i_rog.t <= t_on)]
            post = i_rog.y[(i_rog.t >= t_off) & (i_rog.t <= t_off + 0.3)]
            if pre.size > 2 and post.size > 2:
                steps.append(float(np.nanmedian(post) - np.nanmedian(pre)))
        if steps:
            out["d_irog_mean"] = float(np.nanmedian(steps))
        t_end = float(result.t[-1]) if result.t.size else None
        ip_parts = _persistent_step(ip, events, t_end)
        irog_parts = _persistent_step(i_rog, events, t_end)
        dyn_parts = _persistent_step(ip_dyn, events, t_end)
        out.update({f"ip_{k}": v for k, v in ip_parts.items()})
        out.update({f"irog_{k}": v for k, v in irog_parts.items()})
        out.update({f"ip_dyn_{k}": v for k, v in dyn_parts.items()})
        if np.isfinite(ip_parts["dip"]) and ip_parts["dip"] != 0:
            out["dip_ratio_irog_ip"] = float(irog_parts["dip"] / ip_parts["dip"])
            out["dip_ratio_dyn_pub"] = float(dyn_parts["dip"] / ip_parts["dip"])
    return out


def machine_coherence(signals: Dict[str, Signal], events: pd.DataFrame,
                      keys: Sequence[str] = ("Bt", "Ich")) -> Dict[str, float]:
    """Do other machine circuits show a synchronised feature at each collapse?

    A step in the toroidal-field or chamber-current channel locked to the
    collapse points at facility-wide switching or pickup rather than at
    something the plasma did. Reported in units of the pre-event scatter of
    the event-triggered average.
    """
    out: Dict[str, float] = {}
    if events.empty:
        return out
    for key in keys:
        sig = signals.get(key)
        if sig is None:
            continue
        avg = event_triggered_average(sig, events["t_on"].to_numpy(), normalise=True)
        if avg is None:
            continue
        pre = avg.mean[avg.lag_ms < -0.1]
        during = avg.mean[(avg.lag_ms >= 0) & (avg.lag_ms < 0.3)]
        noise = _mad(pre)
        step = float(np.nanmax(np.abs(during))) if during.size else np.nan
        out[f"{key}_coherent_sigma"] = (step / noise if np.isfinite(noise) and noise > 0
                                        else np.nan)
    # Common-mode Mirnov response: switching pickup appears on every coil in
    # phase (m = 0), a real mode does not.
    coils = _ring_signals(signals)
    if len(coils) >= 4:
        common = np.mean([sig.at(events["t_on"].to_numpy() + 0.005) for _, sig in coils], axis=0)
        per_coil = np.std([sig.at(events["t_on"].to_numpy() + 0.005) for _, sig in coils], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.abs(common) / per_coil
        out["mirnov_common_mode_ratio"] = float(np.nanmedian(ratio))
    return out


def clipping_test(result: DetectionResult) -> Dict[str, float]:
    """Is the collapsed loop voltage a real zero or a saturated digitiser?

    Returns the fraction of in-event samples sitting within 1 % of the record
    minimum (a hard rail suggests clipping), the in-event scatter and the
    floor level itself.
    """
    nan = {"rail_fraction": np.nan, "in_event_std": np.nan, "u_floor": np.nan}
    if result.events.empty:
        return nan
    mask = np.zeros_like(result.t, dtype=bool)
    for _, ev in result.events.iterrows():
        mask |= (result.t >= ev["t_on"]) & (result.t <= ev["t_off"])
    seg = result.u[mask]
    if seg.size < 5:
        return nan
    floor = float(np.nanpercentile(seg, 1))
    span = float(np.nanmax(result.u) - floor)
    tol = max(0.01 * span, 1e-6)
    return {"rail_fraction": float(np.mean(np.abs(seg - floor) < tol)),
            "in_event_std": float(np.nanstd(seg)),
            "u_floor": floor}


def density_response(signals: Dict[str, Signal], events: pd.DataFrame) -> Dict[str, float]:
    """Line-averaged density step per collapse and the shot mean density."""
    sig = signals.get("ne_lav")
    if sig is None or events.empty:
        return {"ne_step_rel": np.nan, "ne_mean": np.nan}
    steps = []
    for t_on, t_off in zip(events["t_on"], events["t_off"]):
        pre = sig.y[(sig.t >= t_on - 0.5) & (sig.t < t_on)]
        post = sig.y[(sig.t > t_off) & (sig.t <= t_off + 0.5)]
        if pre.size > 3 and post.size > 3 and abs(np.nanmedian(pre)) > 0:
            steps.append(float((np.nanmedian(post) - np.nanmedian(pre)) / np.nanmedian(pre)))
    positive = sig.y[sig.y > 0]
    return {"ne_step_rel": float(np.nanmedian(steps)) if steps else np.nan,
            "ne_mean": float(np.nanmean(positive)) if positive.size else np.nan}


def mirnov_mode_number(signals: Dict[str, Signal], events: pd.DataFrame,
                       pre_ms: float = 0.5, min_coherence: float = 0.5,
                       min_snr: float = 5.0) -> Dict[str, float]:
    """Poloidal mode number from the Mirnov ring in the pre-collapse window.

    The ring coils are assumed equispaced poloidally. The dominant pre-event
    frequency is found from the coil-averaged spectrum; the cross-coil phase at
    that frequency is unwrapped around the ring and its slope, in units of
    ``2*pi`` per full turn, is the poloidal mode number ``m``. The result is
    only reported when the phase fit is coherent and the spectral peak stands
    clear of the background, otherwise noise would produce a random ``m``.
    """
    nan = {"m_est": np.nan, "f_mode_khz": np.nan, "mode_coherence": np.nan,
           "mode_snr": np.nan}
    coils = _ring_signals(signals)
    if len(coils) < 6 or events.empty:
        return nan
    m_values, f_values, coherences, snrs = [], [], [], []
    for t_on in events["t_on"]:
        segs = []
        for _, sig in coils:
            mask = (sig.t >= t_on - pre_ms) & (sig.t < t_on)
            if mask.sum() < 64:
                segs = []
                break
            segs.append(sig.y[mask] - np.mean(sig.y[mask]))
        if not segs:
            continue
        n = min(len(s) for s in segs)
        block = np.vstack([s[:n] for s in segs])
        fs_khz = coils[0][1].fs_khz
        freqs = np.fft.rfftfreq(n, d=1.0 / fs_khz)
        spec = np.fft.rfft(block * np.hanning(n), axis=1)
        power = np.mean(np.abs(spec) ** 2, axis=0)
        band = (freqs > 1.0) & (freqs < 200.0)
        if not band.any():
            continue
        idx = int(np.argmax(np.where(band, power, 0)))
        snr = float(power[idx] / np.median(power[band]))
        phases = np.unwrap(np.angle(spec[:, idx]))
        positions = np.array([i for i, _ in coils], dtype=float)
        fit = np.polyfit(positions, phases, 1)
        residual = phases - np.polyval(fit, positions)
        coherence = float(np.exp(-np.var(residual)))
        m_values.append(fit[0] * len(RING_ANGLES) / (2 * np.pi))
        f_values.append(freqs[idx])
        coherences.append(coherence)
        snrs.append(snr)
    if not m_values:
        return nan
    coherence = float(np.nanmedian(coherences))
    snr = float(np.nanmedian(snrs))
    if coherence < min_coherence or snr < min_snr:
        return {"m_est": np.nan, "f_mode_khz": float(np.nanmedian(f_values)),
                "mode_coherence": coherence, "mode_snr": snr}
    return {"m_est": float(np.nanmedian(m_values)),
            "f_mode_khz": float(np.nanmedian(f_values)),
            "mode_coherence": coherence, "mode_snr": snr}


def ring_asymmetry(signals: Dict[str, Signal], events: pd.DataFrame,
                   pre_ms: float = 0.5, quiet_ms: float = 2.0) -> Dict[str, float]:
    """Growth of the m = 1 (column displacement) component before each collapse.

    The Mirnov coils measure ``dB/dt``; they are integrated, projected on
    ``cos`` and ``sin`` of the poloidal angle, and the amplitude of that m = 1
    component just before the collapse is compared with a quiet window. A
    position excursion running into the limiter shows up here.
    """
    coils = _ring_signals(signals)
    if len(coils) < 6 or events.empty:
        return {"ring_asymmetry_growth": np.nan}
    t_axis = coils[0][1].t
    dt = float(np.median(np.diff(t_axis)))
    integrated = []
    angles = []
    for idx, sig in coils:
        y = sig.y - np.median(sig.y[sig.t < 0]) if (sig.t < 0).any() else sig.y - np.median(sig.y)
        integrated.append(np.cumsum(y) * dt)
        angles.append(RING_ANGLES[idx % len(RING_ANGLES)])
    block = np.vstack(integrated)
    angles = np.asarray(angles)
    cos_c = (block * np.cos(angles)[:, None]).mean(axis=0)
    sin_c = (block * np.sin(angles)[:, None]).mean(axis=0)
    m1 = np.hypot(cos_c, sin_c)
    growths = []
    for quiet_t0, quiet_t1, lead_t0, t_on, _ in _event_windows(
            events, lead_ms=pre_ms, default_gap_ms=quiet_ms + pre_ms):
        pre = m1[(t_axis >= lead_t0) & (t_axis < t_on)]
        quiet = m1[(t_axis >= quiet_t0) & (t_axis < quiet_t1)]
        if pre.size > 3 and quiet.size > 3 and np.nanmedian(quiet) > 0:
            growths.append(float(np.nanmedian(pre) / np.nanmedian(quiet)))
    return {"ring_asymmetry_growth": float(np.nanmedian(growths)) if growths else np.nan}


def periodicity_spectrum(result: DetectionResult) -> Dict[str, float]:
    """Dominant repetition frequency of the collapse train from the U_loop residual."""
    nan = {"f_repeat_khz": np.nan, "repeat_peak_ratio": np.nan}
    if result.t.size < 64:
        return nan
    mask = np.isfinite(result.ratio)
    if mask.sum() < 64:
        return nan
    dt = float(np.median(np.diff(result.t)))
    x = result.ratio[mask] - np.nanmean(result.ratio[mask])
    freqs, psd = sps.welch(x, fs=1.0 / dt, nperseg=min(4096, x.size))
    band = freqs > 0.05  # ignore the slow envelope
    if not band.any():
        return nan
    idx = int(np.argmax(np.where(band, psd, 0)))
    return {"f_repeat_khz": float(freqs[idx]),
            "repeat_peak_ratio": float(psd[idx] / np.median(psd[band]))}


def staircase_metrics(result: DetectionResult, ip: Optional[Signal]) -> Dict[str, float]:
    """How much of the current decay happens inside the collapses?

    ``staircase_fraction`` is the current lost inside the events divided by the
    net change over the burst. Values at or above 1 mean the current is flat
    (or even recovering) between events and only steps down during them - a
    true staircase; values well below 1 mean the events are incidental to a
    current that was decaying anyway.
    """
    events = result.events
    if ip is None or events.empty:
        return {"staircase_fraction": np.nan, "ip_at_onset": np.nan, "ip_at_end": np.nan}
    t0 = float(events["t_on"].iloc[0])
    t1 = float(events["t_off"].iloc[-1])
    ip_start = float(np.interp(t0, ip.t, ip.y))
    ip_end = float(np.interp(t1, ip.t, ip.y))
    total = ip_end - ip_start
    in_events = float(events["d_ip"].sum(skipna=True))
    return {"staircase_fraction": float(in_events / total) if total else np.nan,
            "ip_at_onset": ip_start, "ip_at_end": ip_end}


# --------------------------------------------------------------------------
# Hypothesis scoring
# --------------------------------------------------------------------------
@dataclass
class HypothesisVerdict:
    """Outcome of confronting one mechanism with the measured metrics.

    Attributes:
        key: Short identifier (``H1`` ...).
        name: Human readable mechanism name.
        score: ``+1`` strongly supported ... ``-1`` strongly contradicted;
            ``nan`` when no deciding diagnostic was available.
        evidence: Lines of the form "metric = value (what it would mean)".
        missing: Diagnostics that would settle the case but did not run.
    """

    key: str
    name: str
    score: float
    evidence: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.score):
            return "inconclusive"
        if self.score >= 0.4:
            return "supported"
        if self.score <= -0.4:
            return "contradicted"
        return "weak/mixed"


def _combine(votes: List[float], weights: Optional[List[float]] = None) -> float:
    pairs = [(v, w) for v, w in zip(votes, weights or [1.0] * len(votes)) if np.isfinite(v)]
    if not pairs:
        return float("nan")
    total_w = sum(w for _, w in pairs)
    return float(sum(v * w for v, w in pairs) / total_w)


#: A response is only believed above this many robust sigmas.
Z_SIGNIFICANT: float = 5.0


def score_hypotheses(metrics: Dict[str, float], available: Sequence[str]) -> List[HypothesisVerdict]:
    """Turn the metric dict into a ranked list of :class:`HypothesisVerdict`."""

    def get(name: str) -> float:
        try:
            return float(metrics.get(name, np.nan))
        except (TypeError, ValueError):
            return float("nan")

    verdicts: List[HypothesisVerdict] = []

    r_ratio = get("R_ratio")
    cv = get("period_cv")
    rail = get("rail_fraction")
    prec = get("precursor_ratio")
    common_mode = get("mirnov_common_mode_ratio")
    ha_z, ha_lead = get("Halpha_z"), get("Halpha_onset_lag_ms")
    imp_z, imp_lead = get("CII_514_z"), get("CII_514_onset_lag_ms")
    rad_z, rad_lag = get("CeBr_z"), get("CeBr_peak_lag_ms")
    ne_step, ne_mean = get("ne_step_rel"), get("ne_mean")
    d_ip = get("d_ip_mean")

    # ---- H1: the transformer drive is interrupted (circuit side) -----------
    votes, weights, ev, missing = [], [], [], []
    if np.isfinite(r_ratio):
        votes.append(1.0 if r_ratio < 1.5 else -1.0)
        weights.append(2.0)
        ev.append(f"R_in/R_pre = {r_ratio:.2f} - a circuit cause leaves the plasma "
                  f"untouched, so the resistance should not change (~1)")
    if np.isfinite(cv):
        votes.append(1.0 if cv < 0.15 else (-0.5 if cv > 0.5 else 0.0))
        weights.append(1.0)
        ev.append(f"period CV = {cv:.2f} - switching electronics is clock-regular (CV < 0.15)")
    if np.isfinite(rail):
        votes.append(0.7 if rail > 0.3 else -0.2)
        weights.append(0.7)
        ev.append(f"rail fraction = {rail:.2f} - a switched-off drive holds U_loop on a flat floor")
    for key in ("Bt_coherent_sigma", "Ich_coherent_sigma"):
        value = get(key)
        if np.isfinite(value):
            votes.append(0.8 if value > 4 else -0.3)
            weights.append(0.8)
            ev.append(f"{key} = {value:.1f} - other machine circuits switching in step")
    if np.isfinite(common_mode):
        votes.append(0.8 if common_mode > 1.0 else -0.4)
        weights.append(0.8)
        ev.append(f"Mirnov common-mode ratio = {common_mode:.2f} - switching pickup is in phase "
                  f"on all coils (m = 0), a plasma mode is not")
    if np.isfinite(prec):
        votes.append(0.8 if prec < 1.5 else -0.8)
        weights.append(1.5)
        ev.append(f"MHD precursor ratio = {prec:.2f} - a circuit cause has no magnetic precursor")
    else:
        missing.append("MHDring-TM")
    if np.isfinite(ha_z):
        votes.append(0.6 if ha_z < Z_SIGNIFICANT else -0.6)
        weights.append(1.0)
        ev.append(f"H-alpha response = {ha_z:.1f} sigma - a passive plasma should not flash")
    verdicts.append(HypothesisVerdict(
        "H1", "Drive interruption in the primary circuit (thyristor commutation, "
              "current-feedback chopping, crowbar)", _combine(votes, weights), ev, missing))

    # ---- H2: repetitive minor disruptions (MHD relaxation) -----------------
    votes, weights, ev, missing = [], [], [], []
    if np.isfinite(prec):
        votes.append(1.0 if prec > 2.0 else -0.8)
        weights.append(2.0)
        ev.append(f"MHD precursor ratio = {prec:.2f} - a tearing mode must grow before each crash")
    else:
        missing.append("MHDring-TM")
    m_est = get("m_est")
    if np.isfinite(m_est):
        votes.append(0.8 if abs(abs(m_est) - 2.0) < 0.6 else 0.0)
        weights.append(1.0)
        ev.append(f"Mirnov mode number m = {m_est:.1f} at {get('f_mode_khz'):.1f} kHz "
                  f"(coherence {get('mode_coherence'):.2f}) - m = 2/n = 1 is the classic "
                  f"disruptive mode of a small ohmic tokamak")
    if np.isfinite(ha_z):
        supports = ha_z > Z_SIGNIFICANT and (not np.isfinite(ha_lead) or ha_lead > -0.05)
        votes.append(0.8 if supports else -0.5)
        weights.append(1.2)
        ev.append(f"H-alpha response = {ha_z:.1f} sigma, first rise at {ha_lead:+.3f} ms - "
                  f"the crash expels particles, so recycling must FOLLOW the collapse")
    else:
        missing.append("FastSpectrometry")
    if np.isfinite(r_ratio):
        votes.append(1.0 if r_ratio > 1.5 else -1.0)
        weights.append(2.0)
        ev.append(f"R_in/R_pre = {r_ratio:.2f} - a crash cools the plasma and raises its resistance")
    if np.isfinite(ne_step):
        votes.append(0.6 if ne_step < -0.05 else -0.2)
        weights.append(1.0)
        ev.append(f"density step = {ne_step * 100:+.0f} % per event - a crash expels particles")
    else:
        missing.append("Interferometry")
    verdicts.append(HypothesisVerdict(
        "H2", "Repetitive minor disruptions / MHD relaxation oscillation",
        _combine(votes, weights), ev, missing))

    # ---- H3: runaway-electron dominated low-resistivity phases -------------
    votes, weights, ev, missing = [], [], [], []
    if np.isfinite(rad_z):
        votes.append(1.0 if rad_z > Z_SIGNIFICANT else -0.8)
        weights.append(1.5)
        ev.append(f"scintillator response = {rad_z:.1f} sigma, peak at {rad_lag:+.3f} ms - "
                  f"runaway losses give hard-X-ray bursts at the collapse")
    else:
        missing.append("ScintillationProbes")
    if np.isfinite(r_ratio):
        votes.append(1.0 if r_ratio < 0.6 else -0.6)
        weights.append(1.5)
        ev.append(f"R_in/R_pre = {r_ratio:.2f} - a runaway-carried current has near-zero resistance")
    if np.isfinite(ne_mean):
        low = ne_mean < 1e19
        votes.append(0.5 if low else -0.5)
        weights.append(0.8)
        ev.append(f"mean density = {ne_mean:.2e} m^-3 - runaway generation needs a low-density plasma")
    else:
        missing.append("Interferometry")
    verdicts.append(HypothesisVerdict(
        "H3", "Runaway-electron dominated phases (resistance collapse and loss bursts)",
        _combine(votes, weights), ev, missing))

    # ---- H4: radiative / recycling limit cycle -----------------------------
    votes, weights, ev, missing = [], [], [], []
    if np.isfinite(imp_z):
        leads = np.isfinite(imp_lead) and imp_lead < -0.05
        votes.append(1.0 if (imp_z > Z_SIGNIFICANT and leads) else -0.7)
        weights.append(2.0)
        ev.append(f"C II response = {imp_z:.1f} sigma, first rise at {imp_lead:+.3f} ms - "
                  f"a radiative collapse requires the impurity radiation to LEAD the "
                  f"loop-voltage drop, not follow it")
    else:
        missing.append("FastSpectrometry")
    period = get("period_mean")
    if np.isfinite(period):
        votes.append(0.5 if period > 0.5 else -0.5)
        weights.append(0.6)
        ev.append(f"period = {period:.2f} ms - radiative/recycling cycles are slow (>~0.5 ms), "
                  f"set by the energy confinement and refuelling times")
    if np.isfinite(r_ratio):
        votes.append(0.8 if r_ratio > 1.5 else -0.8)
        weights.append(1.2)
        ev.append(f"R_in/R_pre = {r_ratio:.2f} - radiative cooling raises the resistance")
    verdicts.append(HypothesisVerdict(
        "H4", "Impurity-radiation / recycling limit cycle (MARFE-like thermal relaxation)",
        _combine(votes, weights), ev, missing))

    # ---- H5: volt-second exhaustion / transformer core saturation ----------
    votes, weights, ev, missing = [], [], [], []
    flux = get("flux_before_onset")
    if np.isfinite(flux):
        ev.append(f"flux delivered before onset = {flux:.1f} mWb - the decisive test is "
                  f"cross-shot: if the onset is set by the core, this number clusters "
                  f"tightly while the onset TIME does not (see the scan-level test)")
    onset = get("t_onset")
    if np.isfinite(onset):
        votes.append(0.4 if onset > 12.0 else -0.4)
        weights.append(1.0)
        ev.append(f"onset at t = {onset:.1f} ms - late in the discharge, as expected when the "
                  f"core approaches saturation")
    if np.isfinite(r_ratio):
        votes.append(0.6 if r_ratio < 1.5 else -0.6)
        weights.append(1.0)
        ev.append(f"R_in/R_pre = {r_ratio:.2f} - saturation removes the drive without "
                  f"touching the plasma")
    verdicts.append(HypothesisVerdict(
        "H5", "Volt-second exhaustion / iron-core saturation of the transformer",
        _combine(votes, weights), ev, missing))

    # ---- H6: position instability / limiter contact ------------------------
    votes, weights, ev, missing = [], [], [], []
    asym = get("ring_asymmetry_growth")
    if np.isfinite(asym):
        votes.append(1.0 if asym > 1.5 else -0.5)
        weights.append(1.5)
        ev.append(f"Mirnov m = 1 (displacement) growth = {asym:.2f} before each event")
    else:
        missing.append("MHDring-TM (its m = 1 asymmetry tracks the column position)")
    if np.isfinite(imp_z) and np.isfinite(ha_z):
        impurity_first = (np.isfinite(imp_lead) and np.isfinite(ha_lead) and imp_lead < ha_lead
                          and imp_z > Z_SIGNIFICANT)
        votes.append(0.6 if impurity_first else -0.3)
        weights.append(1.0)
        ev.append(f"C II first rise {imp_lead:+.3f} ms vs H-alpha {ha_lead:+.3f} ms - fresh "
                  f"limiter contact releases wall impurities before the hydrogen flash")
    verdicts.append(HypothesisVerdict(
        "H6", "Vertical/radial position excursion with limiter contact",
        _combine(votes, weights), ev, missing))

    # ---- H7: diagnostic artefact -------------------------------------------
    votes, weights, ev, missing = [], [], [], []
    in_std = get("in_event_std")
    u_floor = get("u_floor")
    if np.isfinite(rail) and np.isfinite(in_std):
        artefact = rail > 0.8 and in_std < 0.02 * max(get("u_env_at_onset"), 1e-9)
        votes.append(0.8 if artefact else -0.6)
        weights.append(1.0)
        ev.append(f"rail fraction = {rail:.2f}, in-event scatter = {in_std:.3f} V, "
                  f"floor = {u_floor:.3f} V - a clipped ADC gives a dead-flat rail")
    swing_ratio = get("dip_over_chamber_swing")
    tau_ratio = get("tau_chamber_over_duration")
    if np.isfinite(swing_ratio):
        suspicious = 0.5 < swing_ratio < 2.0
        votes.append(0.9 if suspicious else -0.6)
        weights.append(2.0)
        ev.append(f"current step / chamber-correction swing = {swing_ratio:.2f} "
                  f"(chamber swing {get('chamber_swing_kA'):.2f} kA) - values near 1 mean "
                  f"the I_p staircase could be written by the U_loop-based chamber "
                  f"correction rather than by the plasma")
    if np.isfinite(tau_ratio):
        votes.append(0.6 if tau_ratio > 1.0 else -0.4)
        weights.append(1.2)
        ev.append(f"chamber L/R ({get('tau_chamber_ms'):.2f} ms) / collapse duration = "
                  f"{tau_ratio:.1f} - above 1 the purely resistive chamber correction "
                  f"cannot follow the transient and mis-corrects I_p")
    dip_ratio = get("dip_ratio_dyn_pub")
    if np.isfinite(dip_ratio):
        votes.append(0.9 if dip_ratio < 0.4 else -0.9)
        weights.append(2.5)
        ev.append(f"current excursion after redoing the chamber correction with an "
                  f"inductive model, relative to the published one = {dip_ratio:.2f} "
                  f"({get('ip_dyn_dip'):+.3f} kA vs {get('ip_dip'):+.3f} kA) - near 0 the "
                  f"staircase is written by the resistive correction, near 1 it is real "
                  f"plasma current")
    recovery = get("ip_recovery")
    if np.isfinite(recovery):
        votes.append(0.6 if recovery > 0.7 else -0.6)
        weights.append(1.2)
        ev.append(f"recovered fraction of the current excursion = {recovery:.2f} - a "
                  f"processing transient recovers completely, a real loss does not")
    elif np.isfinite(d_ip):
        votes.append(-0.5 if d_ip < 0 else 0.3)
        weights.append(1.0)
        ev.append(f"mean current step = {d_ip:+.3f} kA per event - note that on GOLEM the "
                  f"published I_p already contains a U_loop-derived chamber correction, "
                  f"so this is not an independent measurement")
    verdicts.append(HypothesisVerdict(
        "H7", "Measurement or processing artefact (digitiser clipping, or the "
              "U_loop-derived chamber-current correction writing the I_p staircase)",
        _combine(votes, weights), ev, missing))

    # Only report a diagnostic as "missing" when it genuinely did not run; a
    # metric can also be unavailable because it could not be computed.
    present = set(available)
    for verdict in verdicts:
        verdict.missing = sorted({m for m in verdict.missing
                                  if m.split()[0] not in present})

    verdicts.sort(key=lambda v: -(v.score if np.isfinite(v.score) else -9))
    return verdicts


#: Signals whose robust response is measured for every candidate shot.
RESPONSE_KEYS: Tuple[str, ...] = (
    "Halpha", "Hbeta", "CII_514", "OI_777", "NII_568", "HeI_447", "L_broad",
    "CeBr", "YAP", "DDRE_1", "ne_lav", "U_diam_in",
)


def analyse_shot(shot: Shot, result: DetectionResult,
                 signals: Optional[Dict[str, Signal]] = None) -> Dict[str, object]:
    """Full phenomenological analysis of one shot.

    Args:
        shot: The shot.
        result: Output of :func:`~golem_instability.detect.detect_in_shot`.
        signals: Pre-loaded signals; loaded from the shot when omitted.

    Returns:
        Dict with ``metrics`` (flat numbers), ``verdicts``
        (:class:`HypothesisVerdict` list), ``responses``, ``conditional``
        (event-triggered averages), ``signals`` and the ``result`` itself.
    """
    if signals is None:
        signals = shot.signals(shot.available_signals())
    events = result.events
    metrics: Dict[str, float] = dict(result.summary)
    metrics.update(mhd_precursor(signals, events))
    metrics.update(resistance_test(result, signals.get("Ip")))
    metrics.update(machine_coherence(signals, events))
    metrics.update(chamber_correction_test(result, signals))
    metrics.update(clipping_test(result))
    metrics.update(density_response(signals, events))
    metrics.update(mirnov_mode_number(signals, events))
    metrics.update(ring_asymmetry(signals, events))
    metrics.update(periodicity_spectrum(result))
    metrics.update(staircase_metrics(result, signals.get("Ip")))

    resp = responses(signals, events, RESPONSE_KEYS)
    for key, response in resp.items():
        metrics.update(response.as_dict(key))

    interesting = [k for k in ("U_loop", "Ip", "ne_lav", "Halpha", "CII_514", "OI_777",
                               "CeBr", "Bt", "mhd_01", "mhd_05", "mhd_09", "mhd_13")
                   if k in signals]
    conditional = conditional_averages(
        {k: signals[k] for k in interesting},
        events["t_on"].to_numpy() if not events.empty else [], normalise=True)
    verdicts = score_hypotheses(metrics, shot.active_diagnostics)
    return {"shot_no": shot.shot_no, "metrics": metrics, "verdicts": verdicts,
            "responses": resp, "conditional": conditional, "signals": signals,
            "result": result}


def volt_second_test(scan: pd.DataFrame) -> Dict[str, float]:
    """Cross-shot test of the volt-second/core-saturation hypothesis (H5).

    If the onset is set by the transformer running out of flux, the flux
    delivered before the first collapse clusters much more tightly across shots
    than the onset time does. Compares the two coefficients of variation.
    """
    needed = {"flux_before_onset", "t_onset"}
    if not needed.issubset(scan.columns):
        return {"flux_cv": np.nan, "onset_cv": np.nan, "flux_clustering_ratio": np.nan, "n": 0.0}
    sub = scan.dropna(subset=list(needed))
    if len(sub) < 3:
        return {"flux_cv": np.nan, "onset_cv": np.nan, "flux_clustering_ratio": np.nan,
                "n": float(len(sub))}
    flux_cv = float(np.std(sub["flux_before_onset"]) / np.mean(sub["flux_before_onset"]))
    onset_cv = float(np.std(sub["t_onset"]) / np.mean(sub["t_onset"]))
    return {"flux_cv": flux_cv, "onset_cv": onset_cv,
            "flux_clustering_ratio": float(onset_cv / flux_cv) if flux_cv else np.nan,
            "n": float(len(sub))}
