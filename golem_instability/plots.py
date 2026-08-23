"""Figures for the loop-voltage instability study.

Design notes
------------
Every figure is a stack of **small multiples sharing one time axis** rather
than a multi-scale overlay: the diagnostics differ by orders of magnitude in
units, and a second y-axis would make the comparison unreadable. Each panel
carries a single series, named directly on its own axis label, so identity is
never carried by colour alone. The categorical hues are the validated
colour-blind-safe order (worst adjacent CVD dE 9.1, normal-vision dE 19.6).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import signal as sps

from .analyze import ConditionalAverage
from .detect import DetectionResult
from .shotdata import Shot, Signal

logger = logging.getLogger(__name__)

#: Validated categorical order (see the data-visualisation palette reference).
SERIES: List[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                     "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
EVENT_SHADE = "#e34948"

#: Colour slot per registry group, so a signal keeps its identity across figures.
GROUP_COLOR: Dict[str, str] = {
    "basic": SERIES[0],
    "derived": SERIES[6],
    "density": SERIES[2],
    "spectro": SERIES[1],
    "radiation": SERIES[3],
    "mhd": SERIES[4],
    "probe": SERIES[5],
    "energy": SERIES[7],
}


def apply_style() -> None:
    """Recessive grid, quiet axes, text in ink tokens rather than series colour."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.8,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "text.color": INK,
        "font.size": 9,
        "axes.titlesize": 10,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "figure.dpi": 130,
    })


def _shade_events(ax, events: pd.DataFrame, label_first: bool = False) -> None:
    for i, (t_on, t_off) in enumerate(zip(events.get("t_on", []), events.get("t_off", []))):
        ax.axvspan(t_on, t_off, color=EVENT_SHADE, alpha=0.13, lw=0,
                   label="collapse" if (label_first and i == 0) else None)


def _panel(ax, sig: Signal, color: Optional[str] = None, window: Optional[tuple] = None) -> None:
    if window:
        sig = sig.window(*window)
    ax.plot(sig.t, sig.y, color=color or GROUP_COLOR.get(sig.group, SERIES[0]))
    unit = f" [{sig.unit}]" if sig.unit and sig.unit != "-" else ""
    ax.set_ylabel(f"{sig.label}{unit}", fontsize=9)


def _save(fig: Figure, path: Optional[Path]) -> Figure:
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        logger.info("wrote %s", path)
        plt.close(fig)
    return fig


def overview_figure(shot: Shot, result: DetectionResult, signals: Dict[str, Signal],
                    keys: Optional[Sequence[str]] = None,
                    path: Optional[Path] = None) -> Figure:
    """Whole-discharge multi-diagnostic overview with the collapses shaded.

    The first panel adds the running envelope and the collapse threshold that
    the detector actually used, so the detection can be judged by eye.
    """
    apply_style()
    keys = [k for k in (keys or ("U_loop", "Ip", "ne_lav", "Halpha", "CII_514",
                                 "CeBr", "mhd_01", "Bt")) if k in signals]
    if "U_loop" not in keys and "U_loop" in signals:
        keys = ["U_loop"] + list(keys)
    window = (float(result.t[0]), float(result.t[-1])) if result.t.size else None

    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 1.55 * len(keys)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        _panel(ax, signals[key], window=window)
        _shade_events(ax, result.events, label_first=(ax is axes[0]))
    top = axes[0]
    if result.t.size:
        top.plot(result.t, result.u_env, color=INK_MUTED, lw=1.0, ls="--",
                 label="running envelope")
        top.plot(result.t, result.params.enter_ratio * result.u_env, color=EVENT_SHADE,
                 lw=0.9, ls=":", label="collapse threshold")
        top.legend(loc="upper right", fontsize=8, ncols=3)
    meta = shot.meta if shot else {}
    axes[-1].set_xlabel("time [ms]")
    fig.suptitle(f"GOLEM #{result.shot_no} - {result.n_events} loop-voltage collapses"
                 f"   ({meta.get('shot_date', '')} {meta.get('comment', '')})".rstrip(),
                 y=0.995, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def event_zoom_figure(result: DetectionResult, signals: Dict[str, Signal],
                      n_events: int = 3, pad_ms: float = 0.6,
                      keys: Optional[Sequence[str]] = None,
                      path: Optional[Path] = None) -> Figure:
    """Zoom on the first ``n_events`` collapses, one column per event."""
    apply_style()
    events = result.events.head(n_events)
    if events.empty:
        raise ValueError("no events to zoom on")
    keys = [k for k in (keys or ("U_loop", "Ip", "Halpha", "mhd_01", "CeBr"))
            if k in signals]
    fig, axes = plt.subplots(len(keys), len(events), sharex="col", sharey="row",
                             figsize=(3.4 * len(events), 1.5 * len(keys)), squeeze=False)
    for col, (_, ev) in enumerate(events.iterrows()):
        t0, t1 = ev["t_on"] - pad_ms, ev["t_off"] + pad_ms
        for row, key in enumerate(keys):
            ax = axes[row][col]
            _panel(ax, signals[key], window=(t0, t1))
            ax.axvspan(ev["t_on"], ev["t_off"], color=EVENT_SHADE, alpha=0.13, lw=0)
            if col:
                ax.set_ylabel("")
        axes[0][col].set_title(f"event {col + 1}: t = {ev['t_on']:.2f} ms, "
                               f"{ev['duration_ms'] * 1e3:.0f} us, "
                               f"dIp = {ev['d_ip']:+.2f} kA", fontsize=9)
        axes[-1][col].set_xlabel("time [ms]")
    fig.suptitle(f"GOLEM #{result.shot_no} - individual collapses", y=0.998, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def conditional_figure(conditional: Dict[str, ConditionalAverage], shot_no: int,
                       path: Optional[Path] = None) -> Figure:
    """Event-triggered averages: what every diagnostic does at a collapse.

    Each panel is the mean over all collapses with a +/-1 sigma band, aligned
    on the collapse onset (dashed line at lag 0). Signals are baseline-removed
    per event, so the panels show the *change* caused by the event.
    """
    apply_style()
    keys = list(conditional)
    if not keys:
        raise ValueError("no conditional averages to plot")
    fig, axes = plt.subplots(len(keys), 1, figsize=(7.5, 1.4 * len(keys)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        avg = conditional[key]
        color = SERIES[keys.index(key) % len(SERIES)]
        ax.fill_between(avg.lag_ms, avg.mean - avg.std, avg.mean + avg.std,
                        color=color, alpha=0.18, lw=0)
        ax.plot(avg.lag_ms, avg.mean, color=color)
        ax.axvline(0.0, color=INK_MUTED, lw=0.9, ls="--")
        ax.axhline(0.0, color=INK_MUTED, lw=0.6)
        unit = f" [{avg.unit}]" if avg.unit and avg.unit != "-" else ""
        ax.set_ylabel(f"D {avg.label}{unit}", fontsize=9)
    axes[-1].set_xlabel("lag from collapse onset [ms]")
    fig.suptitle(f"GOLEM #{shot_no} - event-triggered averages "
                 f"(n = {conditional[keys[0]].n} collapses)", y=0.997, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def statistics_figure(result: DetectionResult, ip: Optional[Signal] = None,
                      path: Optional[Path] = None) -> Figure:
    """Event statistics: repetition, depth, current step, and the I_p staircase."""
    apply_style()
    events = result.events
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    ax = axes[0][0]
    ax.plot(np.arange(1, len(events) + 1), events["period_ms"], "o-", color=SERIES[0])
    ax.set_xlabel("event number")
    ax.set_ylabel("repetition period [ms]")
    ax.set_title("period vs. event number")

    ax = axes[0][1]
    ax.hist(events["depth"].dropna(), bins=np.linspace(0, 1, 21), color=SERIES[1])
    ax.set_xlabel("relative depth  1 - U_min/U_env")
    ax.set_ylabel("events")
    ax.set_title("collapse depth")

    ax = axes[1][0]
    ax.scatter(events["duration_ms"] * 1e3, events["d_ip"], color=SERIES[2], s=28)
    ax.set_xlabel("collapse duration [us]")
    ax.set_ylabel("current step [kA]")
    ax.set_title("current lost per collapse")
    ax.axhline(0, color=INK_MUTED, lw=0.8)

    ax = axes[1][1]
    if ip is not None and result.t.size:
        window = ip.window(float(result.t[0]), float(result.t[-1]))
        ax.plot(window.t, window.y, color=SERIES[0])
        _shade_events(ax, events)
        ax.set_ylabel(r"$I_p$ [kA]")
    ax.set_xlabel("time [ms]")
    ax.set_title("plasma-current staircase")

    fig.suptitle(f"GOLEM #{result.shot_no} - collapse statistics", y=0.998, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def spectrogram_figure(result: DetectionResult, signals: Dict[str, Signal],
                       mhd_key: str = "mhd_01", path: Optional[Path] = None) -> Figure:
    """Time-frequency view of the loop voltage and one Mirnov coil.

    A tearing-mode precursor shows up as a band that brightens and chirps
    before each collapse; pure switching shows up as broadband spikes.
    """
    apply_style()
    keys = [k for k in ("U_loop", mhd_key) if k in signals]
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 3.0 * len(keys)), sharex=True)
    axes = np.atleast_1d(axes)
    window = (float(result.t[0]), float(result.t[-1])) if result.t.size else None
    for ax, key in zip(axes, keys):
        sig = signals[key].window(*window) if window else signals[key]
        fs_khz = sig.fs_khz
        nperseg = int(min(1024, max(64, len(sig) // 40)))
        freqs, times, spec = sps.spectrogram(sig.y - np.mean(sig.y), fs=fs_khz,
                                             nperseg=nperseg, noverlap=nperseg // 2)
        ax.pcolormesh(times + sig.t[0], freqs, 10 * np.log10(spec + 1e-20),
                      shading="auto", cmap="magma")
        ax.set_ylabel(f"{sig.label} freq [kHz]", fontsize=9)
        ax.set_ylim(0, min(200, freqs.max()))
        for t_on in result.events.get("t_on", []):
            ax.axvline(t_on, color="#ffffff", lw=0.7, alpha=0.6)
    axes[-1].set_xlabel("time [ms]")
    fig.suptitle(f"GOLEM #{result.shot_no} - spectrograms (white lines: collapse onsets)",
                 y=0.998, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def scan_figure(scan: pd.DataFrame, path: Optional[Path] = None) -> Figure:
    """Cross-shot view of a scan: where the instability appears and how strongly."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    have = scan.dropna(subset=["instability_score"]) if "instability_score" in scan else scan

    ax = axes[0][0]
    if not have.empty:
        ax.scatter(have["shot_no"], have["instability_score"], s=14, color=SERIES[0])
    ax.set_xlabel("shot number")
    ax.set_ylabel("instability score")
    ax.set_title("candidates across the scanned range")

    ax = axes[0][1]
    if "n_events" in have:
        ax.hist(have["n_events"].dropna(), bins=20, color=SERIES[1])
    ax.set_xlabel("collapses per shot")
    ax.set_ylabel("shots")
    ax.set_title("event count")

    ax = axes[1][0]
    if {"t_onset", "flux_before_onset"}.issubset(have.columns):
        ax.scatter(have["t_onset"], have["flux_before_onset"], s=18, color=SERIES[2])
    ax.set_xlabel("onset time [ms]")
    ax.set_ylabel("flux delivered before onset [mWb]")
    ax.set_title("volt-second test (H5): flat = flux-limited onset")

    ax = axes[1][1]
    if {"period_mean", "depth_mean"}.issubset(have.columns):
        ax.scatter(have["period_mean"], have["depth_mean"], s=18, color=SERIES[3])
    ax.set_xlabel("mean repetition period [ms]")
    ax.set_ylabel("mean depth")
    ax.set_title("character of the oscillation")

    fig.suptitle("Cross-shot scan for repetitive loop-voltage collapses", y=0.998, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def figure_set(shot: Shot, analysis: Dict[str, object], outdir: Path) -> List[Path]:
    """Write the standard figure set for one shot and return the paths."""
    outdir = Path(outdir)
    result: DetectionResult = analysis["result"]          # type: ignore[assignment]
    signals: Dict[str, Signal] = analysis["signals"]      # type: ignore[assignment]
    written: List[Path] = []
    tasks = [
        ("overview", lambda p: overview_figure(shot, result, signals, path=p)),
        ("events", lambda p: event_zoom_figure(result, signals, path=p)),
        ("conditional", lambda p: conditional_figure(analysis["conditional"], result.shot_no, path=p)),
        ("statistics", lambda p: statistics_figure(result, signals.get("Ip"), path=p)),
        ("spectrogram", lambda p: spectrogram_figure(result, signals, path=p)),
    ]
    for name, make in tasks:
        path = outdir / f"shot{result.shot_no}_{name}.png"
        try:
            make(path)
            written.append(path)
        except Exception as exc:
            logger.warning("figure %s failed: %s", name, exc)
    return written


def onset_criterion_figure(conditions: pd.DataFrame, path: Optional[Path] = None) -> Figure:
    """Which quantity is constant when the instability starts?

    One row per candidate trigger; each dot is one shot, plotted as the value
    divided by that criterion's median, so all criteria share one axis and the
    tightest cluster is the answer. A trigger that is a genuine threshold sits
    on a narrow spike while the others spread out.
    """
    apply_style()
    columns = [c for c in ("Ip_frac_of_max", "t_onset_ms", "flux_at_onset_mWb",
                           "Ip_at_onset_kA", "q_a_at_onset", "ne_at_onset")
               if c in conditions]
    labels = {"Ip_frac_of_max": r"$I_p/I_{p,max}$", "t_onset_ms": "onset time",
              "flux_at_onset_mWb": "delivered flux", "Ip_at_onset_kA": r"$I_p$",
              "q_a_at_onset": r"$q_a$", "ne_at_onset": r"$\bar{n}_e$"}
    fig, ax = plt.subplots(figsize=(9, 0.75 * len(columns) + 1.6))
    for row, column in enumerate(columns):
        values = pd.to_numeric(conditions[column], errors="coerce").dropna()
        if values.empty:
            continue
        median = float(values.median())
        if not median:
            continue
        scatter = (values / median).to_numpy()
        jitter = np.linspace(-0.16, 0.16, scatter.size)
        ax.scatter(scatter, np.full_like(scatter, row) + jitter,
                   s=26, color=SERIES[row % len(SERIES)], alpha=0.85, lw=0)
        iqr = float(values.quantile(0.75) - values.quantile(0.25)) / abs(median)
        ax.text(2.05, row, f"IQR/median = {iqr * 100:.1f} %", va="center",
                fontsize=9, color=INK_SECONDARY)
    ax.axvline(1.0, color=INK_MUTED, lw=1.0, ls="--")
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels([labels.get(c, c) for c in columns])
    ax.set_xlim(0, 2.6)
    ax.set_xlabel("value at onset, divided by its median across shots")
    ax.set_title("What is constant when the instability starts? "
                 f"({len(conditions)} candidate shots)")
    ax.invert_yaxis()
    fig.tight_layout()
    return _save(fig, path)
