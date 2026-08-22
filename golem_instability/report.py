"""Markdown reports generated from the analysis output."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .analyze import HypothesisVerdict, volt_second_test
from .detect import DetectionResult
from .shotdata import Shot

#: Metrics printed in the per-shot summary table, with their meaning.
METRIC_GLOSSARY: List[tuple] = [
    ("n_events", "collapses detected", ""),
    ("t_onset", "first collapse", "ms"),
    ("period_mean", "mean repetition period", "ms"),
    ("period_cv", "period coefficient of variation", ""),
    ("duration_mean", "mean collapse duration", "ms"),
    ("depth_mean", "mean depth, 1 = down to zero", ""),
    ("duty_cycle", "fraction of the burst spent collapsed", ""),
    ("d_ip_mean", "current step per collapse", "kA"),
    ("d_ip_sum", "current lost in all collapses", "kA"),
    ("staircase_fraction", "share of the current decay inside collapses", ""),
    ("R_pre_mOhm", "plasma resistance before the collapse", "mOhm"),
    ("R_in_mOhm", "plasma resistance during the collapse", "mOhm"),
    ("R_ratio", "resistance ratio in/before", ""),
    ("tau_LR_est_ms", "L/R decay time implied by the collapse", "ms"),
    ("flux_before_onset", "flux delivered before the first collapse", "mWb"),
    ("flux_deficit_sum", "flux not delivered during the collapses", "mWb"),
    ("precursor_ratio", "Mirnov activity growth before the collapse", ""),
    ("m_est", "poloidal mode number of the precursor", ""),
    ("f_mode_khz", "precursor frequency", "kHz"),
    ("mirnov_common_mode_ratio", "in-phase (m=0) share of the Mirnov jump", ""),
    ("ring_asymmetry_growth", "growth of the m=1 displacement component", ""),
    ("Halpha_z", "H-alpha excursion", "sigma"),
    ("Halpha_onset_lag_ms", "H-alpha first rise vs. collapse (negative = leads)", "ms"),
    ("CII_514_z", "C II excursion", "sigma"),
    ("CII_514_onset_lag_ms", "C II first rise vs. collapse (negative = leads)", "ms"),
    ("CeBr_z", "scintillator excursion", "sigma"),
    ("ne_step_rel", "density step per collapse", ""),
    ("ne_mean", "mean line-averaged density", "m^-3"),
    ("rail_fraction", "in-collapse samples on a flat floor", ""),
    ("chamber_swing_kA", "size of the chamber-current correction per collapse", "kA"),
    ("dip_over_chamber_swing", "current step divided by that correction swing", ""),
    ("tau_chamber_ms", "L/R of the chamber current path", "ms"),
    ("tau_chamber_over_duration", "chamber L/R over collapse duration", ""),
    ("d_irog_mean", "step of the reconstructed uncorrected total current", "kA"),
    ("ip_dip", "transient current excursion at the collapse", "kA"),
    ("ip_step", "current step still present before the next collapse", "kA"),
    ("ip_recovery", "recovered fraction of the current excursion", ""),
    ("dip_ratio_irog_ip", "excursion in the uncorrected vs published current", ""),
    ("ip_dyn_dip", "current excursion with an inductive chamber correction", "kA"),
    ("dip_ratio_dyn_pub", "that excursion divided by the published one", ""),
    ("R_chamber_implied_mOhm", "chamber resistance implied by the shot's own Ich", "mOhm"),
    ("instability_score", "detector ranking score", ""),
]


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "-"
    if number != 0 and (abs(number) >= 1e4 or abs(number) < 1e-3):
        return f"{number:.3e}"
    return f"{number:.3f}"


def metrics_table(metrics: Dict[str, float]) -> str:
    """Markdown table of the measured quantities."""
    rows = ["| quantity | value | unit | meaning |", "|---|---:|---|---|"]
    for key, meaning, unit in METRIC_GLOSSARY:
        if key not in metrics:
            continue
        rows.append(f"| `{key}` | {_fmt(metrics[key])} | {unit or '-'} | {meaning} |")
    return "\n".join(rows)


def verdict_table(verdicts: Sequence[HypothesisVerdict]) -> str:
    """Markdown table ranking the hypotheses."""
    rows = ["| # | mechanism | score | verdict |", "|---|---|---:|---|"]
    for v in verdicts:
        score = "-" if not np.isfinite(v.score) else f"{v.score:+.2f}"
        rows.append(f"| {v.key} | {v.name} | {score} | **{v.verdict}** |")
    return "\n".join(rows)


def verdict_details(verdicts: Sequence[HypothesisVerdict]) -> str:
    """The evidence behind each verdict, hypothesis by hypothesis."""
    blocks: List[str] = []
    for v in verdicts:
        score = "-" if not np.isfinite(v.score) else f"{v.score:+.2f}"
        block = [f"#### {v.key}. {v.name}  ({v.verdict}, score {score})", ""]
        block += [f"- {line}" for line in v.evidence] or ["- no deciding metric available"]
        if v.missing:
            block.append(f"- *not measurable on this shot:* {', '.join(v.missing)}")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def event_table(result: DetectionResult, max_rows: int = 25) -> str:
    """Markdown table of the individual collapses."""
    cols = ["t_on", "duration_ms", "period_ms", "u_env", "u_min", "depth",
            "ip_pre", "d_ip", "tau_LR_ms"]
    events = result.events[[c for c in cols if c in result.events]].head(max_rows)
    if events.empty:
        return "_no collapses detected_"
    return events.to_markdown(index=False, floatfmt=".3f")


def shot_report(shot: Optional[Shot], analysis: Dict[str, object],
                figures: Optional[Sequence[Path]] = None,
                outdir: Optional[Path] = None,
                inventory: Optional[pd.DataFrame] = None) -> str:
    """Full Markdown report for a single shot."""
    result: DetectionResult = analysis["result"]      # type: ignore[assignment]
    metrics: Dict[str, float] = analysis["metrics"]   # type: ignore[assignment]
    verdicts: List[HypothesisVerdict] = analysis["verdicts"]  # type: ignore[assignment]
    meta = {}
    diagnostics = ""
    if shot is not None:
        try:
            meta = shot.meta
            diagnostics = ", ".join(shot.active_diagnostics)
        except Exception:
            pass

    lines = [
        f"# GOLEM #{result.shot_no} - repetitive loop-voltage collapse analysis",
        "",
        f"- date/time: {meta.get('shot_date', '?')} {meta.get('shot_time', '')}",
        f"- session: {meta.get('session_id', '?')}",
        f"- comment: {meta.get('comment', '')}",
        f"- diagnostics active: {diagnostics or 'unknown'}",
        f"- loop-voltage polarity applied: {result.polarity:+.0f}",
        "",
        "## 1. What was detected",
        "",
        f"{result.n_events} loop-voltage collapses, starting at "
        f"{_fmt(metrics.get('t_onset'))} ms, repeating every "
        f"{_fmt(metrics.get('period_mean'))} ms "
        f"(CV {_fmt(metrics.get('period_cv'))}), each lasting "
        f"{_fmt(metrics.get('duration_mean'))} ms and taking "
        f"{_fmt(metrics.get('d_ip_mean'))} kA off the plasma current.",
        "",
        "## 2. Diagnostics available on this shot",
        "",
        (inventory.to_markdown(index=False, floatfmt=".4g")
         if inventory is not None and not inventory.empty
         else "_signal inventory not requested_"),
        "",
        "## 3. Measured quantities",
        "",
        metrics_table(metrics),
        "",
        "## 4. Individual events",
        "",
        event_table(result),
        "",
        "## 5. Hypothesis ranking",
        "",
        verdict_table(verdicts),
        "",
        "### Evidence",
        "",
        verdict_details(verdicts),
    ]
    if figures:
        lines += ["", "## 6. Figures", ""]
        for path in figures:
            rel = Path(path).name if outdir else str(path)
            lines.append(f"![{rel}]({rel})")
    return "\n".join(lines) + "\n"


def scan_report(scan: pd.DataFrame, min_score: float = 0.5) -> str:
    """Markdown report for a cross-shot scan, including the volt-second test."""
    from .scan import candidates  # imported here to avoid a cycle at import time

    cand = candidates(scan, min_score=min_score)
    flux = volt_second_test(cand if not cand.empty else scan)
    cols = [c for c in ("shot_no", "shot_date", "comment", "n_events", "t_onset",
                        "period_mean", "period_cv", "depth_mean", "d_ip_mean",
                        "flux_before_onset", "instability_score") if c in scan]
    lines = [
        "# Cross-shot scan for repetitive loop-voltage collapses",
        "",
        f"- shots examined: {len(scan)}",
        f"- shots with at least one collapse: "
        f"{int((scan.get('n_events', pd.Series(dtype=float)).fillna(0) > 0).sum())}",
        f"- candidates (score >= {min_score}): {len(cand)}",
        "",
        "## Candidate shots",
        "",
        cand[cols].to_markdown(index=False, floatfmt=".3f") if not cand.empty
        else "_none found_",
        "",
        "## Volt-second test of the core-saturation hypothesis (H5)",
        "",
        f"- coefficient of variation of the flux delivered before onset: {_fmt(flux['flux_cv'])}",
        f"- coefficient of variation of the onset time: {_fmt(flux['onset_cv'])}",
        f"- clustering ratio (onset CV / flux CV): {_fmt(flux['flux_clustering_ratio'])} "
        f"over {int(flux['n'])} shots",
        "",
        "A ratio well above 1 means the onset is set by the flux the transformer has "
        "delivered rather than by the clock, which is what an iron core running into "
        "saturation would do. A ratio near or below 1 argues against H5.",
    ]
    return "\n".join(lines) + "\n"


def write_report(text: str, path: Path) -> Path:
    """Write a report to disk, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
