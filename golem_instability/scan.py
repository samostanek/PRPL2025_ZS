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
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .detect import DetectionParams, detect_in_shot, result_to_row
from .fetch import GolemFetcher
from .shotdata import Shot, ShotNotFound

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
