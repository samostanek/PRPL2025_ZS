"""Analysis of the repetitive loop-voltage collapse instability on GOLEM.

Typical use::

    from golem_instability import GolemFetcher, Shot, detect_in_shot, analyse_shot

    fetcher = GolemFetcher()
    shot = Shot(53740, fetcher)
    result = detect_in_shot(shot)
    analysis = analyse_shot(shot, result)

or from the command line::

    python -m golem_instability shot 53740 --outdir out/
"""

from .analyze import HypothesisVerdict, analyse_shot, score_hypotheses, volt_second_test
from .config import SIGNALS, SignalSpec, signal
from .detect import DetectionParams, DetectionResult, detect_collapses, detect_in_shot
from .fetch import FetcherConfig, GolemFetcher
from .scan import candidates, scan_shots, scan_range, summarise_scan
from .shotdata import Shot, Signal, load_shot

__all__ = [
    "GolemFetcher", "FetcherConfig", "Shot", "Signal", "load_shot",
    "SIGNALS", "SignalSpec", "signal",
    "DetectionParams", "DetectionResult", "detect_collapses", "detect_in_shot",
    "analyse_shot", "score_hypotheses", "HypothesisVerdict", "volt_second_test",
    "scan_shots", "scan_range", "candidates", "summarise_scan",
]
__version__ = "0.1.0"
