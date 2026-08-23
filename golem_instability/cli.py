"""Command line interface.

Examples::

    # everything about one shot: figures + markdown report
    python -m golem_instability shot 53740 --outdir out/

    # look for the same behaviour in a range of shots
    python -m golem_instability scan --from 53600 --to 53800 --outdir out/

    # the last 200 shots, whatever the latest one happens to be
    python -m golem_instability scan --last 200 --outdir out/

    # validate the whole pipeline offline, without touching the network
    python -m golem_instability selftest
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .analyze import analyse_shot
from .detect import DetectionParams, detect_in_shot
from .fetch import FetcherConfig, GolemFetcher
from .plots import figure_set, onset_criterion_figure, scan_figure
from .report import scan_report, shot_report, write_report
from .scan import candidates, onset_conditions, scan_shots, summarise_scan
from .shotdata import Shot

logger = logging.getLogger("golem_instability")


def build_fetcher(args: argparse.Namespace) -> GolemFetcher:
    return GolemFetcher(FetcherConfig(cache_dir=Path(args.cache), offline=args.offline,
                                      refresh=args.refresh))


def detection_params(args: argparse.Namespace) -> DetectionParams:
    params = DetectionParams()
    for name in ("enter_ratio", "exit_ratio", "min_duration_ms", "env_window_ms"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(params, name, value)
    return params


def cmd_shot(args: argparse.Namespace) -> int:
    fetcher = build_fetcher(args)
    shot = Shot(args.shot, fetcher)
    if not shot.exists():
        logger.error("shot %d not found", args.shot)
        return 1
    print(shot.describe())
    result = detect_in_shot(shot, detection_params(args))
    if result is None:
        logger.error("shot %d has no loop-voltage trace", args.shot)
        return 1
    analysis = analyse_shot(shot, result)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    inventory = shot.inventory()
    inventory.to_csv(outdir / f"shot{args.shot}_signals.csv", index=False)
    print(f"{len(inventory)} signals available across "
          f"{inventory['diagnostic'].nunique() if not inventory.empty else 0} diagnostics")
    figures: List[Path] = []
    if not args.no_figures and result.n_events:
        figures = figure_set(shot, analysis, outdir)
    report = shot_report(shot, analysis, figures, outdir, inventory)
    path = write_report(report, outdir / f"shot{args.shot}_report.md")
    result.events.to_csv(outdir / f"shot{args.shot}_events.csv", index=False)
    print(f"\n{result.n_events} collapses detected; report written to {path}")
    for verdict in analysis["verdicts"][:3]:                    # type: ignore[index]
        print(f"  {verdict.key} {verdict.verdict:<13} {verdict.name}")
    return 0


def _scan_shot_list(args: argparse.Namespace, fetcher: GolemFetcher) -> Sequence[int]:
    if args.shots:
        return [int(s) for s in args.shots]
    if args.last:
        latest = args.to or fetcher.latest_shot()
        return list(range(latest - args.last + 1, latest + 1, args.step))
    if args.since_days:
        import datetime as dt
        from .scan import find_shot_near_date
        latest = args.to or fetcher.latest_shot()
        target = dt.date.today() - dt.timedelta(days=args.since_days)
        first = find_shot_near_date(fetcher, target, latest)
        logger.info("date search: %s -> shot %d", target, first)
        return list(range(first, latest + 1, args.step))
    if args.__dict__["from"] is None or args.to is None:
        raise SystemExit("specify --from/--to, --shots, --last or --since-days")
    return list(range(args.__dict__["from"], args.to + 1, args.step))


def cmd_scan(args: argparse.Namespace) -> int:
    fetcher = build_fetcher(args)
    shots = _scan_shot_list(args, fetcher)
    logger.info("scanning %d shots", len(shots))
    scan = scan_shots(fetcher, shots, detection_params(args),
                      require_plasma=not args.include_no_plasma, workers=args.workers)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scan.to_csv(outdir / "scan.csv", index=False)
    print(summarise_scan(scan))
    conditions = onset_conditions(fetcher, candidates(scan, args.min_score))
    if not conditions.empty:
        conditions.to_csv(outdir / "onset_conditions.csv", index=False)
        onset_criterion_figure(conditions, outdir / "onset_criterion.png")
    write_report(scan_report(scan, args.min_score, conditions),
                 outdir / "scan_report.md")
    if not scan.empty:
        scan_figure(scan, outdir / "scan_overview.png")
    if args.analyse_top:
        top = candidates(scan, args.min_score).head(args.analyse_top)
        for shot_no in top["shot_no"]:
            sub = argparse.Namespace(**vars(args))
            sub.shot = int(shot_no)
            cmd_shot(sub)
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Validate the pipeline offline against synthetic shots of known cause."""
    from .synthetic import write_test_set

    cache = Path(args.cache)
    shots = write_test_set(cache)
    fetcher = GolemFetcher(FetcherConfig(cache_dir=cache, offline=True))
    expected = {99001: "H1", 99002: "H2", 99003: None, 99004: "H1", 99005: "H4"}
    failures = 0
    for shot_no in shots:
        shot = Shot(shot_no, fetcher)
        result = detect_in_shot(shot)
        if result is None:
            print(f"  shot {shot_no}: no loop voltage - FAIL")
            failures += 1
            continue
        want = expected.get(shot_no)
        if want is None:
            ok = result.n_events == 0
            print(f"  shot {shot_no}: {result.n_events} events (expected 0) "
                  f"- {'OK' if ok else 'FAIL'}")
            failures += 0 if ok else 1
            continue
        analysis = analyse_shot(shot, result)
        top = analysis["verdicts"][0]                            # type: ignore[index]
        ok = top.key == want
        print(f"  shot {shot_no}: {result.n_events} events, top hypothesis {top.key} "
              f"(expected {want}) - {'OK' if ok else 'FAIL'}")
        failures += 0 if ok else 1
    print("selftest:", "PASSED" if not failures else f"{failures} FAILURES")
    return 0 if not failures else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="golem_instability",
        description="Find and analyse repetitive loop-voltage collapses on the GOLEM tokamak.")
    parser.add_argument("--cache", default="golem_cache", help="local mirror directory")
    parser.add_argument("--offline", action="store_true",
                        help="use only what is already cached; never open a socket")
    parser.add_argument("--refresh", action="store_true", help="ignore cached copies")
    parser.add_argument("--outdir", default="out", help="where figures and reports go")
    parser.add_argument("-v", "--verbose", action="store_true")
    for name, default in (("enter-ratio", None), ("exit-ratio", None),
                          ("min-duration-ms", None), ("env-window-ms", None)):
        parser.add_argument(f"--{name}", type=float, default=default,
                            dest=name.replace("-", "_"))

    sub = parser.add_subparsers(dest="command", required=True)

    p_shot = sub.add_parser("shot", help="analyse one shot")
    p_shot.add_argument("shot", type=int)
    p_shot.add_argument("--no-figures", action="store_true")
    p_shot.set_defaults(func=cmd_shot)

    p_scan = sub.add_parser("scan", help="search a range of shots")
    p_scan.add_argument("--from", dest="from", type=int, default=None)
    p_scan.add_argument("--to", type=int, default=None)
    p_scan.add_argument("--step", type=int, default=1)
    p_scan.add_argument("--last", type=int, default=None,
                        help="scan the last N shots")
    p_scan.add_argument("--since-days", type=int, default=None,
                        help="scan everything from N days ago until now")
    p_scan.add_argument("--shots", nargs="*", default=None, help="explicit shot numbers")
    p_scan.add_argument("--workers", type=int, default=8)
    p_scan.add_argument("--min-score", type=float, default=0.5)
    p_scan.add_argument("--include-no-plasma", action="store_true")
    p_scan.add_argument("--analyse-top", type=int, default=0,
                        help="also produce a full report for the N best candidates")
    p_scan.add_argument("--no-figures", action="store_true")
    p_scan.set_defaults(func=cmd_scan)

    p_self = sub.add_parser("selftest", help="offline validation on synthetic shots")
    p_self.set_defaults(func=cmd_selftest)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
