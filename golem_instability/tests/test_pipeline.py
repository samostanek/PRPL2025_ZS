"""Offline validation of the whole pipeline against synthetic shots.

The synthetic shots have a *known* cause, so these tests check not only that
the code runs but that the physics discriminators point at the right
mechanism and that the derived resistance recovers the value that was put in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from golem_instability.analyze import analyse_shot, volt_second_test
from golem_instability.detect import DetectionParams, detect_in_shot
from golem_instability.fetch import FetcherConfig, GolemFetcher
from golem_instability.scan import candidates, scan_shots
from golem_instability.shotdata import Shot, parse_signal_csv
from golem_instability.synthetic import (R_PLASMA_OHM, SyntheticConfig, write_shot,
                                         write_test_set)


@pytest.fixture(scope="module")
def fetcher(tmp_path_factory) -> GolemFetcher:
    cache = tmp_path_factory.mktemp("cache")
    write_test_set(cache)
    return GolemFetcher(FetcherConfig(cache_dir=cache, offline=True))


def test_parse_signal_csv_two_columns():
    raw = b"-1.0,0.0\n0.0,1.5\n1.0,2.5\n"
    sig = parse_signal_csv(raw, key="test")
    assert len(sig) == 3
    assert sig.t[0] == -1.0
    assert sig.y[-1] == 2.5
    assert sig.dt == pytest.approx(1.0)


def test_parse_signal_csv_skips_junk_rows():
    raw = b"time,value\n0.0,1.0\nbroken row\n1.0,2.0\n"
    sig = parse_signal_csv(raw, key="test")
    assert len(sig) == 2


def test_shot_metadata_and_diagnostics(fetcher):
    shot = Shot(99001, fetcher)
    assert shot.exists()
    assert shot.has_plasma()
    assert "BasicDiagnostics" in shot.active_diagnostics
    assert "MHDring-TM" in shot.active_diagnostics
    assert shot.plasma_window() is not None


def test_missing_diagnostic_is_not_an_error(fetcher, tmp_path):
    """A diagnostic that did not run must load as None, not raise."""
    write_shot(tmp_path, 98001, SyntheticConfig(kind="circuit"), diagnostics=())
    local = GolemFetcher(FetcherConfig(cache_dir=tmp_path, offline=True))
    shot = Shot(98001, local)
    assert shot.signal("U_loop") is not None
    assert shot.signal("mhd_01") is None
    assert shot.signal("ne_lav") is None


def test_detector_finds_the_planted_events(fetcher):
    result = detect_in_shot(Shot(99001, fetcher))
    assert result is not None
    assert result.n_events == 6
    assert result.summary["depth_mean"] > 0.9
    assert result.summary["period_mean"] == pytest.approx(0.9, abs=0.1)
    assert result.summary["d_ip_mean"] < 0        # the current steps down
    assert result.is_candidate


def test_detector_is_quiet_on_a_clean_shot(fetcher):
    result = detect_in_shot(Shot(99003, fetcher))
    assert result is not None
    assert result.n_events == 0
    assert not result.is_candidate


def test_detected_times_match_the_planted_ones(fetcher, tmp_path):
    cfg = SyntheticConfig(kind="circuit", seed=11, jitter_ms=0.0)
    write_shot(tmp_path, 98002, cfg)
    local = GolemFetcher(FetcherConfig(cache_dir=tmp_path, offline=True))
    result = detect_in_shot(Shot(98002, local))
    planted = cfg.t_onset + np.arange(cfg.n_events) * cfg.period_ms
    detected = result.events["t_on"].to_numpy()
    assert len(detected) == len(planted)
    assert np.allclose(detected, planted, atol=0.02)


def test_resistance_test_recovers_the_input_resistance(fetcher):
    """With the drive switched off the freewheeling current must give back R."""
    shot = Shot(99001, fetcher)
    result = detect_in_shot(shot)
    metrics = analyse_shot(shot, result)["metrics"]
    assert metrics["R_in_mOhm"] == pytest.approx(R_PLASMA_OHM * 1e3, rel=0.25)
    assert metrics["R_ratio"] == pytest.approx(1.0, abs=0.35)


@pytest.mark.parametrize("shot_no,expected", [(99001, "H1"), (99002, "H2"), (99005, "H4")])
def test_hypotheses_discriminate_known_causes(fetcher, shot_no, expected):
    shot = Shot(shot_no, fetcher)
    result = detect_in_shot(shot)
    verdicts = analyse_shot(shot, result)["verdicts"]
    assert verdicts[0].key == expected
    assert verdicts[0].verdict == "supported"


def test_mhd_metrics_separate_plasma_from_circuit(fetcher):
    circuit = analyse_shot(Shot(99001, fetcher), detect_in_shot(Shot(99001, fetcher)))["metrics"]
    plasma = analyse_shot(Shot(99002, fetcher), detect_in_shot(Shot(99002, fetcher)))["metrics"]
    assert circuit["precursor_ratio"] < 1.5 < plasma["precursor_ratio"]
    assert plasma["m_est"] == pytest.approx(2.0, abs=0.4)
    assert circuit["R_ratio"] < plasma["R_ratio"]


def test_conditional_average_shape(fetcher):
    shot = Shot(99002, fetcher)
    analysis = analyse_shot(shot, detect_in_shot(shot))
    conditional = analysis["conditional"]
    assert "U_loop" in conditional
    avg = conditional["U_loop"]
    # the last event sits too close to the end of the record for a full
    # post-window, so it is dropped rather than averaged with NaNs
    assert 4 <= avg.n <= 5
    # the loop voltage must be below its pre-event level right after the onset
    assert avg.mean[np.argmin(np.abs(avg.lag_ms - 0.05))] < 0


def test_scan_ranks_candidates(fetcher):
    scan = scan_shots(fetcher, [99001, 99002, 99003, 99004, 99005, 90000], workers=2)
    assert len(scan) == 5                      # the non-existent shot is dropped
    assert scan["instability_score"].is_monotonic_decreasing
    assert 99003 not in candidates(scan)["shot_no"].to_numpy()


def test_volt_second_test_reports_clustering():
    scan = pd.DataFrame({"flux_before_onset": [180.0, 181.0, 179.5, 180.5],
                         "t_onset": [12.0, 20.0, 16.0, 25.0]})
    out = volt_second_test(scan)
    assert out["flux_cv"] < out["onset_cv"]
    assert out["flux_clustering_ratio"] > 5


def test_detection_params_are_honoured(fetcher):
    strict = DetectionParams(enter_ratio=0.02, min_depth=0.99)
    result = detect_in_shot(Shot(99002, fetcher), strict)
    assert result.n_events <= 5


def test_das_dump_parser_reads_channel_labels():
    raw = (b"Model,MSO56\n"
           b"Channel,CH1,,Channel,CH2,,Channel,MATH1\n"
           b"Label,U_loop,coil,,Label,U_Rog,coil,,Label,I_p\n"
           b"Horizontal Units,s,,Horizontal Units,s,,Horizontal Units,s\n"
           b"Sample Interval,1e-06,,Sample Interval,1e-06,,Sample Interval,1e-06\n"
           b"0.000,1.0,,0.000,10.0,,0.000,100.0\n"
           b"0.001,2.0,,0.001,20.0,,0.001,200.0\n")
    from golem_instability.shotdata import parse_das_csv
    channels = parse_das_csv(raw)
    assert set(channels) == {"U_loop", "U_Rog", "I_p"}
    assert channels["U_loop"].t.tolist() == [0.0, 1.0]     # s -> ms
    assert channels["I_p"].y.tolist() == [100.0, 200.0]


def test_chamber_current_model_relaxes_to_ohms_law():
    from golem_instability.analyze import (L_CHAMBER_H, R_CHAMBER_OHM,
                                           chamber_current_model)
    t = np.linspace(0.0, 5.0, 5001)          # ms
    u = np.full_like(t, 10.0)
    current = chamber_current_model(t, u)
    tau_ms = L_CHAMBER_H / R_CHAMBER_OHM * 1e3
    assert current[-1] == pytest.approx(10.0 / R_CHAMBER_OHM / 1e3, rel=1e-3)
    # one time constant in, the step response must be within a percent of 1-1/e
    idx = int(np.argmin(np.abs(t - tau_ms)))
    assert current[idx] / current[-1] == pytest.approx(1 - np.exp(-1), abs=0.01)


def test_chamber_correction_metrics_flag_a_processing_artefact(tmp_path):
    """A staircase written purely by the resistive chamber correction.

    The "artefact" synthetic has real loop-voltage dropouts but an untouched
    plasma current; redoing the correction with the chamber's own L/R must
    remove most of the apparent current excursion, while for the "circuit"
    shot - where the plasma really does lose current - it must not.
    """
    write_shot(tmp_path, 97001, SyntheticConfig(kind="artefact", seed=21))
    write_shot(tmp_path, 97002, SyntheticConfig(kind="circuit", seed=21))
    local = GolemFetcher(FetcherConfig(cache_dir=tmp_path, offline=True))

    def metrics_and_scores(shot_no):
        shot = Shot(shot_no, local)
        analysis = analyse_shot(shot, detect_in_shot(shot))
        scores = {v.key: v.score for v in analysis["verdicts"]}
        return analysis["metrics"], scores

    artefact, artefact_scores = metrics_and_scores(97001)
    circuit, circuit_scores = metrics_and_scores(97002)

    assert artefact["R_chamber_implied_mOhm"] == pytest.approx(9.7, rel=0.05)
    assert artefact["tau_chamber_over_duration"] > 1.0
    assert artefact["dip_ratio_dyn_pub"] < 0.4
    assert circuit["dip_ratio_dyn_pub"] > 0.8
    assert artefact_scores["H7"] > circuit_scores["H7"]


def test_seconds_time_axis_is_detected_and_rescaled(fetcher):
    """MHDring-TM writes seconds, not the documented milliseconds."""
    shot = Shot(99002, fetcher)
    ring = shot.signal("mhd_01")
    basic = shot.signal("U_loop")
    assert ring is not None and basic is not None
    assert ring.time_rescaled and not basic.time_rescaled
    # after rescaling both diagnostics must share the same time base
    assert ring.t[0] == pytest.approx(basic.t[0], abs=0.05)
    assert ring.t[-1] == pytest.approx(basic.t[-1], abs=0.05)
    assert shot.inventory().set_index("key").loc["mhd_01", "time_rescaled"]
