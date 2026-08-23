"""Synthetic GOLEM shots written in the server's own directory layout.

Used for two things:

1. Validating the whole pipeline (loader -> detector -> analysis -> plots)
   without network access to ``golem.fjfi.cvut.cz``.
2. Sanity-checking that the hypothesis tests actually discriminate: the same
   loop-voltage staircase is generated once from a *circuit* cause (the drive
   is chopped, plasma untouched) and once from a *plasma* cause (repetitive
   MHD-driven minor disruptions), and the tests must tell them apart.

The generated tree is byte-compatible with :class:`~golem_instability.fetch.GolemFetcher`
in ``offline=True`` mode, so no code path is special-cased for test data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# Rough GOLEM-scale circuit constants used by the toy model.
L_PLASMA_H = 1.2e-6    # plasma self-inductance [H]
R_PLASMA_OHM = 3.5e-3  # ohmic plasma resistance [Ohm]
L_CHAMBER_H = 1.2e-6   # inductance of the chamber current path [H]
R_CHAMBER_OHM = 9.7e-3  # resistance of the GOLEM vacuum vessel [Ohm]


@dataclass
class SyntheticConfig:
    """Parameters of one fabricated discharge.

    Attributes:
        kind: ``"clean"``, ``"circuit"``, ``"plasma"``, ``"radiative"`` or
            ``"artefact"``. The last one reproduces the *processing* artefact:
            the loop-voltage dropouts are real but the plasma current is not,
            and the staircase seen in the published ``Ip`` is written entirely
            by the resistive chamber-current correction (see
            :func:`golem_instability.analyze.chamber_correction_test`).
        dt_ms: Sample spacing (0.002 ms = 500 kHz, close to the real DAS).
        t_start_ms / t_end_ms: Extent of the record.
        u_peak: Peak loop voltage [V].
        u_decay_ms: Decay time of the loop-voltage envelope [ms].
        t_breakdown: Breakdown time [ms].
        t_onset: First collapse [ms].
        period_ms: Mean repetition period of the collapses [ms].
        n_events: Number of collapses.
        collapse_ms: Duration of one collapse [ms].
        jitter_ms: Random jitter added to the period [ms].
        seed: RNG seed.
    """

    kind: str = "circuit"
    dt_ms: float = 0.002
    t_start_ms: float = -2.0
    t_end_ms: float = 26.0
    u_peak: float = 18.0
    u_decay_ms: float = 14.0
    t_breakdown: float = 1.2
    t_onset: float = 20.0
    period_ms: float = 0.9
    n_events: int = 6
    collapse_ms: float = 0.10
    jitter_ms: float = 0.03
    seed: int = 7


def _event_times(cfg: SyntheticConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.kind == "clean" or cfg.n_events <= 0:
        return np.array([])
    base = cfg.t_onset + np.arange(cfg.n_events) * cfg.period_ms
    return base + rng.normal(0.0, cfg.jitter_ms, size=base.size)


def _drive_envelope(t: np.ndarray, cfg: SyntheticConfig) -> np.ndarray:
    """Loop voltage the transformer would deliver without any collapses."""
    env = cfg.u_peak * np.exp(-(t - cfg.t_breakdown) / cfg.u_decay_ms)
    env[t < cfg.t_breakdown] = 0.0
    ramp = np.clip((t - cfg.t_breakdown) / 0.15, 0.0, 1.0)
    return env * ramp


def _gate(t: np.ndarray, t_on: float, width: float, edge: float = 0.01) -> np.ndarray:
    """Smoothed 0/1 gate that is 1 inside ``[t_on, t_on+width]``."""
    rise = 0.5 * (1 + np.tanh((t - t_on) / edge))
    fall = 0.5 * (1 - np.tanh((t - (t_on + width)) / edge))
    return rise * fall


def _integrate_ip(t: np.ndarray, u: np.ndarray, r_plasma: np.ndarray) -> np.ndarray:
    """Integrate ``L dIp/dt = U - R Ip`` (forward Euler, ms/kA units)."""
    dt_s = np.diff(t, prepend=t[0]) * 1e-3
    ip = np.zeros_like(t)
    current = 0.0
    for i in range(1, t.size):
        current += dt_s[i] * (u[i] - r_plasma[i] * current) / L_PLASMA_H
        current = max(current, 0.0)
        ip[i] = current
    return ip / 1e3  # A -> kA


def generate(cfg: Optional[SyntheticConfig] = None) -> Dict[str, np.ndarray]:
    """Build all traces of one synthetic discharge.

    Returns:
        Dict of arrays; ``t`` plus one entry per registry signal key.
    """
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    t = np.arange(cfg.t_start_ms, cfg.t_end_ms, cfg.dt_ms)
    events = _event_times(cfg, rng)

    env = _drive_envelope(t, cfg)
    collapse_gate = np.zeros_like(t)
    for t_on in events:
        collapse_gate = np.maximum(collapse_gate, _gate(t, t_on, cfg.collapse_ms))

    resistance = np.full_like(t, R_PLASMA_OHM)
    theta = 2 * np.pi * np.arange(16) / 16.0   # poloidal angles of the Mirnov ring
    mhd_common = rng.normal(0.0, 0.004, size=t.size)
    mode_amp = np.zeros_like(t)                # amplitude of the rotating mode
    mode_phase = 2 * np.pi * 12.0 * t          # 12 kHz rotation
    m_number = 2
    displacement = np.zeros_like(t)            # column shift proxy (m=1 component)
    h_alpha = np.full_like(t, 0.05)
    impurity = np.full_like(t, 0.05)
    density = np.zeros_like(t)
    scint = 0.05 + np.abs(rng.normal(0.0, 0.005, size=t.size))

    if cfg.kind == "plasma":
        # Repetitive minor disruption: a tearing mode grows for a few hundred
        # microseconds, crashes, the plasma cools (R jumps), recycling flashes
        # on Halpha and particles are expelled.
        for t_on in events:
            growth = np.exp(-(t_on - t) / 0.25) * ((t <= t_on) & (t > t_on - 1.2))
            mode_amp += 0.20 * growth
            crash = _gate(t, t_on, cfg.collapse_ms, edge=0.02)
            resistance += 2.5 * R_PLASMA_OHM * crash
            after = np.clip(t - t_on, 0, None)
            h_alpha += 1.2 * np.exp(-after / 0.20) * (t >= t_on)
            impurity += 0.5 * np.exp(-after / 0.30) * (t >= t_on)
            density -= 0.15 * (t >= t_on) * np.exp(-after / 3.0)
            scint += 1.0 * np.exp(-after / 0.05) * (t >= t_on)
    elif cfg.kind == "radiative":
        # Impurity radiation grows first, cools the edge, the collapse follows:
        # emission LEADS the loop-voltage drop instead of following it.
        for t_on in events:
            before = np.clip(t_on - t, 0, None)
            rising = (t <= t_on) & (t > t_on - 1.0)
            impurity += 1.5 * np.exp(-before / 0.35) * rising
            h_alpha += 0.4 * np.exp(-before / 0.35) * rising
            resistance += 2.0 * R_PLASMA_OHM * _gate(t, t_on - 0.05, cfg.collapse_ms + 0.1, edge=0.05)
            density -= 0.05 * (t >= t_on) * np.exp(-np.clip(t - t_on, 0, None) / 3.0)
    elif cfg.kind in ("circuit", "artefact"):
        # The transformer drive is interrupted; the plasma is a passive victim,
        # so no precursor, no recycling flash, resistance unchanged. The only
        # magnetic signature is common-mode pickup from the switching itself.
        for t_on in events:
            mhd_common += 0.03 * _gate(t, t_on, 0.01, edge=0.004)

    mhd_ring = np.vstack([
        mode_amp * np.sin(mode_phase + m_number * ang)
        + displacement * np.cos(ang)
        + mhd_common
        + rng.normal(0, 0.003, t.size)
        for ang in theta
    ])

    if cfg.kind in ("plasma", "radiative"):
        # A resistive plasma event does not zero the transformer EMF as hard,
        # and the drive spikes back up as the current tries to decay.
        u_loop = env * (1.0 - 0.85 * collapse_gate)
        u_loop += 0.4 * env * np.concatenate(([0.0], np.diff(collapse_gate) < 0))
    else:
        u_loop = env * (1.0 - 0.99 * collapse_gate)
    u_loop += rng.normal(0.0, 0.08, size=t.size)

    ip_true = _integrate_ip(t, np.clip(u_loop, 0, None), resistance)
    # Chamber current: driven by the same loop voltage but with its own L/R,
    # so it cannot follow a fast U_loop transient resistively.
    i_chamber = np.zeros_like(t)
    dt_s = np.diff(t, prepend=t[0]) * 1e-3
    current = 0.0
    for i in range(1, t.size):
        current += dt_s[i] * (u_loop[i] - R_CHAMBER_OHM * current) / L_CHAMBER_H
        i_chamber[i] = current
    i_chamber /= 1e3                              # A -> kA
    i_chamber_resistive = np.clip(u_loop, 0, None) / R_CHAMBER_OHM / 1e3
    if cfg.kind == "artefact":
        # The plasma current itself is untouched by the dropouts; what the
        # pipeline publishes is Rogowski minus the *resistive* estimate.
        ip_true = _integrate_ip(t, np.clip(env, 0, None), resistance)
        ip = ip_true + i_chamber - i_chamber_resistive
    else:
        ip = ip_true
    bt = 0.38 * np.exp(-t / 60.0) * (t > -0.2)
    ne = np.clip(4.0 + density * 4.0, 0, None) * 1e18 * (ip > 0.05)
    h_alpha = h_alpha + 0.6 * np.clip(ip, 0, None) / max(ip.max(), 1e-9)

    data: Dict[str, np.ndarray] = {
        "t": t,
        "U_loop": u_loop,
        "Ip": ip,
        "Bt": bt,
        "Ich": i_chamber_resistive + rng.normal(0, 2e-4, t.size),  # published in kA
        "ne_lav": ne,
        "Halpha": h_alpha + rng.normal(0, 0.01, t.size),
        "CII_514": impurity + rng.normal(0, 0.01, t.size),
        "OI_777": 0.6 * impurity + rng.normal(0, 0.01, t.size),
        "CeBr": scint,
        "dIp_dt": np.gradient(ip, t),
    }
    for i in range(1, 17):
        data[f"mhd_{i:02d}"] = mhd_ring[i - 1]
    data["_ip_true"] = ip_true
    data["_i_chamber"] = i_chamber
    data["_events"] = events
    return data


def _write_csv(path: Path, t: np.ndarray, y: np.ndarray, decimate: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stack = np.column_stack([t[::decimate], y[::decimate]])
    np.savetxt(path, stack, delimiter=",", fmt="%.8g")


def write_shot(cache_dir: Path, shot_no: int, cfg: Optional[SyntheticConfig] = None,
               diagnostics: Sequence[str] = ("Interferometry", "FastSpectrometry",
                                             "MHDring-TM", "ScintillationProbes")) -> Path:
    """Write one synthetic shot into a fetcher cache tree.

    Args:
        cache_dir: Cache root (what ``FetcherConfig.cache_dir`` points at).
        shot_no: Shot number to fabricate.
        cfg: Discharge configuration.
        diagnostics: Off-stage diagnostics to advertise and populate.

    Returns:
        Path of the shot directory that was written.
    """
    cfg = cfg or SyntheticConfig()
    data = generate(cfg)
    root = Path(cache_dir) / str(shot_no)
    root.mkdir(parents=True, exist_ok=True)

    (root / "shot_no").write_text(f"{shot_no}\n")
    (root / "shot_date").write_text("26-08-20\n")
    (root / "shot_time").write_text("12:00:00\n")
    (root / "session_id").write_text("synthetic\n")
    (root / "comment").write_text(f"synthetic {cfg.kind} shot\n")

    prod = root / "Production"
    prod.mkdir(exist_ok=True)
    (prod / "Diagnostics_Basic").write_text("PlasmaDetection/Basic BasicDiagnostics/Basic\n")
    (prod / "Diagnostics_OnStage").write_text("\n")
    (prod / "Diagnostics_OffStage").write_text(
        " ".join(f"{name}/Default" for name in diagnostics) + "\n")

    from . import config as cfg_mod

    t = data["t"]
    for key, spec in cfg_mod.SIGNALS.items():
        if key not in data:
            continue
        if spec.diagnostic not in ("BasicDiagnostics", "PlasmaDetection") \
                and spec.diagnostic not in diagnostics:
            continue
        _write_csv(root / spec.path, t, data[key])

    det = root / "Diagnostics" / "PlasmaDetection" / "Results"
    det.mkdir(parents=True, exist_ok=True)
    ip = data["Ip"]
    alive = t[ip > 0.05 * max(ip.max(), 1e-9)]
    t0, t1 = (float(alive[0]), float(alive[-1])) if alive.size else (0.0, 0.0)
    (det / "b_plasma").write_text("1.000\n" if alive.size else "0.000\n")
    (det / "t_plasma_start").write_text(f"{t0:.4f}\n")
    (det / "t_plasma_end").write_text(f"{t1:.4f}\n")
    (det / "t_plasma_duration").write_text(f"{t1 - t0:.4f}\n")

    basic = root / "Diagnostics" / "BasicDiagnostics" / "Results"
    basic.mkdir(parents=True, exist_ok=True)
    (basic / "Ip_max").write_text(f"{ip.max():.4f}\n")
    (basic / "Ip_mean").write_text(f"{ip[ip > 0].mean() if (ip > 0).any() else 0:.4f}\n")
    (basic / "U_loop_max").write_text(f"{data['U_loop'].max():.4f}\n")
    (basic / "Bt_max").write_text(f"{data['Bt'].max():.4f}\n")
    return root


def write_test_set(cache_dir: Path, shots: Optional[Dict[int, SyntheticConfig]] = None) -> List[int]:
    """Write a small mixed population: circuit-like, plasma-like and clean shots."""
    shots = shots or {
        99001: SyntheticConfig(kind="circuit", seed=1),
        99002: SyntheticConfig(kind="plasma", seed=2, period_ms=1.3, n_events=5),
        99003: SyntheticConfig(kind="clean", seed=3, n_events=0),
        99004: SyntheticConfig(kind="circuit", seed=4, t_onset=18.0, n_events=8, period_ms=0.7),
        99005: SyntheticConfig(kind="radiative", seed=5, period_ms=1.6, n_events=5,
                               collapse_ms=0.25),
    }
    for shot_no, cfg in shots.items():
        write_shot(Path(cache_dir), shot_no, cfg)
    return sorted(shots)
