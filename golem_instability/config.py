"""Static description of the GOLEM shot database: URLs and signal registry.

Everything the rest of the package knows about *where* data lives on
``golem.fjfi.cvut.cz`` and *what* it means physically is collected here, so a
change in the server layout only has to be made in one place.

Layout reference (verified against shots #51117 / #53735)::

    shotdir/<N>/
      shot_no, shot_date, shot_time, session_id, comment   tiny text files
      Production/Diagnostics_Basic|OnStage|OffStage        what actually ran
      Diagnostics/<Name>/Results/*.csv                     processed signals
      Infrastructure/<Name>/                               machine subsystems
      Analysis/PlasmaParameters/                           derived headline numbers

All processed signal CSVs are two columns without a header: ``time_ms,value``
with ``t=0`` at the discharge trigger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

BASE_URL: str = "https://golem.fjfi.cvut.cz"

#: GOLEM geometry: major radius and limiter (minor) radius in metres.
MAJOR_RADIUS_M: float = 0.40
MINOR_RADIUS_M: float = 0.085
SHOTDIR_TEMPLATE: str = BASE_URL + "/shotdir/{shot}/"

#: Files that are cheap enough to fetch for every shot while scanning.
META_FILES: Tuple[str, ...] = (
    "shot_no",
    "shot_date",
    "shot_time",
    "session_id",
    "comment",
)

#: Machine-generated lists of the diagnostics that really ran on a shot.
PRODUCTION_DIAGNOSTIC_FILES: Dict[str, str] = {
    "basic": "Production/Diagnostics_Basic",
    "on_stage": "Production/Diagnostics_OnStage",
    "off_stage": "Production/Diagnostics_OffStage",
}

#: Single value, "1.000" when a plasma was created.
PLASMA_FLAG_PATH: str = "Diagnostics/PlasmaDetection/Results/b_plasma"


@dataclass(frozen=True)
class SignalSpec:
    """One loadable time trace.

    Attributes:
        key: Short name used throughout this package (e.g. ``"U_loop"``).
        diagnostic: Folder under ``Diagnostics/`` that must be active for the
            signal to exist. Empty string for paths outside ``Diagnostics/``.
        path: Path relative to ``shotdir/<N>/``.
        unit: Physical unit of the second CSV column.
        label: Human readable label used in plot legends.
        group: Coarse category, used to lay figures out and to decide which
            signals participate in which hypothesis test.
        time_scale_ms: Multiplier from the file's time column to milliseconds.
            Almost every diagnostic already writes ms, but MHDring-TM writes
            **seconds**, so the convention cannot be assumed.
    """

    key: str
    diagnostic: str
    path: str
    unit: str
    label: str
    group: str
    time_scale_ms: float = 1.0


def _spec(key: str, diagnostic: str, rel: str, unit: str, label: str, group: str,
          time_scale_ms: float = 1.0) -> SignalSpec:
    path = f"Diagnostics/{diagnostic}/{rel}" if diagnostic else rel
    return SignalSpec(key, diagnostic, path, unit, label, group, time_scale_ms)


def _build_registry() -> Dict[str, SignalSpec]:
    specs: List[SignalSpec] = [
        # --- Core machine signals, always acquired ------------------------------
        _spec("U_loop", "BasicDiagnostics", "Results/U_loop.csv", "V", r"$U_{loop}$", "basic"),
        _spec("Ip", "BasicDiagnostics", "Results/Ip.csv", "kA", r"$I_p$", "basic"),
        _spec("Bt", "BasicDiagnostics", "Results/Bt.csv", "T", r"$B_t$", "basic"),
        _spec("Ich", "BasicDiagnostics", "Results/Ich.csv", "kA", r"$I_{ch}$ (chamber)", "basic"),
        # --- Derived plasma-detection traces ------------------------------------
        _spec("dIp_dt", "PlasmaDetection", "dIp_dt.csv", "kA/ms", r"$dI_p/dt$", "derived"),
        _spec("dBt_dt", "PlasmaDetection", "dBt_dt.csv", "T/ms", r"$dB_t/dt$", "derived"),
        _spec("V_loop_pd", "PlasmaDetection", "V_loop.csv", "V", r"$U_{loop}$ (PD)", "derived"),
        _spec("V_null", "PlasmaDetection", "V_null.csv", "V", r"$U_{null}$", "derived"),
        _spec("L_tot", "PlasmaDetection", "L_tot.csv", "H", r"$L_{tot}$", "derived"),
        _spec("q_edge_Bt", "PlasmaDetection", "ql_B_t.csv", "-", r"$q_{edge}$ ($B_t$ term)", "derived"),
        _spec("q_edge_Ip", "PlasmaDetection", "ql_I_p.csv", "-", r"$q_{edge}$ ($I_p$ term)", "derived"),
        # --- Line averaged density ----------------------------------------------
        _spec("ne_lav", "Interferometry", "ne_lav.csv", "m^-3", r"$\bar{n}_e$", "density"),
        _spec("ne_lav_raw", "Interferometry", "ne_lav_unrepaired.csv", "m^-3", r"$\bar{n}_e$ (raw)", "density"),
        # --- Fast line spectroscopy ---------------------------------------------
        _spec("Halpha", "FastSpectrometry", "L_Halpha_656.csv", "a.u.", r"H$\alpha$ 656", "spectro"),
        _spec("Hbeta", "FastSpectrometry", "L_Hbeta_486.csv", "a.u.", r"H$\beta$ 486", "spectro"),
        _spec("Hgamma", "FastSpectrometry", "L_Hgamma_434.csv", "a.u.", r"H$\gamma$ 434", "spectro"),
        _spec("HeI_447", "FastSpectrometry", "L_HeI_447.csv", "a.u.", "He I 447", "spectro"),
        _spec("CII_514", "FastSpectrometry", "L_CII_514.csv", "a.u.", "C II 514", "spectro"),
        _spec("NII_568", "FastSpectrometry", "L_NII_568.csv", "a.u.", "N II 568", "spectro"),
        _spec("OI_777", "FastSpectrometry", "L_OI_777.csv", "a.u.", "O I 777", "spectro"),
        _spec("L_broad", "FastSpectrometry", "L_broad.csv", "a.u.", "broadband", "spectro"),
        # --- Radiation / fast electron proxies ----------------------------------
        _spec("CeBr", "ScintillationProbes", "CeBr-a.csv", "a.u.", "CeBr scint.", "radiation"),
        _spec("YAP", "ScintillationProbes", "YAP-a.csv", "a.u.", "YAP scint.", "radiation"),
        _spec("DDRE_1", "ScintillationProbes", "DDRE-1.csv", "a.u.", "DDRE-1", "radiation"),
        _spec("DDRE_2", "ScintillationProbes", "DDRE-2.csv", "a.u.", "DDRE-2", "radiation"),
        _spec("DDRE_3", "ScintillationProbes", "DDRE-3.csv", "a.u.", "DDRE-3", "radiation"),
        _spec("U_loop_scint", "ScintillationProbes", "U_Loop-b.csv", "V", r"$U_{loop}$ (scint. ref.)", "radiation"),
        # --- Stored energy -------------------------------------------------------
        _spec("U_diam_in", "Diamagnetism", "U_diam-in.csv", "V", "diamag. inner", "energy"),
        _spec("U_diam_out", "Diamagnetism", "U_diam-out.csv", "V", "diamag. outer", "energy"),
        _spec("U_coil_hfs", "Diamagnetism", "U_coil-hfs.csv", "V", "HFS comp. coil", "energy"),
        _spec("U_rog_coil", "Diamagnetism", "U_rog-coil.csv", "V", "Rogowski coil", "energy"),
    ]
    # Mirnov ring: 16 coils, dB/dt at fixed poloidal positions.
    # MHDring-TM (and, on some shots, PlasmaDetection, FastSpectrometry,
    # Diamagnetism and the probes) write their time column in SECONDS rather
    # than the documented milliseconds. That is detected per file at load time
    # rather than hard-coded here, so both conventions load correctly - see
    # `shotdata.parse_signal_csv` and the `time_rescaled` flag it sets.
    specs += [
        _spec(f"mhd_{i:02d}", "MHDring-TM", f"ring_{i}.csv", "V", f"Mirnov {i}", "mhd")
        for i in range(1, 17)
    ]
    specs += [
        _spec(f"mc_{i:02d}", "MHDring-TM", f"U_mc{i}.csv", "V", f"Mirnov coil {i}", "mhd")
        for i in (1, 5, 9, 13)
    ]
    specs += [
        _spec("lim_saddle", "MHDring-TM", "LimSaddleCoil.csv", "V", "limiter saddle coil", "mhd"),
        _spec("U_loop_c", "MHDring-TM", "U_loop_c.csv", "V", r"$U_{loop}$ (MHD ref.)", "mhd"),
    ]
    # Edge probes.
    specs += [
        _spec(f"drp_L{i}", "DoubleRakeProbe", f"L{i}.csv", "a.u.", f"DRP L{i}", "probe")
        for i in range(1, 7)
    ]
    specs += [
        _spec(f"drp_R{i}", "DoubleRakeProbe", f"R{i}.csv", "a.u.", f"DRP R{i}", "probe")
        for i in range(1, 7)
    ]
    specs += [
        _spec(f"rp_{i:02d}", "RakeProbe", f"RP{i:02d}.csv", "a.u.", f"RP{i:02d}", "probe")
        for i in range(1, 11)
    ]
    return {s.key: s for s in specs}


#: Every signal this package knows how to load, keyed by short name.
SIGNALS: Dict[str, SignalSpec] = _build_registry()

#: Signals loaded for the standard multi-diagnostic overview, in plot order.
OVERVIEW_KEYS: Tuple[str, ...] = (
    "U_loop",
    "Ip",
    "Bt",
    "ne_lav",
    "Halpha",
    "CII_514",
    "OI_777",
    "CeBr",
    "mhd_01",
)

#: Pre-computed scalars, handy while scanning many shots.
SCALAR_FILES: Dict[str, str] = {
    "Ip_max": "Diagnostics/BasicDiagnostics/Results/Ip_max",
    "Ip_mean": "Diagnostics/BasicDiagnostics/Results/Ip_mean",
    "Bt_max": "Diagnostics/BasicDiagnostics/Results/Bt_max",
    "Bt_mean": "Diagnostics/BasicDiagnostics/Results/Bt_mean",
    "U_loop_max": "Diagnostics/BasicDiagnostics/Results/U_loop_max",
    "U_loop_mean": "Diagnostics/BasicDiagnostics/Results/U_loop_mean",
    "t_plasma_start": "Diagnostics/PlasmaDetection/Results/t_plasma_start",
    "t_plasma_end": "Diagnostics/PlasmaDetection/Results/t_plasma_end",
    "t_plasma_duration": "Diagnostics/PlasmaDetection/Results/t_plasma_duration",
}


def signal(key: str) -> SignalSpec:
    """Return the :class:`SignalSpec` for ``key``, raising a helpful error."""
    try:
        return SIGNALS[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown signal {key!r}; known keys: {sorted(SIGNALS)}") from exc
