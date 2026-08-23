# Repetitive loop-voltage collapse on GOLEM

*Analysis of the instability first noticed on shot #53740, where late in the
discharge the loop voltage repeatedly drops to ~0 and recovers while the plasma
current steps down.*

**Data basis.** Shot #53740 analysed in full, plus a scan of shots
**53500–53761** (262 shots, 56 without plasma): 79 show at least one collapse
and **37 are candidates**. All numbers below are measured. Reproduce with the
commands in §8.

---

## 1. Summary of findings

1. The events are **fast**: 19–171 µs (median 42 µs, with one 0.9 ms
   outlier at the end of the discharge), repeating every
   **≈ 1.0 ms**, starting at 19.7 ms on #53740 and lasting to the end of the
   discharge.
2. They are **not** a loss of drive. At each event the plasma current
   **spikes upwards** by 1.5 % before decaying, the loop voltage undershoots to
   **−2.3 V below its baseline** (i.e. genuinely negative, not a flat zero
   floor: only 0.07 % of in-event samples sit on a rail), and the toroidal
   field shows nothing at all. This is the signature of a **sudden internal
   reconnection at conserved poloidal flux**, implying **Δl_i ≈ −0.07** per
   crash.
3. Each cycle: density accumulates → crash → **13 % of the density is
   expelled**, the current profile flattens, the loop voltage stays ~1 V
   elevated for ~0.5 ms and the current decays by **≈ 0.5 kA** → refuelling and
   reheating → repeat.
4. **The trigger is the end of the current ramp.** Across the 37 candidate
   shots the instability starts when the plasma current reaches
   **96.3 % of that shot's peak (IQR/median = 3.1 %)** — five times tighter
   than any other candidate criterion. It is not set by the clock, the
   delivered flux, the density, the edge safety factor, or the position
   stabilisation.
5. My pre-data prior (circuit-side: thyristor drop-out or core saturation) is
   **wrong** and the measurements say so explicitly — see §6.

## 2. What the two key signals are

**Loop voltage** comes from *a coil around the transformer core*, not a flux
loop through the plasma: it measures the core's dΦ/dt, i.e. the EMF delivered
to everything linked by the core. A sudden *increase* of the secondary
(plasma) current therefore drives a **negative** U_loop spike — which is
exactly what is observed, and is the opposite of what a loss of drive looks
like.

**Plasma current** is the Rogowski signal minus a chamber-current correction
estimated resistively as `I_ch = U_loop / R_chamber`. On #53740 this shot's own
`Ich` trace gives `U_loop/Ich = 9.67 mΩ`, confirming the correction is exactly
resistive with the documented `R_chamber ≈ 9.7 mΩ`. So `Ip` **contains**
`U_loop`, and the chamber correction swings by **0.82 kA** at each event —
larger than the current step being interpreted. This had to be ruled out
before anything else; §6 (H7) shows it survives the check.

| quantity | value | note |
|---|---|---|
| major / minor radius | R₀ = 0.4 m, a = 0.085 m | Mo poloidal limiter |
| #53740 settings | U_CD = 400 V, U_Bt = 800 V, H₂ at p = 13 | from the logged `Dirigent.sh` line |
| #53740 plasma | I_p^max = 7.66 kA, B_t = 0.43 T, 4.2–30.0 ms | |
| chamber resistance | 9.67 mΩ (measured), 9.7 mΩ (documented) | |
| plasma inductance | L_p ≈ 1.2 µH | μ₀R₀(ln(8R₀/a) − 1.75 + lᵢ/2) |
| plasma resistance | 0.79 mΩ before onset, 1.01 mΩ after | +28 % |
| line-averaged density | 3.2·10¹⁸ before onset, 6.5·10¹⁸ after | roughly doubles |

## 3. What one event looks like

Event-triggered average over the 10 collapses of #53740, baseline removed,
lag in ms relative to the onset:

| lag | −0.30 | −0.10 | −0.01 | +0.01 | +0.03 | +0.10 | +0.40 | +0.80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ΔU_loop [V] | +0.21 | −0.53 | −0.79 | **−2.33** | −1.87 | +0.66 | +1.10 | −0.40 |
| ΔI_p [kA] | −0.00 | −0.01 | −0.01 | +0.02 | **+0.048** | −0.02 | −0.33 | **−0.44** |
| Δn̄_e [10¹⁸ m⁻³] | +0.02 | **+0.25** | −0.28 | −0.43 | −0.58 | **−1.02** | −0.43 | +0.05 |
| ΔB_t [T] | 0.000 | 0.001 | 0.001 | 0.001 | 0.002 | 0.002 | 0.004 | 0.006 |

Read as a cycle:

1. **Ramp (−1.0 to −0.05 ms).** The density climbs and the current rises; the
   loop voltage sags slightly as the profile peaks.
2. **Crash (0 to +0.05 ms).** U_loop plunges 2.3 V below baseline for a few
   tens of µs while I_p spikes **up** by 48 A. Conserving `L·I` gives
   `ΔL = −L·ΔI/I ≈ −0.018 µH`, i.e. **Δl_i ≈ −0.07** — a modest, entirely
   plausible flattening of the current profile.
3. **Expulsion (+0.05 to +0.15 ms).** The line-averaged density falls by
   ~1.0·10¹⁸ m⁻³ (13 %).
4. **Recovery (+0.1 to +0.8 ms).** U_loop sits ~1 V *above* its old baseline —
   the plasma is more resistive, having cooled — and I_p decays by 0.44 kA.
   The density refills and the cycle repeats after ~1 ms.

B_t does nothing but drift smoothly through all of it: **no machine circuit is
switching in step with these events.**

## 4. The cross-shot trigger

Every candidate trigger was measured at the onset in all 37 candidate shots.
The real threshold is whichever one stays put while the others scatter
(scatter as interquartile range over median, robust to the few shots whose
onset is caught unusually early or late):

| criterion at onset | n | median | IQR/median |
|---|---:|---:|---:|
| **I_p / I_p,max** | 37 | **0.963** | **3.1 %** |
| onset time | 37 | 17.98 ms | 14.8 % |
| B_t | 37 | 0.368 T | 17.7 % |
| delivered flux ∫U dt | 37 | 78.6 mWb | 18.6 % |
| I_p | 37 | 6.72 kA | 20.2 % |
| q_a | 37 | 4.94 | 40.3 % |
| n̄_e | 27 | 6.0·10¹⁸ m⁻³ | 79.7 % |

The instability starts when the plasma current has essentially finished
rising, in shots spanning two sessions with very different toroidal fields
(B_t ≈ 0.51 T in the July session, ≈ 0.36 T in August) and hence very different
edge safety factors (q_a ≈ 7.5 versus ≈ 4.5). Neither q_a nor the density is
the threshold; the approach to the current flat-top is.

That is the classic condition for the **ohmic current profile to finish
penetrating**. During the ramp the profile is skin-like and the central safety
factor is high; once dI_p/dt → 0 the current diffuses inward resistively, the
profile peaks, and the plasma becomes unstable to internal reconnection. The
relaxation cycle that follows is what clamps the profile.

## 5. Hypotheses and how each was decided

Seven mechanisms were scored against specific measurements (see
`analyze.score_hypotheses`). The ranking on #53740:

| # | mechanism | score | verdict |
|---|---|---:|---|
| H2 | Repetitive minor disruptions / MHD relaxation | **+0.89** | supported |
| H4 | Impurity-radiation / recycling limit cycle | +0.50 | supported (weakly — see below) |
| H5 | Volt-second exhaustion / core saturation | +0.40 | refuted cross-shot |
| H1 | Drive interruption in the primary circuit | −0.39 | weak/mixed |
| H3 | Runaway-electron dominated phases | −0.43 | contradicted |
| H7 | Measurement / processing artefact | −0.66 | contradicted |
| H6 | Position excursion with limiter contact | — | inconclusive |

**H2 — repetitive minor disruptions / internal reconnection. Supported.**
The current spikes up at 90 % of the events (Δl_i ≈ −0.07), and 13 % of the
density is expelled per event. Both are what a crash does and neither is what
a lost drive does. *Caveat:* #53740 ran without the Mirnov ring, so the
tearing-mode precursor and its poloidal mode number could not be measured, and
the band-passed loop voltage shows no growing precursor of its own. This is
the one gap worth closing (§9).

**H1 — drive interruption. Contradicted.** Three independent counts: the
current spikes *upwards*, which a switched-off drive cannot do; the loop
voltage goes genuinely negative rather than sitting on a floor (rail fraction
7·10⁻⁴); and B_t shows no coherent step at the events.

**H5 — core saturation / volt-second exhaustion. Refuted cross-shot.** If the
onset were flux-limited, the delivered flux would cluster more tightly than
the onset time. It does not: flux CV 0.147 against onset-time CV 0.118, a
clustering ratio of **0.81**. The flux is a *worse* predictor than the clock,
and both are far worse than I_p/I_p,max.

**H3 — runaway electrons. Contradicted.** The naive resistance test does
report an apparent collapse (`R_in/R_pre = 0.10`) — but that test assumes the
current is freewheeling, and here it is *rising*. The test is now flagged
invalid whenever dI_p/dt > 0 during the event, and the inductive spike explains
the apparent resistance collapse with no runaway population needed. This was a
genuine false positive on the first pass and is exactly why the validity guard
exists.

**H7 — chamber-correction artefact. Contradicted, but it was close.** The
correction swings by 0.82 kA per event. Redoing it with a chamber model that
respects the vessel's own L/R (0.124 ms, comparable to the event duration)
leaves the current excursion **larger**, not smaller
(`dip_ratio_dyn_pub = 1.62`), and the excursion does not recover between events
(`ip_recovery = −0.54`). The staircase is real lost plasma current.

**H4 — radiative/recycling limit cycle. Not decidable here.** It scores +0.50
purely on the ~1 ms period being in the right range; #53740 had no fast
spectroscopy, so the decisive lead/lag of impurity radiation against the
collapse could not be measured. Density does modulate strongly in phase with
the cycle, so a radiative contribution to the *cooling* between crashes is
plausible and untested.

**H6 — position excursion. Not necessary.** #53740's radial and vertical
stabilisation waveforms do ramp right through the onset, which looked
suspicious. But across the candidates the radial drive at onset ranges from
−11.3 to +11.8 with the widest scatter of any quantity examined
(median 3.75, IQR 5.74, coefficient of variation 1.11), and in **8 of the 37
shots it is exactly zero at onset** — the instability starts before the ramp
begins. One shot (#53723) had current
stabilisation switched off entirely and still shows it.

## 6. Where my pre-data prior was wrong

Before the data was reachable I argued that, because U_loop is a *core*
measurement and a plasma crash should spike it *upwards*, the leading
candidates were circuit-side (H1/H5). The first half of that reasoning was
right and the conclusion drawn from it was wrong: the crash spikes U_loop
**downwards**, because the plasma current *increases* at constant flux and the
extra secondary MMF opposes the core flux change. I had the sign of the
back-reaction backwards. The cross-shot volt-second test I proposed as the
decisive check for H5 did its job — it refuted the hypothesis it was built for.

## 7. What is still open

- **No Mirnov data on any of these shots**, and §7a shows the archive cannot
  supply it: the ring recorded nothing in this campaign, and where it did
  record, the instability is weaker and the ring is not calibrated well enough
  to give a trustworthy mode number. The m = 2 versus m = 1 question is open.
- **No fast spectroscopy**, so H4's lead/lag test is untested.
- **q_0 is not measured**, so "the profile finished penetrating" is an
  inference from the I_p/I_p,max law, not a direct observation. A diamagnetic
  loop (`Diamagnetism`, present on other shots) would give the stored energy
  and constrain l_i independently.
- The 37 candidates span only two sessions; a wider scan (`--since-days 365`)
  would show whether the I_p/I_p,max law holds generally.

## 7a. The Mirnov survey: the decisive measurement is missing from the archive

§7 named the precursor mode number as the most valuable missing measurement.
It was searched for directly.

**Survey.** `MHDring-TM` is listed as active on **231 plasma shots** in the
sampled range (June–August 2026). Of those, **206 have real ring data** — but
**all 20 shots of the August campaign** (sessions 53712, 53714 and 53747, the
same run as #53740) have **zero-byte** `ring_*.csv`, `U_mc*.csv` and
`LimSaddleCoil.csv`. The upstream `DAS_raw_data_dir/rawData.csv` is absent
there too (19 MB on a working shot), and `ErrorLog` is empty, so the NI-6133
acquisition failed silently and nothing can be recovered. **In the regime where
this instability is fully developed, the Mirnov ring never recorded.**

**Nearest usable shots.** Running the detector over the 206 shots that do have
ring data gives 42 with at least one collapse and 9 candidates, of which 11
have ≥ 2 events. These are a *weaker* version of the phenomenon — 2–3 events
per shot against 10 on #53740 — and their precursors are correspondingly faint
(`precursor_ratio` median 1.23, range 0.90–1.75, i.e. barely above the quiet
level).

**Mode number: not determined, and the reason matters.** Three obstacles, all
in the diagnostic rather than the physics:

- the 16 ring channels are **not gain-matched** — RMS spans 0.10–1.78 V on one
  shot, an 18× range — and several clip at the ±10 V digitiser rail;
- `MHDring-TM.sh` shows `ring_i` is simply DAS column *i+1*, with **no
  documented mapping from channel index to poloidal angle**, so the equispaced
  assumption behind a phase-slope fit is unverified;
- consequently the fitted `m` is not robust: it moves from ≈1.9 to ≈0.3 purely
  on analysis choices (frequency band, per-coil normalisation).

A gain-insensitive cross-check was therefore built from the four separately
wired quadrature coils `U_mc1/5/9/13` (0°, 90°, 180°, 270°, one amplifier
group), where the mode number follows from two correlations alone —
`r_180 = corr(mc1, mc9)` and `r_90 = corr(mc1, mc5)` — with
`(+1, +1) → m = 0`, `(−1, 0) → m = 1`, `(+1, −1) → m = 2`. Across the 11 shots
it returns weak correlations (`r_180` 0.24–1.00, `r_90` −0.00–0.78) that sit
closest to **m = 0** in 8 of 11 shots but never near the ideal ±1 values. Read
honestly: **there is no dominant coherent rotating mode in the 2–50 kHz band
before these crashes**, at least not in these weak-instability shots.

So the m = 2/n = 1 (minor disruption) versus m = 1/n = 1 (sawtooth) question
**remains open**. What was learned is that it cannot be answered from the
existing archive, and exactly what a future run must fix.

**What the June/July shots *do* independently confirm.** They are a different
campaign, different sessions, different toroidal field and a different
diagnostic set from #53740, and they reproduce both key results:

| quantity | August campaign (#53740 et al.) | June/July MHD shots |
|---|---|---|
| I_p/I_p,max at onset | 0.963, IQR/median 3.1 % (n = 37) | **0.974, IQR/median 1.1 %** (n = 10) |
| Δl_i per crash | −0.072 | −0.036 (range −0.071 … +0.13) |

The onset law is therefore not an artefact of one campaign.

**Recommended next run.** Repeat the August regime (`PlasmaRegimeOptimization`
settings: U_CD = 400 V, U_Bt = 800 V, H₂ at p = 13) with `MHDring-TM`
*verified to be acquiring* — the failure is silent, so check that
`ring_1.csv` is non-zero before trusting the shot. Worth fixing at the same
time: gain-match the ring channels or publish their calibration, and document
the channel-to-poloidal-angle map. Adding `Diamagnetism` would give the stored
energy and constrain l_i independently, and `FastSpectrometry` would settle H4.

## 8. Reproducing this

```bash
pip install -r golem_instability/requirements.txt

python -m golem_instability shot 53740 --outdir out/
python -m golem_instability scan --from 53500 --to 53761 --outdir out/
python -m golem_instability selftest        # offline pipeline validation

# find shots carrying a given diagnostic before pulling any waveform
python -c "
from golem_instability import GolemFetcher, FetcherConfig
from golem_instability.scan import scan_diagnostics
f = GolemFetcher(FetcherConfig(cache_dir='golem_cache'))
d = scan_diagnostics(f, range(53000, 53762), required=('MHDring-TM',))
print(d.groupby('session_id').size())"
```

The scan writes `scan.csv`, `onset_conditions.csv`, `scan_report.md` (with the
onset-criterion ranking and the volt-second test), `scan_overview.png` and
`onset_criterion.png`. Each analysed shot gets an overview, an event zoom, an
event-triggered average, a statistics panel, a spectrogram, a signal inventory,
an events CSV and a markdown report ranking H1–H7 with the numbers behind each
verdict.

Note the base URL is **https** — plain HTTP is blocked by some egress proxies
while HTTPS works.

## 9. Sources

- [Online experimentation at the GOLEM tokamak](https://golem.fjfi.cvut.cz/wiki/Education/library/paper.pdf) — machine parameters, diagnostic set, `U_loop` from a coil around the transformer core.
- [Plasma current measurement using the Rogowski coil (GOLEM wiki)](http://golem.fjfi.cvut.cz/wiki/Diagnostics/Magnetic/RogowskiCoilArroundChamber/Theory/diagnostics_rogowski) — the chamber-current correction and `R_chamber = 9.7 mΩ`.
- [Instructions for student measurements on the GOLEM tokamak](http://golem.fjfi.cvut.cz/wiki/TrainingCourses/Universities/BUTE_Budapest.hu/10/GOLEM_instructions.pdf) — `C_CD = 11.3 mF`, `U_CD ≤ 700 V`, thyristor switching.
- [Investigation of turbulent transport at the edge of the GOLEM tokamak](https://arxiv.org/pdf/2502.10191) — operating ranges.
- [An active feedback recovery technique from disruption events induced by m=2/n=1 tearing modes](https://arxiv.org/pdf/1503.00852) — minor-disruption phenomenology, including the positive I_p spike at a crash.
