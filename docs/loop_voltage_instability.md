# Repetitive loop-voltage collapse on GOLEM

*Working note for the instability first noticed on shot #53740, where roughly
20 ms into the discharge the loop voltage begins repeatedly dropping to ~0 and
recovering, with the plasma current stepping down a little each time.*

> **Data status.** The analysis code in `golem_instability/` is complete and
> validated end-to-end, but this session had no network route to
> `golem.fjfi.cvut.cz` (blocked by the environment's egress policy), so **no
> real shot data has been loaded and none of the numbers below are
> measurements**. Everything here is either machine documentation, circuit
> physics, or a prediction to be checked by running the toolkit. The commands
> that produce the answers are given in §7.

---

## 1. What the two key signals actually are

This matters more than usual, because the naive reading of the phenomenon
("the plasma stopped being driven, so it lost current") assumes the two traces
are independent measurements. On GOLEM they are not.

**Loop voltage.** `U_loop` comes from *a coil wound around the transformer
core*, not from a flux loop threading the plasma. It measures the core's
`dΦ/dt`, i.e. the electromotive force the transformer is delivering to
everything linked by the core: the plasma ring, the vacuum vessel, and any
other conducting path. So `U_loop → 0` means **the transformer stopped changing
its flux**, which is a statement about the primary circuit and the core first,
and about the plasma only through back-reaction.

**Plasma current.** The Rogowski coil sits around the vessel and therefore
measures the *total* toroidal current: plasma plus the current induced in the
stainless-steel chamber. The published `Ip` is obtained by subtracting the
chamber contribution, estimated resistively as

```
I_chamber = U_loop / R_chamber ,     R_chamber = 9.7 mΩ
```

Two consequences run through everything below:

1. `Ip` **contains** `U_loop`. A step in `U_loop` writes a step into `Ip`
   whether or not the plasma did anything. With a 5 V loop voltage the
   correction term is `5/0.0097 ≈ 515 A`, which is the same order as the
   "slight" current drops being reported.
2. The chamber path has its own inductance, so its current relaxes on
   `τ_ch = L_ch/R_ch ≈ 0.12 ms` (for `L_ch ~ 1.2 μH`). For transients faster
   than that the resistive estimate is simply wrong, and the correction
   over- or under-shoots at every edge.

This is not a reason to dismiss the phenomenon — the loop-voltage collapse is
certainly real — but the *size* of the current staircase has to be re-derived
before it is interpreted. `analyze.chamber_correction_test()` does exactly
that: it undoes the resistive correction to recover the raw Rogowski current,
re-applies one that respects the chamber's `L/R`
(`analyze.chamber_current_model()`), and reports how much of the staircase
survives (`dip_ratio_dyn_pub`).

## 2. Machine numbers used below

| quantity | value | note |
|---|---|---|
| major / minor radius | R₀ = 0.4 m, a = 0.085 m | Mo poloidal limiter |
| plasma current | I_p < 8 kA | |
| toroidal field | B_t < 0.5 T typical (≤0.8 T possible) | 81 mF bank, ≤1.3 kV |
| discharge length | τ_p < 25 ms | |
| current-drive bank | C_CD = 11.3 mF, U_CD ≤ 700 V | thyristor-switched into the primary |
| transformer | iron core | finite volt-second budget |
| chamber resistance | R_chamber = 9.7 mΩ | used in the I_p correction |
| plasma inductance | L_p ≈ 1.2 μH | μ₀R₀(ln(8R₀/a) − 1.75 + lᵢ/2) |
| ohmic resistance | R_p ≈ U_loop/I_p ≈ 3–5 mΩ | at 15 V / 4 kA |
| plasma L/R | ≈ 0.3 ms | sets how fast I_p decays with the drive off |

The plasma `L/R` of ~0.3 ms is the single most useful number: with the drive
removed, the current decays by ~3 % per 10 μs. A collapse lasting 100 μs
therefore costs of order 30 % of the current *if the plasma resistance is
unchanged*, and correspondingly more if the plasma also cooled. Measuring the
actual decay rate inside the collapse is what turns this into a diagnostic.

## 3. The circuit equation, and what "U_loop = 0" implies

For the plasma ring,

```
U_loop = R_p I_p + L_p dI_p/dt
```

so inside a collapse, where `U_in ≈ 0`,

```
R_in = (U_in − L_p dI_p/dt) / I_p
```

is directly measurable, and can be compared with the pre-collapse ohmic value
`R_pre ≈ U_env/I_p`. The ratio `R_in/R_pre` is the primary fork in the road:

| `R_in/R_pre` | reading |
|---|---|
| ≈ 1 | the plasma is unchanged; it simply lost its drive → **circuit-side cause** |
| ≫ 1 | the plasma cooled or disrupted → **plasma-side cause** |
| ≪ 1 | the resistance collapsed → **runaway-dominated current** |

This is `analyze.resistance_test()`. Note that `R_in` scales linearly with the
assumed `L_p`, so a factor-of-2 error in `L_p` matters for the absolute value
but not for telling ~1 from ≫1.

## 4. What to measure (the characterisation)

`detect.detect_collapses()` finds the events as excursions of `U_loop` below a
fraction of its own running upper envelope — envelope-relative rather than
absolute, because `U_loop` decays throughout the discharge and a fixed
threshold would either miss early events or fire on the natural end-of-shot
tail. For each event it reports onset, duration, depth, the missing flux
`∫(U_env − U) dt`, the current step, and the implied `L/R`. Shot-level it
reports the repetition period and its coefficient of variation, the duty cycle,
the staircase fraction, and the flux delivered before the first event.

Across diagnostics (`analyze.analyse_shot()`), every trace gets an
event-triggered average and a robust response measured in units of its own
pre-event scatter, together with the **lead/lag of its first 3σ excursion
relative to the collapse onset**. That lead/lag is the cause-versus-consequence
discriminator: a quantity that rises *before* the loop voltage falls can be
driving it; one that rises after is responding to it.

## 5. Candidate mechanisms

Ranked by how well they fit what is known so far. Each entry gives the
signature that would confirm it and the measurement that kills it.

### H1 — Drive interruption in the primary circuit *(leading candidate)*

The current-drive bank is switched into the primary by thyristors. A thyristor
conducts only while its current exceeds the holding value and cannot be turned
off by the gate; as the 11.3 mF bank runs down late in the discharge the
primary current is small and near its zero crossing, which is precisely where
intermittent conduction, drop-out and re-triggering occur. Each non-conducting
interval gives `dΦ/dt = 0` — a flat `U_loop = 0` floor — during which the
plasma current freewheels through `L_p/R_p` and loses a little, then the drive
returns.

- **Predicts:** `R_in/R_pre ≈ 1`; a flat, clock-regular collapse train
  (`period_cv < 0.15`); a hard floor rather than a noisy minimum
  (`rail_fraction` high); no Mirnov precursor; Mirnov jumps in phase on all
  coils (m = 0 pickup, `mirnov_common_mode_ratio > 1`); nothing happening on
  H-alpha before the collapse; onset late, when the bank is nearly flat.
- **Killed by:** a growing magnetic precursor, or `R_in/R_pre ≫ 1`.
- **Cross-shot check:** the onset should track the CD bank charging voltage
  `U_CD` (from the shot's `.setup` / `Production/Parameters`), not the clock.

### H2 — Repetitive minor disruptions / MHD relaxation

The textbook candidate for "current drops in steps": an m = 2/n = 1 tearing
mode grows, crashes, flattens the current profile and expels particles and
energy; the cycle repeats.

There is, however, a specific reason to doubt it *as the driver here*. A
disruption raises the plasma resistance and reduces the secondary current,
which makes the core flux change **faster**, not slower — the classic magnetic
signature of a minor disruption on a core-mounted loop is a positive `U_loop`
spike (plus a negative `I_p` spike), not a collapse to zero. So H2 predicts the
wrong sign for the primary observation, and would more naturally explain a
*response* at each event than the event itself.

- **Predicts:** Mirnov precursor growing over a few hundred μs
  (`precursor_ratio > 2`), coherent m ≈ 2 at ~5–20 kHz, `R_in/R_pre ≫ 1`,
  H-alpha recycling flash *after* the onset, density expelled, hard X-rays.
- **Killed by:** quiet Mirnov coils before each event, or `R_in/R_pre ≈ 1`.

### H3 — Runaway-electron dominated phases

If a runaway population carries a growing share of the current, the effective
resistance collapses and the loop voltage needed to sustain the current falls
towards zero; a loss event (mode, wall contact) dumps the runaways, the
resistance and `U_loop` jump back, and the current steps down. Low density and
late-discharge timing both favour it.

- **Predicts:** `R_in/R_pre ≪ 1`; scintillator/Timepix bursts at the recovery,
  not at the onset; low `n_e`; `U_loop` recovery *overshooting* its envelope.
- **Killed by:** quiet scintillators, or a high-density discharge.

### H4 — Impurity-radiation / recycling limit cycle

Impurity influx raises radiated power, the edge cools, the current profile
contracts, confinement drops, the influx is quenched, the plasma reheats — a
thermal relaxation oscillation on the confinement/refuelling timescale
(≳0.5 ms).

- **Predicts:** C II / O I / N II rising **before** the loop-voltage drop
  (`CII_514_onset_lag_ms < 0`), `R_in/R_pre > 1`, slow and somewhat irregular
  periods.
- **Killed by:** line emission that only ever follows the collapse.

### H5 — Volt-second exhaustion / iron-core saturation

GOLEM's transformer has an iron core with a finite flux swing. As the core
approaches saturation its magnetising inductance collapses, the magnetising
current surges, the bank voltage sags and `dΦ/dt` — hence `U_loop` — drops. As
the flux falls back below the knee the core recovers and the voltage returns.
That is a genuine relaxation oscillator, and it explains a *periodic* collapse
train without any active electronics.

- **Predicts:** onset set by the *flux already delivered* rather than by the
  clock. This is the decisive and rather elegant cross-shot test: compute
  `∫U_loop dt` up to the first collapse in every candidate shot and compare its
  scatter with the scatter of the onset time. Tight flux clustering with
  scattered onset times means the core is the culprit
  (`analyze.volt_second_test()`, reported as `flux_clustering_ratio`).
- **Killed by:** onsets that cluster in *time* while the delivered flux varies.

### H6 — Position excursion with limiter contact

A vertical or radial drift brings the column onto the limiter, cooling it and
releasing wall material; the stabilisation system pushes it back and the cycle
repeats.

- **Predicts:** growth of the m = 1 component of the Mirnov ring before each
  event (`ring_asymmetry_growth`), impurity lines leading H-alpha,
  `PositionStabilization` actuator activity in step with the events.
- **Killed by:** a quiet m = 1 component.

### H7 — Measurement / processing artefact

Two distinct versions, and the second is genuinely likely to contribute:

1. *Digitiser clipping* of the loop-voltage channel — a dead-flat rail with no
   scatter (`rail_fraction ≈ 1`, `in_event_std ≈ 0`). Easy to check against the
   raw scope dump (`Shot.raw_das()`).
2. *The chamber-current correction* (§1) writing part of the `I_p` staircase.
   Since `τ_ch ≈ 0.12 ms` is comparable to the collapse duration, the resistive
   correction cannot follow and mis-corrects at every edge, by up to
   `U_env/R_chamber ≈ 0.5 kA`.

- **Predicts (version 2):** the excursion largely disappears when the
  correction is redone with an inductive chamber model
  (`dip_ratio_dyn_pub ≪ 1`), and the current excursion recovers between events
  rather than accumulating (`ip_recovery → 1`).
- **Killed by:** a staircase that survives the re-correction, which would make
  it real lost plasma current.

## 6. Decision tree

```
                      R_in / R_pre  (resistance_test)
                               |
      ┌────────────────────────┼────────────────────────┐
     ≈1                       ≫1                       ≪1
      │                        │                        │
 circuit-side             plasma-side              runaway-side
      │                        │                        │
 precursor_ratio?        precursor_ratio?         scintillators?
   low → H1/H5            high → H2                burst at recovery → H3
      │                    low  → H4/H6
 flux clustering
 across shots?
   tight → H5
   loose → H1
```

Run in parallel with the tree, on every shot: `dip_ratio_dyn_pub` and
`rail_fraction`, to know how much of the `I_p` staircase is real (H7).

## 7. How to get the answers

```bash
pip install -r golem_instability/requirements.txt

# the reference shot: figures + a full markdown report with the hypothesis table
python -m golem_instability shot 53740 --outdir out/

# find the other shots that do this, then report on the best candidates
python -m golem_instability scan --from 53500 --to 53800 --analyse-top 5 --outdir out/

# a wider trawl, e.g. the last year, sampling every 3rd shot
python -m golem_instability scan --since-days 365 --step 3 --outdir out/

# no network needed: validate the pipeline against synthetic shots of known cause
python -m golem_instability selftest
```

The scan writes `scan.csv`, `scan_report.md` (including the volt-second test)
and `scan_overview.png`; each analysed shot gets an overview, an event zoom, an
event-triggered average, a statistics panel, a spectrogram, an events CSV and a
markdown report ranking H1–H7 with the numbers behind each verdict.

### Suggested order of work once the data is reachable

1. `shot 53740` — read `R_ratio`, `rail_fraction`, `precursor_ratio` and
   `dip_ratio_dyn_pub`. Those four numbers already split the tree.
2. `scan` a few hundred neighbouring shots. The instability is expected to be a
   session-level phenomenon, so look for clustering by `session_id` and by
   `U_CD` rather than a uniform sprinkle.
3. Run `volt_second_test` on the candidates (H5), and correlate the onset with
   `U_CD`, `B_t`, gas species and pressure from the `.setup` files.
4. Only then reach for the raw scope dumps, to confirm the `U_loop` floor is
   physical and to look at the uncorrected Rogowski channel directly.

## 8. Prior expectation

Stated up front so it can be falsified rather than quietly confirmed: given
that `U_loop` is a *core* measurement, that a plasma-side crash would spike it
upwards rather than to zero, and that the onset is late in a discharge whose
capacitor-driven, thyristor-switched primary is by then near its current zero,
the leading candidates are **H1** (intermittent primary conduction) and **H5**
(core saturation), which the cross-shot volt-second test separates. **H7.2**
(the chamber-correction artefact) is expected to account for a significant
fraction — possibly most — of the apparent `I_p` staircase, and must be
quantified before the current loss is interpreted. The plasma-side mechanisms
H2/H4 are expected to appear as *responses* to the collapse rather than its
cause; if a Mirnov precursor is found growing before each event, that ordering
is wrong and H2 moves to the front.

## 9. Sources

- [Online experimentation at the GOLEM tokamak](https://golem.fjfi.cvut.cz/wiki/Education/library/paper.pdf) — machine parameters, diagnostic set, `U_loop` from a coil around the transformer core.
- [Plasma current measurement using the Rogowski coil (GOLEM wiki)](http://golem.fjfi.cvut.cz/wiki/Diagnostics/Magnetic/RogowskiCoilArroundChamber/Theory/diagnostics_rogowski) — the chamber-current correction and `R_chamber = 9.7 mΩ`.
- [Instructions for student measurements on the GOLEM tokamak](http://golem.fjfi.cvut.cz/wiki/TrainingCourses/Universities/BUTE_Budapest.hu/10/GOLEM_instructions.pdf) — `C_CD = 11.3 mF`, `U_CD ≤ 700 V`, thyristor switching, `C_B = 81 mF`.
- [Remote operation of the GOLEM tokamak for fusion education](https://pure.mpg.de/rest/items/item_2381669_4/component/file_2418561/content) — capacitor-bank based supplies, remote control of the discharge.
- [Investigation of turbulent transport at the edge of the GOLEM tokamak](https://arxiv.org/pdf/2502.10191) — operating ranges (`B_t < 0.5 T`, `I_p < 8 kA`, `τ_p < 25 ms`).
- [An active feedback recovery technique from disruption events induced by m=2/n=1 tearing modes](https://arxiv.org/pdf/1503.00852) — minor-disruption phenomenology used for H2.
