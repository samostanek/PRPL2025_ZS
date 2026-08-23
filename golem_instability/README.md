# `golem_instability` — repetitive loop-voltage collapse on the GOLEM tokamak

Toolkit for finding, characterising and explaining the instability in which the
loop voltage repeatedly drops to ~0 and recovers late in a GOLEM discharge,
with the plasma current stepping down at each collapse (first noticed on shot
#53740).

The physics write-up — the measured phenomenology, the seven candidate
mechanisms and how each was decided — is in
[`../docs/loop_voltage_instability.md`](../docs/loop_voltage_instability.md).
Short version: the events are fast internal reconnections (Δl_i ≈ −0.07 per
crash, 13 % of the density expelled), and across 37 candidate shots they start
when the plasma current reaches 96 % of its peak.

## Install

```bash
pip install -r requirements.txt
```

## Use

```bash
# one shot: figures + a markdown report ranking the hypotheses
python -m golem_instability shot 53740 --outdir out/

# search for other shots that do the same thing
python -m golem_instability scan --from 53500 --to 53761 --analyse-top 5 --outdir out/
python -m golem_instability scan --last 500 --outdir out/
python -m golem_instability scan --since-days 365 --step 3 --outdir out/

# validate the whole pipeline offline, against synthetic shots of known cause
python -m golem_instability selftest
```

Everything fetched is mirrored into `--cache` (default `golem_cache/`) in the
server's own directory layout, so a second run costs no network, and
`--offline` replays a scan with no connection at all.

As a library:

```python
from golem_instability import GolemFetcher, Shot, detect_in_shot, analyse_shot

shot = Shot(53740, GolemFetcher())
result = detect_in_shot(shot)                 # events + shot-level summary
analysis = analyse_shot(shot, result)         # metrics, verdicts, conditional averages

print(result.events[["t_on", "duration_ms", "depth", "d_ip"]])
for verdict in analysis["verdicts"]:
    print(verdict.key, verdict.verdict, verdict.name)
```

## What is where

| module | role |
|---|---|
| `config.py` | URLs and the signal registry (path, unit, meaning, owning diagnostic) |
| `fetch.py` | HTTP with retries and an on-disk mirror; 404 = "diagnostic did not run", not an error |
| `shotdata.py` | `Shot` (metadata, active diagnostics, plasma window, setup parameters) and `Signal`; parsers for both the 2-column CSV convention and raw digitiser dumps |
| `detect.py` | the collapse detector: envelope-relative threshold with hysteresis, per-event metrics, shot-level summary and ranking score |
| `analyze.py` | the physics: resistance test, MHD precursor and mode number, recycling lead/lag, chamber-correction artefact test, hypothesis scoring |
| `scan.py` | parallel cross-shot search, date→shot binary search, candidate selection, onset-condition and onset-criterion tests |
| `plots.py` | overview, event zoom, event-triggered averages, statistics, spectrograms, scan overview |
| `report.py` | markdown reports for a shot and for a scan |
| `synthetic.py` | fabricated shots with a *known* cause, written in the server's layout |
| `cli.py` | `shot` / `scan` / `selftest` |

## Why there is a synthetic-shot generator

The hypothesis tests are only worth anything if they can tell the mechanisms
apart. `synthetic.py` writes discharges whose cause is known by construction —
`circuit` (the drive is interrupted, plasma untouched), `plasma` (a tearing
mode crashes each time), `radiative` (impurity radiation leads the collapse),
`artefact` (the loop-voltage dropouts are real but the current staircase is
written by the chamber correction) and `clean` — into a cache tree, so the
normal loading path runs against them unmodified. `selftest` requires the top
hypothesis to come out right for each, and the test suite additionally checks
that the derived plasma resistance recovers the value that was put in.

```bash
python -m pytest golem_instability/tests -q
```
