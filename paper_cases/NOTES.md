# Rosati & Littenberg (2024) template cases: run notes

Runs of `rosati_littenberg_cases.py` on 2026-11-15: an injection recovery and
an upper limit for each SGWB template studied in Rosati & Littenberg (2024),
arXiv:2410.17180, using this repo's noise blocks. **Provisional:** one noise
realization per case and short chains. Read these as a smoke test of the setup,
not as calibrated limits.

The cases were run three times on the same data: with 50-day cells (Q = 100),
then 4-day (Q = 8) after `q_scan.py` showed Q = 100 discards much of the narrow
templates' information, then 2-day (Q = 4), the knee of the cost-information
trade (see Choosing Q). Reproduce (19, 29 and 36 minutes for Q = 100, 8 and 4,
with a peak memory footprint of about 8 GB; see Memory):

```sh
uv run python rosati_littenberg_cases.py                  # the six cases, Q = 8
uv run python rosati_littenberg_cases.py --coarse-q 4     # and Q = 4, Q = 100
```

Each case writes `q<Q>/<case>.json` (summary) and `q<Q>/<case>.npz` (thinned
chain, `chain` (steps, walkers, parameters) and `names`; not committed).

## Setup

| Item | Value |
|---|---|
| Duration, cadence | one year, dt = 5 s |
| WDM grid | Nf = 8640, Nt = 730, so ΔT = Nf·dt = 12 h and layers are 11.6 µHz wide |
| Fitted region | 0.1–8 mHz (683 layers) × 724 columns (3 cropped at each end) |
| Coarse cells | Q = 8 columns (4 days): 91 cells, the last 4 columns long. First set: Q = 100 (50 days), 8 cells, the last 24 columns long |
| Orbit | NOISE brick's directed delays, averaged per WDM column |
| Data | Gaussian draw at full time resolution (`synthetic.simulate`) from the truth model below; the brick supplies only the grid and orbit |
| Model | `StochasticNoiseBlock`: `UnequalArmEqualNoiseBlock` + modulated `GalacticForeground` + one SGWB template, sampled jointly |
| Sampler | Eryn stretch move, 64 walkers, 10 Wheel cycles × 1000 steps; first 4 cycles discarded, thinned by 10: 38,400 samples |
| Seed | 2024 for every case, so all six share one underlying noise draw |
| Spectral fold | 8 quadrature nodes per layer |

Truth model. Instrument ASDs are read off the paper's corner plots:
15 pm/√Hz and 3 fm/s²/√Hz for the power law, 7 pm/√Hz and 2.4 fm/s²/√Hz for
the other two templates. The confusion noise is the tanh model with
amp = 1.8e-44, fk = 2.58 mHz, alpha = 0.133, f_1 = 1 Hz and f_2 = 1/917 Hz,
modulated in time by `modulation_unequal.dat`.

| Template | Injection | Upper-limit conditioning (the paper's) |
|---|---|---|
| Power law | log10 A = −8.45, α = 2/3 (Radler LDC1-6) | α = 2/3 |
| Log-normal | log10 A = −2.5, f⋆ = 0.2 mHz, Δ = 0.3 | f⋆ = 0.2 mHz, Δ = 0.3 |
| Phase transition | r_b = 0.15, b = 0.65, log10 A = −10.4, f_ph = 0.2 mHz | r_b = 0.15, b = 0.65, f_ph = 0.2 mHz |

The paper gives no injections for the log-normal and phase-transition
templates. These use its conditioning shapes, with amplitudes chosen here an
order of magnitude in Ω above its synthetic-confusion limits. Prior boxes are
read off the paper's corner-plot axes.

## Results

Injection recovery (median and 90% interval, sampling coordinates; the same
data analysed with 50-, 4- and 2-day cells):

| Case | Parameter | Truth | Q = 100 | Q = 8 | Q = 4 |
|---|---|---|---|---|---|
| det-powerlaw | log10 A | −8.45 | −8.98 [−9.87, −7.75] | −9.14 [−9.92, −8.18] | −9.12 [−9.84, −8.24] |
| | α | 0.667 | 0.52 [0.28, 0.86] | 0.48 [0.27, 0.74] | 0.49 [0.30, 0.73] |
| | log10 Ω(3 mHz) | −11.06 | −11.03 [−11.14, −10.96] | −11.03 [−11.12, −10.96] | −11.03 [−11.11, −10.96] |
| det-lognormal | log10 A | −2.500 | −2.508 [−2.522, −2.493] | −2.503 [−2.511, −2.494] | −2.503 [−2.512, −2.495] |
| | log10 f⋆ | −3.699 | −3.699 [−3.704, −3.694] | −3.697 [−3.701, −3.694] | −3.697 [−3.700, −3.694] |
| | log10 Δ | −0.523 | −0.527 [−0.545, −0.510] | −0.530 [−0.542, −0.518] | −0.530 [−0.543, −0.518] |
| det-pt | r_b | 0.15 | 0.25 [0.12, 0.41] | 0.23 [0.11, 0.38] | 0.24 [0.12, 0.38] |
| | b | 0.65 | 0.33 [0.04, 0.65] | 0.51 [0.16, 0.68] | 0.51 [0.18, 0.68] |
| | log10 A | −10.40 | −10.41 [−10.47, −10.34] | −10.40 [−10.42, −10.37] | −10.40 [−10.42, −10.37] |
| | log10 f_ph | −3.699 | −3.75 [−4.01, −3.67] | −3.72 [−3.77, −3.68] | −3.72 [−3.76, −3.68] |

At every Q every injected template parameter is inside its 90% interval. The
power-law amplitude and slope trade off along a ridge; Ω at 3 mHz is the
combination the data pin down.

At Q = 8 the narrow templates' intervals shrink as `q_scan.py` predicts. The
phase-transition amplitude's interval narrows by ×2.33 (predicted ×2.34), the
log-normal amplitude's by ×1.64 (predicted ×1.76), and the power-law
amplitude's by ×1.22 (predicted ×1.16). The predicted factor is the ratio of reported widths,
(σ_true · σ_post/σ_true) at Q = 100 over Q = 8. The phase-transition peak
frequency narrows more than predicted (×3.7 against ×2.1): its Q = 100
posterior has a long tail, which a Fisher estimate does not capture. Going
on to Q = 4 narrows the intervals by 2–8% (the phase-transition amplitude by
3%, predicted 1.5%), and one widens by 3% (log-normal f⋆). Differences that
small are at the level of the chains' own sampling noise.

95% upper limits, with no background in the data (a KDE of the template's
posterior, evaluated along the amplitude at the conditioning values):

| Template | Q = 100 | Q = 8 | Q = 4 | Paper, synthetic confusion, 12 STFTs | Paper, no confusion |
|---|---|---|---|---|---|
| Power law, log10 A_p | −9.47 | −9.48 | −9.58 | −9.76 | −10.41 |
| Log-normal, log10 A_R | −3.70 | −3.80 | −3.80 | −3.00 | −3.40 |
| Phase transition, log10 A_ph | −12.06 | −11.99 | −11.85 | −11.36 | −11.46 |

The limits fall within 0.8 decades of the paper's. They are not expected to
match: see Differences from the paper.

**The limit estimator is too noisy to show a Q dependence.** Recomputing each
limit from the two halves of its own chain (split in time, or by walker)
moves it by 0.1–0.3 dex for the power law and log-normal, and by up to
0.65 dex for the phase transition (−12.52 and −11.87 from the two walker
halves at Q = 100). That is as large as every difference between Q in the
table, and as the 0.2–0.4 dex by which a Fisher argument predicts the Q = 100
limits should be looser. That prediction comes from the Godambe width of the
amplitude, with the shape conditioned, at the threshold amplitude: ×2.8
(log-normal) and ×2.6 (phase transition) at Q = 100 against the pixel
likelihood, and within 4% at Q = 8. So the earlier "limits barely depend on
Q, against the Fisher prediction" is not established either way.

The noise comes from the estimator, a KDE of the 2–4 template coordinates
evaluated at one conditioning point where samples are sparse, and its
bandwidth changes with sample count. A sharper test: rerun the upper limits
with the shape coordinates fixed at the conditioning values, so the limit is
the amplitude's 95th percentile with no KDE, and average over several noise
draws.

Convergence: the largest drift between the two halves of the kept chain is
0.2–0.4 posterior standard deviations in every case at Q = 100, 0.15–0.26 at
Q = 8 and 0.15–0.32 at Q = 4. No autocorrelation times were computed.

## Instrument-level offset: one noise draw, not the coarse cells

In all six cases the two instrument levels sat just outside their 90%
intervals, in the same direction: OMS ASD about 0.25% high and acceleration
about 1% low (roughly +2.4σ and −2.6σ). The six cases share one seed, so this
could be a single draw seen six times, or the 50-day cells could narrow the
posterior: they average the foreground modulation, which varies substantially
within a cell. The power-law detection was rerun twice:

| Run | OMS offset | Acceleration offset | log10 Ω(3 mHz) (truth −11.06) |
|---|---|---|---|
| seed 2024, Q = 100 (as above) | +2.4σ | −2.6σ | −11.03 [−11.14, −10.96] |
| seed 7, Q = 100 | −0.2σ | −1.4σ | −11.02 [−11.14, −10.93] |
| seed 2024, Q = 10 (5-day cells; 5 cycles, 2 discarded) | +2.3σ | −2.6σ | −11.03 [−11.12, −10.96] |

With another seed the truth is inside both intervals, and tenfold finer cells
leave the offsets unchanged: the offset belongs to the seed-2024 draw. Two
draws cannot establish coverage; that needs a few dozen realizations (about
3 minutes each, run one at a time).

**Correction.** An earlier version of these notes concluded from the unchanged
interval widths that Q = 100 shows no under-coverage. It does: the exact
calculation below shows that at Q = 100 the acceleration level's true scatter
grows by 13% while the size-weighted likelihood reports a posterior 11%
narrower than that scatter, so the two cancel in the reported width. The
narrow templates lose far more (next section).

## Memory

The first attempt ran five year-long processes in parallel and exhausted the
19 GB machine. The fixes:

- `L1Data.from_cd1l` loaded 24 TTL tables, 25 GB for a year of data. It now
  skips them unless `load_ttl=True` (`rjrosati/enchilada@0000cf5`, pinned in
  `uv.lock`).
- The cases run one after another in a single process, so the data load once.

Measured as the macOS physical footprint (RSS under-reports it by about half):
one case plateaus at 7.3 GB from its first Wheel cycle, with no growth across
cycles (8.0 GB at Q = 8 and 8.3 GB at Q = 4, where the six cases took 29 and
36 minutes against 19 at Q = 100). Of that, the conditioned data are 1.4 GB, freed filter temporaries
kept by malloc for reuse about 2.6 GB, and each covariance validation (a
Cholesky factor at all 6.3 M grid pixels, repeated every Wheel cycle) adds a
~3 GB transient. Peak over the six-case run: 7.56 GB at Q = 100, 8.04 GB at Q = 8. Two such processes at
once would not fit alongside ordinary desktop use.

## Differences from the paper

- XYZ with the orbit's unequal arms for the instrument, rather than equal-arm
  AET. Stochastic components use the equal-arm response at the nominal 2.5 Gm.
- The confusion modulation is a known table, not a free amplitude per STFT.
  This removes parameters, which may be why our limits on the narrow templates
  are tighter.
- The confusion spectrum is our tanh model. The paper's
  β·f·sin(κf) term is dropped (up to about 2× less power near the knee), and the
  injected amplitude A = 9e-45 is recalled from Cornish & Robson, not stated in
  the paper.
- The band reaches 0.1 mHz so the 0.2 mHz templates are in band (0.3 mHz
  elsewhere in this repo).
- The data are drawn from the analysis model itself (independent WDM pixels),
  so model approximations cannot show up as bias; see `synthetic.py`.
- No GLASS residuals or Gaussianity veto. The no-confusion, stationary-analysis
  and brick-based variants are defined in the script (`det-powerlaw-none`,
  `det-powerlaw-stationary`, `det-powerlaw-bricks`, `ul-*-none`) but were not
  run.

## Choosing Q (and checking ΔT)

`q_scan.py` scores the tiling and Q with the criteria of the WDM paper draft
(`../wdm-paper`: draft2 "Optimal choice of WDM tiling" and "Welch-like
coarse-graining", `tiling_heuristic.md`, `ragged_coarse_graining.md`,
`ws_wishart_route1.py`). It is exact for the model at each detection case's
truth, given independent pixels (as `synthetic.simulate` draws them), and needs
no sampling: about 30 s and a 7.8 GB peak footprint for all three cases.

For each Q it computes the Godambe information of the coarse likelihood (its
expected Hessian H and score variance J under the true per-pixel covariances).
That gives the actual scatter of the estimate (information, against the pixel
likelihood), the width the coarse posterior reports (coverage), and the
parameter directions lost, D = n − tr(F_fine⁻¹ G). Cell weights are either the
cell size (what the blocks use now) or the Route-1 Welch–Satterthwaite ν_eff.

**Tiling.** draft2 Eq. `rms_optimal_tiling` gives ΔT* = 16–18 h for these
spectra (per band from 14 h at 1–3 mHz to 81 h below 0.3 mHz). The pixel-level
cost of the diagonal approximation is about 4 nats per channel at ΔT = 12 h
and 3–4 nats at ΔT*: the optimum is flat and 12 h is fine. Once Q ≥ 2 the
neglected-dependence error (Result R3) drops to 10⁻⁵–10⁻⁸ nats, so the tiling
barely matters after coarse-graining. The synthetic data have no correlations
between pixels at all, so this axis only matters for real data.

**Information lost at each Q** (σ_true/σ_fine; ΔT = 12 h, so Q·ΔT = Q/2 days):

| Q | days | likelihood terms | power law log10 A | log-normal log10 A | phase transition log10 A | Sa_a (worst instrument) | D lost (pl / ln / pt) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.5 | 494,492 | 1 | 1 | 1 | 1 | 0 / 0 / 0 |
| 8 | 4 | 62,153 | 1.003 | 1.012 | 1.016 | 1.001 | 0.01 / 0.05 / 0.20 |
| 16 | 8 | 31,418 | 1.010 | 1.046 | 1.061 | 1.006 | 0.04 / 0.17 / 0.60 |
| 32 | 16 | 15,709 | 1.030 | 1.148 | 1.215 | 1.025 | 0.13 / 0.52 / 1.34 |
| 50 | 25 | 10,245 | 1.058 | 1.295 | 1.461 | 1.053 | 0.25 / 0.92 / 1.94 |
| 100 | 50 | 5,464 | 1.176 | 1.914 | 2.688 | 1.140 | 0.66 / 1.93 / 3.08 |
| 362 | 181 | 1,366 | 2.33 | 9.19 | 18.9 | 1.75 | 1.78 / 3.87 / 4.86 |

Why the narrow templates suffer: they peak at 0.2 mHz, where the modulated
foreground is comparable to the background. What separates a stationary
background from the foreground is the foreground's time modulation, and
averaging over a large part of its six-month cycle removes it. In the ragged
note's terms (Result R4), a cell is free only where the sensitivity
g = ∂ln S/∂ln θ is constant within it, and g = S_gw/S tracks M(t).

**Coverage with cell-size weights** (σ_post/σ_true): at least 0.99 for every
parameter up to Q = 16; at Q = 100 it falls to 0.89 for Sa_a, 0.93 for the
log-normal amplitude and 0.88 for the phase-transition amplitude. Route-1
Welch–Satterthwaite weights keep it within 0.97–1.06 up to Q = 181 (slightly
conservative).

**Recommendation: Q = 8** (4-day cells) at ΔT = 12 h, or Q = 4 to squeeze out
the last 1.5%. Q = 8 keeps every parameter within 2% of its pixel-level
precision, and coverage is exact to 0.1%, so the weights need no
Welch–Satterthwaite correction there. Use Welch–Satterthwaite weights for
Q ≳ 30.

Measured cost (det-pt data, 64 walkers, 6 Numba threads, one-off timing on
this 12-core laptop):

| Q | cell | terms | ms per likelihood call (32 walkers) | s per 1000-step cycle | D lost at threshold amplitude (pt / ln) |
|---:|---:|---:|---:|---:|---|
| 100 | 50 d | 5,464 | 8.0 | 19 | 3.47 / 2.98 |
| 16 | 8 d | 31,418 | 10.7 | 24 | 0.98 / 0.40 |
| 8 | 4 d | 62,153 | 14.9 | 30 | 0.36 / 0.11 |
| 4 | 2 d | 123,623 | 20.9 | 41 | 0.10 / 0.03 |
| 2 | 1 d | 247,246 | 34.3 | 61 | 0.02 / 0.006 |
| 1 | 12 h | 494,492 | 57.2 | 104 | 0 (plus the ~4 nats per channel of leakage on real data) |

A likelihood call is about 7–10 ms of spectrum evaluation (foreground and
template at 683 layers × 8 nodes per walker; independent of Q) plus the Numba
kernel over the cells (0.8 ms at Q = 100, 6.5 ms at Q = 8). Enchilada's
per-cycle overhead is only about 3 s. An earlier version of these notes
blamed Enchilada; that was wrong. Lost information grows roughly as Q² at
small Q (the ragged note's Result R5) while cost falls only as 1/Q towards
that floor, so the choice is a point on a Pareto front; Q = 4 is its knee.

**Tiling at matched cost.** Relative to each tiling's own pixel likelihood,
the information lost depends almost only on the cell duration Q·ΔT, and terms
scale with ΔT, which favours short ΔT. But at ΔT = 6 h the 23 µHz layers
already cost the narrow templates precision at the pixel level (4% on the
phase-transition amplitude, 1% on the log-normal). Absolute amplitude widths
(phase transition / log-normal, relative to 12 h at Q = 8) at matched budgets:

| terms | ΔT = 6 h | ΔT = 12 h | ΔT = 24 h |
|---|---|---|---|
| ~124k | Q=4: 1.019 / 0.991 | Q=4: 0.985 / 0.984 | Q=4: 1.009 / 1.000 |
| ~62k | Q=8: 1.023 / 0.995 | Q=8: 1 / 1 | Q=8: 1.071 / 1.056 |
| ~31k | Q=16: 1.040 / 1.010 | Q=16: 1.062 / 1.057 | Q=16: 1.30 / 1.25 |
| ~16k | Q=32: 1.108 / 1.069 | Q=32: 1.288 / 1.249 | worse |

ΔT = 12 h is the right tiling from ~60k terms up (consistent with draft2's
ΔT* = 16–18 h). ΔT = 6 h with longer Q only wins on small budgets.

Ways to do better than a uniform Q, not yet tried:

- Cheaper spectra. Half of each call at Q = 8 is evaluating smooth spectra on
  683 layers × 8 nodes. Fewer nodes, or evaluating on a coarser frequency grid
  and interpolating, would make small Q cheaper.
- Per-band cells (ragged_coarse_graining.md, Result R5). Where the foreground
  is negligible the noise is stationary, so sensitivities are constant in time
  and whole-year cells lose nothing. Not measured per band here. This needs the
  region and kernel to allow different cell counts per layer; they currently
  assume the same number.

The six cases were rerun at Q = 8 and Q = 4 (Results). The injections tightened
as predicted. The upper limits shifted by about 0.1 dex either way, within
their estimator's own noise.
