# Rosati & Littenberg (2024) template cases: run notes

Runs of `rosati_littenberg_cases.py` on 2026-11-15: an injection recovery and
an upper limit for each SGWB template studied in Rosati & Littenberg (2024),
arXiv:2410.17180, using this repo's noise blocks. **Provisional:** one noise
realization per case and short chains. Read these as a smoke test of the setup,
not as calibrated limits.

Reproduce (about 19 minutes and a 7.6 GB peak memory footprint; see Memory):

```sh
uv run python rosati_littenberg_cases.py            # the six cases below
```

Each case writes `<case>.json` (summary) and `<case>.npz` (thinned chain,
`chain` (steps, walkers, parameters) and `names`; not committed).

## Setup

| Item | Value |
|---|---|
| Duration, cadence | one year, dt = 5 s |
| WDM grid | Nf = 8640, Nt = 730, so ΔT = Nf·dt = 12 h and layers are 11.6 µHz wide |
| Fitted region | 0.1–8 mHz (683 layers) × 724 columns (3 cropped at each end) |
| Coarse cells | Q = 100 columns (50 days): 8 cells, the last 24 columns long |
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

Injection recovery (median and 90% interval, sampling coordinates):

| Case | Parameter | Truth | Recovered |
|---|---|---|---|
| det-powerlaw | log10 A | −8.45 | −8.98 [−9.87, −7.75] |
| | α | 0.667 | 0.52 [0.28, 0.86] |
| | log10 Ω(3 mHz) | −11.06 | −11.03 [−11.14, −10.96] |
| det-lognormal | log10 A | −2.500 | −2.508 [−2.522, −2.493] |
| | log10 f⋆ | −3.699 | −3.699 [−3.704, −3.694] |
| | log10 Δ | −0.523 | −0.527 [−0.545, −0.510] |
| det-pt | r_b | 0.15 | 0.25 [0.12, 0.41] |
| | b | 0.65 | 0.33 [0.04, 0.65] |
| | log10 A | −10.40 | −10.41 [−10.47, −10.34] |
| | log10 f_ph | −3.699 | −3.75 [−4.01, −3.67] |

Every injected template parameter is inside its 90% interval. The power-law
amplitude and slope trade off along a ridge; Ω at 3 mHz is the combination the
data pin down.

95% upper limits, with no background in the data (a KDE of the template's
posterior, evaluated along the amplitude at the conditioning values):

| Template | This setup | Paper, synthetic confusion, 12 STFTs | Paper, no confusion |
|---|---|---|---|
| Power law, log10 A_p | −9.47 | −9.76 | −10.41 |
| Log-normal, log10 A_R | −3.70 | −3.00 | −3.40 |
| Phase transition, log10 A_ph | −12.06 | −11.36 | −11.46 |

The limits fall within 0.7 decades of the paper's. They are not expected to
match: see Differences from the paper.

Convergence: the largest drift between the two halves of the kept chain is
0.2–0.4 posterior standard deviations in every case. No autocorrelation times
were computed.

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
cycles. Of that, the conditioned data are 1.4 GB, freed filter temporaries
kept by malloc for reuse about 2.6 GB, and each covariance validation (a
Cholesky factor at all 6.3 M grid pixels, repeated every Wheel cycle) adds a
~3 GB transient. Peak over the six-case run: 7.56 GB. Two such processes at
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

**Recommendation: Q = 8** (4-day cells) at ΔT = 12 h. Every parameter keeps
over 98% of its pixel-level precision and coverage is exact to 0.1%, so the
weights need no Welch–Satterthwaite correction there. Cost: the Q = 10 rerun
above (55k terms) took about 20 s per 1000-step cycle against about 15 s at
Q = 100, because a cycle's fixed overhead (Enchilada's covariance validation)
dominates. Q = 16 is the cheaper fallback (6% loss on the narrow templates).
For Q ≳ 30, use Welch–Satterthwaite weights. The ragged note's per-band
regions would cut cost further, since instrument-dominated bands lose nothing
to long cells, but the per-cycle overhead makes that unnecessary for now.

The upper limits and injections above used Q = 100, so the narrow-template
limits are looser than this setup allows. The power-law limit is barely
affected.
