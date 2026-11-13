# noisy-enchilada

**Provisional: needs to be verified.** The noise blocks, proposal mixture,
soft OMS constraint, and preliminary mixing estimates are experimental. Passing
unit tests and short brick checks do not establish scientific validity or
convergence; verify normalization, posterior recovery, and mixing before use
in an analysis.

A short Enchilada demo for two unequal-arm instrument-noise amplitudes, using
Eryn and the `wdm` package. Run with Python 3.14:

```sh
uv sync
uv run python inst_noise_demo.py
uv run python -m unittest -q
```

Edit the constants at the top and the short campaign at the bottom of
`inst_noise_demo.py` to change the input, grid or run length. NumPy is pinned
below 2.4 for Eryn's use of `np.in1d`.

Enchilada is sourced from [rjrosati/enchilada's `cd1l-orbit-arrays` branch](https://github.com/rjrosati/enchilada/tree/cd1l-orbit-arrays), with the exact commit recorded in `uv.lock`.
`L1Data.from_cd1l` reads fractional-frequency XYZ data and the raw orbit and
six directed delay arrays. The local `column_delays` helper averages those
arrays onto WDM time columns; no separate HDF5 or orbit reader is needed.
The reader also loads optional clock, TTL, and noise-estimate tables, so use
a short duration/grid when trying large bricks.

The demo retains the reference sampler's conditioning and layer calibration,
with a 0.3–8 mHz band and cropped edge columns.
The likelihood includes XYZ correlations within each pixel and approximates
pixels as independent. WDM coefficients are used directly; a flat one-sided
PSD gives variance S/2.

Set `coarseQ` on `InstrumentNoiseBlock` (default 1) to average consecutive
retained time columns. The Numba kernel weights each cell by its actual size.
For Q>1, covariance bases are averaged in each cell and that same constant
covariance is published on its constituent fine pixels. This approximates the
time variation; Q=1 retains the fine model. Numba compilation occurs on first use.

The block publishes `BlockState` with `Covariance` and physical parameters,
keeping walkers and iteration counts in its own `sampler_state`. Use a separate
block instance for each campaign. It refreshes likelihoods for each conditional
residual and uses Wheel's RNG. Covariance uses unit-variance placeholders outside
the fitted band and cropped columns to meet Enchilada's full-grid requirements;
those pixels are excluded from this block's likelihood. The short campaign runs ten cycles.
A future separate foreground block needs covariance composition support:
Enchilada currently replaces the covariance rather than summing noise blocks.

Alternative noise blocks are in `noise_blocks.py`. Register one noise block per
comparison Wheel (their covariances replace one another):

```python
from noise_blocks import InjectionNoiseBlock, CD1LNoiseEstimateBlock, LinkNoiseBlock

# Choose one:
block = InjectionNoiseBlock(data, delays, amplitudes=(1.5e-11, 3e-15), coarseQ=32)
# block = CD1LNoiseEstimateBlock(data, coarseQ=32)
# block = LinkNoiseBlock(data, delays, nwalkers=32, steps_per_call=100, coarseQ=32)

initial = block.initial_result(rng=np.random.default_rng(42))
wheel = Wheel(data, random_seed=42)
wheel.add(block, initial_block_state=initial, data_domain="wdm")
wheel.run(10)
```

`InjectionNoiseBlock` fixes the two shared ASDs to the supplied values (default:
the demo's fiducial values). The current L1Data does **not** store injection
amplitudes, so these defaults are not read from file metadata. It uses L1Data's
directed delay arrays when `delays` is omitted.

`CD1LNoiseEstimateBlock` uses `data.noise_estimates_xyz`, divides the raw spectra
by `data.laser_frequency_hz**2`, and integrates the real symmetric XYZ spectral
matrix through the WDM filter (a flat one-sided PSD gives S/2). It interpolates
matrix entries linearly in log frequency and linearly in time at WDM column
centres, holding the nearest time estimate outside the tabulated range. It
rejects missing spectra, insufficient frequency coverage, and invalid spectral
matrices. Imaginary CSD and correlations between WDM pixels are omitted, as in
the existing real per-pixel likelihood. `coarseQ` averages consecutive retained
columns. Both fixed blocks publish the same covariance on every call without
sampling or consuming RNG draws.

`LinkNoiseBlock` samples 12 stationary log ASDs: `Soms_d_{link}` followed by
`Sa_a_{link}`, each in link order `(12, 23, 31, 13, 32, 21)` (received at i from j).
The acceleration label denotes the test mass entering eta_ij directly and the
reverse eta with a delay. It projects independent link spectra through the
TDI-2 transfer functions using SEGWO's `C = M diag(S) M†` construction, retaining
the same spectral shapes, band, time cropping, WDM calibration and coarse
likelihood as `InstrumentNoiseBlock`. It defaults to 32 walkers and requires at
least 24. Individual link amplitudes can be weakly constrained or degenerate
with each other; the two-parameter mixing estimates do not describe this model.
All blocks publish unit-variance placeholders outside the comparison region;
use their `active_mask` when comparing likelihoods.

`LinkNoiseBlock` now uses a 50/50 mixture of the stretch move and a small
GLASS-inspired move. The latter chooses equally between a single log-ASD step,
a reverse-link power exchange (OMS or acceleration), and an acceleration cycle
in direction `(+,+,+,-,-,-)` in our link order. Step scales span 0.001–1 times
nominal powers (or log-ASD units for local steps). Power moves include the
Hastings correction for log-ASD coordinates; negative powers are rejected.
Use `targeted_proposals=False` for the stretch-only comparison.

An optional soft OMS constraint penalizes extreme reverse-link splits while
keeping all six OMS amplitudes independent:

```python
block = LinkNoiseBlock(
    data, delays,
    oms_log_power_ratio_sigma=np.log(2),
)
```

For each reverse pair it adds `-0.5 * (ln(P_ij/P_ji) / sigma)**2` to the log
prior, on top of the existing log-ASD bounds. Here `P = ASD**2`; `sigma=ln(2)`
sets a factor-of-two power-ratio scale, before the bounds and data are applied.
This is an explicit prior assumption, not a claim of equal OMS powers. The
default `None` leaves the original independent log-uniform prior unchanged.
The prior's normalization is omitted because these runs use fixed sigma and
MCMC, not model-evidence calculations. Both move choices support the same prior.
