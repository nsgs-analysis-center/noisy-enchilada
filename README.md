# noisy-enchilada

**Provisional: needs to be verified.** The noise blocks, proposal mixture,
soft OMS constraint, and preliminary mixing estimates are experimental. Passing
unit tests and short brick checks do not establish scientific validity or
convergence; verify normalization, posterior recovery, and mixing before use
in an analysis.

Short Enchilada demos for unequal-arm instrument noise, the galactic
foreground and stochastic backgrounds, using Eryn and the `wdm` package. Run
with Python 3.14:

```sh
uv sync
uv run python inst_noise_demo.py
uv run python foreground_demo.py
uv run python sgwb_demo.py
uv run python -m unittest -q
```

Edit the short campaign at the bottom of `inst_noise_demo.py` to change the
input, grid or run length. NumPy is pinned below 2.4 for Eryn's use of `np.in1d`.

| File | Contents |
|---|---|
| `noise_model.py` | TDI-2 spectral model, WDM fold, analysis region, Numba likelihood |
| `noise_blocks.py` | all noise blocks (hierarchy in the module docstring) |
| `noise_proposals.py` | `LinkNoiseMove` and the soft OMS ratio prior |
| `foreground_blocks.py` | modulated hyperbolic-tangent galactic foreground |
| `sgwb_blocks.py` | power-law, log-normal and phase-transition backgrounds |
| `synthetic.py` | synthetic observations drawn from block states (`simulate`) |
| `inst_noise_demo.py` | CD1-L conditioning (`load_observation`, `condition`) and the demo campaign |
| `foreground_demo.py` | NOISE + GALFOR bricks, instrument and foreground sampled together |
| `sgwb_demo.py` | NOISE brick + simulated background, instrument and template sampled together |

Enchilada is sourced from [rjrosati/enchilada's `cd1l-orbit-arrays` branch](https://github.com/rjrosati/enchilada/tree/cd1l-orbit-arrays), with the exact commit recorded in `uv.lock`.
`L1Data.from_cd1l` reads fractional-frequency XYZ data and the raw orbit and
six directed delay arrays. The local `column_delays` helper averages those
arrays onto WDM time columns; no separate HDF5 or orbit reader is needed.
The reader also loads optional clock and noise-estimate tables. It skips the
TTL tables unless asked (`load_ttl=True`): they take about 25 GB for a year of
data. Earlier enchilada commits load them unconditionally.

Memory for a year at ΔT = 12 h (8641 × 730 pixels), measured as the macOS
physical footprint: one Wheel campaign plateaus at about 7.3 GB, of which the
conditioned data are 1.4 GB and freed filter temporaries kept by malloc about
2.6 GB. Each covariance validation adds a ~3 GB transient. Run such cases one
at a time.

The demo retains the reference sampler's conditioning and layer calibration,
with a 0.3–8 mHz band and cropped edge columns.
The likelihood includes XYZ correlations within each pixel and approximates
pixels as independent. WDM coefficients are used directly; a flat one-sided
PSD gives variance S/2.

Set `coarseQ` on any block (default 1) to average consecutive retained time
columns. The Numba kernel weights each cell by its actual size. For Q>1,
covariance bases are averaged in each cell and that same constant covariance is
published on its constituent fine pixels. This approximates the unequal-arm time
variation; Q=1 retains the fine model. Equal-arm covariances are constant in
time, so any Q gives their exact likelihood. Numba compilation occurs on first
use.

Sampling cost: blocks keep only coarse, pixel-major bases, and the kernel loads
each pixel's bases once for all walkers. Problems of at least
`3 * TILE = 768` coarse pixels run in parallel over pixel tiles (set
`NUMBA_NUM_THREADS` to limit threads); results are identical to the serial
path. At coarse Q, each `sample` call is dominated by Enchilada's `Covariance`
validation (a Cholesky factorization on every grid pixel, including the
placeholders), not by the sampler.

The block publishes `BlockState` with `Covariance` and physical parameters,
keeping walkers and iteration counts in its own `sampler_state`. Use a separate
block instance for each campaign. It refreshes likelihoods for each conditional
residual and uses Wheel's RNG. Covariance uses unit-variance placeholders outside
the fitted band and cropped columns to meet Enchilada's full-grid requirements;
those pixels are excluded from this block's likelihood. The short campaign runs ten cycles.
Enchilada replaces the covariance rather than summing noise blocks, so the
foreground and stochastic backgrounds are sampled inside one block with the
instrument model (next section).

## Foreground and stochastic backgrounds

`StochasticNoiseBlock(instrument, components)` adds stochastic components to
any sampled instrument block and samples everything jointly: the instrument's
log ASDs, then each component's coordinates, with the stretch move. It reuses
the instrument block's analysis region, bases and prior (`coarseQ`, band and
cropping are set there) and publishes one combined covariance. A component is
a strain spectrum seen through the equal-arm TDI-2 response
`4 x² sin²x sin²2x` (`x = 2πfL/c`, nominal `L` = 2.5 Gm), folded through each
layer's WDM filter at 64 nodes, times a pattern over the six XYZ elements and
the WDM columns.

| Block | Components | Sampled coordinates after the instrument's |
|---|---|---|
| `GalacticForegroundBlock(instrument, foreground)` | `GalacticForeground` | log10 amp, log10 fk, alpha, log10 f_1, log10 f_2 |
| `SGWBBlock(instrument, PowerLawSGWB())` | power law | log10_A, alpha |
| `SGWBBlock(instrument, LogNormalSGWB())` | log-normal | log10_A, log10_fstar, log10_D |
| `SGWBBlock(instrument, PhaseTransitionSGWB())` | phase transition | rb, b, log10_Ap, log10_fp |
| `StochasticNoiseBlock(instrument, [...])` | any list | each component's, in order |

```python
from foreground_blocks import GalacticForeground, GalacticForegroundBlock
from sgwb_blocks import PowerLawSGWB, SGWBBlock

instrument = UnequalArmEqualNoiseBlock(data, delays, coarseQ=256)
block = GalacticForegroundBlock(instrument, GalacticForeground.from_file("modulation_unequal.dat"))
# block = SGWBBlock(instrument, PowerLawSGWB())
```

`GalacticForeground` is lisatools' `HyperbolicTangentGalacticForeground` with
its prior support, sampled in log10 except `alpha` and published as physical
`galfor_*` values. The modulation table (time, XX, YY, ZZ, XY, XZ, YZ) is
interpolated to column centres and averaged in each time cell; it must cover
the observation. `modulation_unequal.dat` is the CD1-L table from
`cd1l-validation`, on the bricks' clock. Without a table the foreground is
stationary and isotropic. With `UnequalArmEqualNoiseBlock` this is
`noise_sampler_minimal.py`'s foreground mode, except that coarse cells are
weighted by their size rather than by Welch-Satterthwaite degrees of freedom.

The SGWB templates are ported from `lisatools.stochastic` (a unit test pins
them to lisatools' values) and keep lisatools' parameters, published as
`powerlaw_*`, `lognormal_*` and `phase_transition_*`. They are isotropic and
stationary: `XY = XZ = YZ = -XX/2`. Priors and fiducials default to GLASS's
defaults; pass `priors=` or `fiducial=` to change them, and `fref=` to move
the power law's 25 Hz pivot. The phase-transition spectrum is undefined in
part of GLASS's prior box (below the peak when `rb⁴ (5 - b) > 4`); those
points are rejected, and initial walkers are redrawn until they are valid.

`sgwb_demo.py` draws its injection from the block's own covariance, so
recovering it tests the sampler and not the response model. Pick the template
at the bottom of the script.

## Synthetic data

`synthetic.simulate` builds a data set from block states: it adds every
state's template and draws Gaussian noise from every state's covariance.
Every sampled noise block gives its state at named physical parameters, the
same names and units it publishes to the ledger:

```python
from synthetic import blank_observation, simulate

reference = blank_observation(768, 2048, 5.0)   # or a conditioned brick, for its orbit
block = StochasticNoiseBlock(UnequalArmEqualNoiseBlock(reference, delays),
                             [GalacticForeground(), PowerLawSGWB()])
truth = block.state({"Soms_d": 1.3e-11, "Sa_a": 3.5e-15,
                     "galfor_amp": 1.6e-44, "galfor_fk": 4.3e-3, "galfor_alpha": 1.75,
                     "galfor_f_1": 3e-3, "galfor_f_2": 6e-4,
                     "powerlaw_log10_A": -8.45, "powerlaw_alpha": 2 / 3})
data = simulate(reference, [truth], block.active_mask, rng=np.random.default_rng(1))
```

`reference` supplies the WDM grid, epoch and orbit tables; its channel data
are replaced. Any `BlockState` on that grid can be passed, so a signal block's
published template is added the same way. Noise is drawn only on `mask`, the
pixels the covariance models (a noise block's `active_mask`), because
published covariances hold placeholders elsewhere; outside it the data
contain templates only. Each covariance in the list adds an independent draw,
so pass each noise source once. `transform(data, "time")` gives the
band-limited time series.

The draw uses independent WDM pixels with per-pixel XYZ correlations, which
are exactly the statistics the blocks' likelihood assumes. It therefore tests
that a pipeline recovers its inputs, including the effect of fitting with
`coarseQ > 1` data simulated at `coarseQ=1`. It cannot reveal errors in the
model's own approximations (correlations between pixels, the layer-calibrated
fold, the equal-arm stochastic response); that needs a time-domain simulation.

## Block naming

Sampled noise models are named `<arms>Arm<links>NoiseBlock`:

- **Arms**: `Unequal` uses the orbit's time-dependent directed delays, so the
  covariance varies over the orbit even when link levels are constant; `Equal`
  uses six equal, constant delays (`armlength_s`, default the observation's mean
  delay), so the covariance is constant in time. The CD1-L bricks have unequal
  arms, so equal-arm blocks are deliberately misspecified on them.
- **Links**: `Equal` shares one OMS and one acceleration level across all six
  links; `Unequal` gives each link its own levels.
- **Time**: constant link levels are the default and unnamed. Blocks whose
  levels change in time will add `Drifting`, e.g.
  `UnequalArmUnequalDriftingNoiseBlock` (planned).

| Block | Parameters | Moves |
|---|---|---|
| `UnequalArmEqualNoiseBlock` | 2 shared log ASDs | stretch |
| `UnequalArmUnequalNoiseBlock` | 12 per-link log ASDs | stretch + targeted link moves |
| `EqualArmEqualNoiseBlock` | 2 shared log ASDs | stretch |
| `EqualArmUnequalNoiseBlock` | 3 tied OMS pairs + 6 acceleration log ASDs (GLASS model) | stretch + targeted link moves (acceleration swaps) |
| `FixedNoiseBlock(model, amplitudes)` | none (any model above, held fixed) | none |
| `L1NoiseEstimateBlock` | none (L1Data noise estimates, as provided) | none |

`FixedNoiseBlock` registers as `'Fixed' + model.name` in the Wheel ledger.

Register one noise block per comparison Wheel (their covariances replace one
another):

```python
from noise_blocks import (
    EqualArmEqualNoiseBlock, EqualArmUnequalNoiseBlock, FixedNoiseBlock,
    L1NoiseEstimateBlock, UnequalArmEqualNoiseBlock, UnequalArmUnequalNoiseBlock,
)

# Choose one:
block = FixedNoiseBlock(UnequalArmEqualNoiseBlock(data, delays, coarseQ=32),
                        amplitudes=(1.5e-11, 3e-15))
# block = L1NoiseEstimateBlock(data, coarseQ=32)
# block = UnequalArmUnequalNoiseBlock(data, delays, nwalkers=32, steps_per_call=100, coarseQ=32)
# block = EqualArmEqualNoiseBlock(data, coarseQ=32)            # or armlength_s=2.5e9 / C_SI
# block = EqualArmUnequalNoiseBlock(data, nwalkers=32, coarseQ=32)

initial = block.initial_result(rng=np.random.default_rng(42))
wheel = Wheel(data, random_seed=42)
wheel.add(block, initial_block_state=initial, data_domain="wdm")
wheel.run(10)
```

`FixedNoiseBlock` holds a model's ASDs at the supplied values, in that model's
parameter order (2, 9 or 12), defaulting to its fiducial values. The current
L1Data does **not** store injection amplitudes, so these defaults are not read
from file metadata. Unequal-arm blocks use L1Data's directed delay arrays when
`delays` is omitted.

`L1NoiseEstimateBlock` uses the noise estimates provided with L1Data
(currently the CD1-L table, `data.noise_estimates_xyz`), divides the raw spectra
by `data.laser_frequency_hz**2`, and integrates the real symmetric XYZ spectral
matrix through the WDM filter (a flat one-sided PSD gives S/2). It interpolates
matrix entries linearly in log frequency and linearly in time at WDM column
centres, holding the nearest time estimate outside the tabulated range. It
rejects missing spectra, insufficient frequency coverage, and invalid spectral
matrices. Imaginary CSD and correlations between WDM pixels are omitted, as in
the existing real per-pixel likelihood. `coarseQ` averages consecutive retained
columns. Both fixed blocks publish the same covariance on every call without
sampling or consuming RNG draws.

`UnequalArmUnequalNoiseBlock` samples 12 constant log ASDs: `Soms_d_{link}` followed by
`Sa_a_{link}`, each in link order `(12, 23, 31, 13, 32, 21)` (received at i from j).
The acceleration label denotes the test mass entering eta_ij directly and the
reverse eta with a delay. It projects independent link spectra through the
TDI-2 transfer functions using SEGWO's `C = M diag(S) M†` construction, retaining
the same spectral shapes, band, time cropping, WDM calibration and coarse
likelihood as `UnequalArmEqualNoiseBlock`; its covariance varies with the arm
lengths exactly as that block's does. It defaults to 32 walkers and requires at
least 24. Individual link amplitudes can be weakly constrained or degenerate
with each other; the two-parameter mixing estimates do not describe this model.
All blocks publish unit-variance placeholders outside the comparison region;
use their `active_mask` when comparing likelihoods.

`UnequalArmUnequalNoiseBlock` uses a 50/50 mixture of the stretch move and a small
GLASS-inspired move. The latter chooses equally between a single log-ASD step,
a reverse-link power exchange (OMS or acceleration), and an acceleration cycle
in direction `(+,+,+,-,-,-)` in our link order. Step scales span 0.001–1 times
nominal powers (or log-ASD units for local steps). Power moves include the
Hastings correction for log-ASD coordinates; negative powers are rejected.
Use `targeted_proposals=False` for the stretch-only comparison.

An optional soft OMS constraint penalizes extreme reverse-link splits while
keeping all six OMS amplitudes independent:

```python
block = UnequalArmUnequalNoiseBlock(
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

`EqualArmUnequalNoiseBlock` reproduces GLASS's equal-arm instrument model
(`glass/noise/src/glass_noise_model.c`, `generate_instrument_noise_model`,
three channels) with the TDI-2 factor `4 sin²(2x)`, `x = 2πfL/c`; our
normalization is 4× GLASS's, and GLASS's `sacc_ij` is our `Sa_a_ji`. A unit test
checks a direct transcription of the C expression. At equal arms the real XYZ
covariance sees reverse-link OMS powers only through their sum, so, as in GLASS,
each pair is tied: `Soms_d_12_21` sets both links 12 and 21. Below
`f ≪ c/(2πL)` the acceleration powers are likewise constrained mostly through
reverse-pair sums; their splits are weakly identified, which the
acceleration reverse-pair swaps and the GLASS acceleration cycle (`ACC_CYCLE`,
GLASS's rank-one correlation-matrix move) are meant to explore. We sample log
ASDs with log-uniform priors rather than GLASS's power steps and uniform priors.
