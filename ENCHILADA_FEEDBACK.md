# Enchilada feedback from the noise-block proof of concept

Problems we hit while building the instrument-noise blocks in this repo.
Apart from the TTL loading in §7, nothing here has been fixed upstream; the
noise blocks work around them or accept the cost. Line references are to
`rjrosati/enchilada@cd1l-orbit-arrays` at `d7d5f6f`; `uv.lock` now pins
`0000cf5`, which adds only the TTL change.

Timings: synthetic WDM data on the demo grid (768 × 24576 pixels, 18.9 M per
channel), `UnequalArmEqualNoiseBlock` with `coarseQ=1024`, 4 walkers and 100
Eryn steps per call, profiled with cProfile.

| Per Wheel cycle | Time |
|---|---|
| Eryn sampling, 100 steps (including our likelihood) | 0.1 s |
| Covariance positive-definiteness checks (4 per cycle) | ~21–25 s |
| Deep copies, `check_compatible`, building our covariance | ~2 s |
| **Total** | **~27 s** |

The sampler is about 0.4% of a cycle; Enchilada's covariance handling is the
rest.

## 1. Covariance validation is expensive and cannot be turned off

`Covariance.__post_init__` checks positive definiteness at every grid pixel
(`covariance.py:225` `_check_positive_diagonal_form`, `:243`
`_check_block_positive`). At the demo grid one check costs about 5–6 s:

- ~2.6 s assembling an `(18.9 M, 3, 3)` array in Python/NumPy;
- ~2.3 s in a batched `np.linalg.cholesky`.

Most of those pixels are unit-variance placeholders (see §3) that cannot fail.
There is no way for a trusted caller to skip the check.

It is also the memory peak. For one year at ΔT = 12 h (8641 × 730 pixels), a
`Covariance` holds 0.3 GB, but constructing it allocates a further 2.95 GB of
transients (traced), and the Wheel repeats the check on every cycle. A whole
noise campaign on that grid plateaus at a 7.3 GB physical footprint, which is
what limits how many cases fit on one machine.

Suggestions:
- a keyword-only `validate: InitVar[bool] = True` on `Covariance` that skips
  only the positive-definiteness check (structural checks stay); as an
  `InitVar` it does not affect `copy`, `deepcopy` or pickling;
- `Wheel(..., validate_covariance=True)`, applied in `run()` only, so `add()`
  and `initial_covariance` still validate once at setup;
- a cheaper check regardless of the flag: skip pixels whose channel matrix is
  diagonal, and use a closed-form 3 × 3 test instead of a batched Cholesky.

## 2. The same covariance is validated four times per cycle

For a block registered with `data_domain="wdm"` on WDM data, each cycle runs
the full check four times:

1. `wheel.py:344`: `run()` passes the ledger state through `transform()` to the
   block's domain. The domain and grid are unchanged, but the `Covariance`
   branch of `transform()` (`translation.py:140`) calls `Covariance.copy`,
   which revalidates.
2. The block builds its new `Covariance`.
3. `wheel.py:298`: `_canonical_state` transforms the returned state back,
   again unchanged, again revalidated.
4. `wheel.py:466`: `_adopt` calls `Covariance.copy` on the returned covariance.

`_block_inputs` already skips the transform when the block's domain matches
the observation grid, with a comment explaining why; `run()` and
`_canonical_state` do not. Applying the same skip removes checks 1 and 3 and
the deep copies that come with them, with no loss of safety.

## 3. Covariances must cover the full grid

A block that fits part of the grid (here 0.3–8 mHz, with three time columns
cropped at each end) still has to publish a covariance on every pixel. We fill
the rest with unit-variance placeholders. Consequences:

- validation and copying scale with the whole grid, not the fitted region
  (696 coarse cells against 18.9 M pixels per channel in the demo);
- likelihood comparisons between blocks need each block's `active_mask`,
  because the placeholders are not a real noise model;
- downstream code cannot tell placeholder pixels from modelled ones.

A covariance that carries its own support mask (or a "not modelled" value)
would avoid all three.

## 4. Whole-covariance deep copies on every read and write

One pair array at the demo grid is 151 MB; a three-channel covariance holds six,
about 0.9 GB. It is deep-copied:

- on every `wheel.ledger[name]` read (`ledger.py:25`), including the demo's
  per-cycle `wheel.ledger[block.name].parameters` print, which only needs two
  floats;
- in `_adopt` (`wheel.py:435`) before validation;
- in `transform()` for a `BlockState` (`translation.py`, `deepcopy(value)`).

Together with `check_compatible` and building our own covariance this is about
2 s per cycle, which becomes the floor once validation is skipped. The stored
arrays are already read-only, so the ledger could hand out shared read-only
views instead of copies, and a parameters-only accessor would serve callers
that do not need the arrays.

## 5. One covariance owner per Wheel; no composition

A block's covariance replaces the current one rather than adding to it
(`wheel.py:35`, `block.py:29`); a second publisher takes ownership with a
`NoiseOverwrittenWarning` (`wheel.py:503`). Our noise model now has three
physically separate terms: instrument noise, the galactic confusion foreground
and a stochastic background. We want them as three blocks, each sampling its
own parameters and contributing its own term to one total covariance. Today
they have to be merged: `StochasticNoiseBlock` wraps an instrument model and
samples every component jointly in one block. Costs:

- each combination of components is a different block, with its own name,
  parameter vector and walkers, so a component cannot be added to or removed
  from a Wheel, updated at its own cadence, or given its own sampler (the
  instrument's targeted link moves are not used in the joint block);
- the foreground and SGWB demos each carry their own copy of the instrument
  model, and a run with all three needs a block holding all three;
- comparing noise models means one Wheel per model.

What composition would need:

- Wheel sums the covariance terms of every publishing block, and passes each
  block the sum of the *other* blocks' terms, as `conditional_residual` does
  for signals;
- positive definiteness is checked on the sum, not on each term. A stochastic
  term alone is singular: the isotropic XYZ pattern (1 on the diagonal, -1/2
  off it) has a null direction, and the foreground's modulation table is
  rank 2 to one part in 10⁹. Such a term fails today's per-`Covariance` check
  (§1) on its own;
- a term covers only the pixels it models (§3).

## 6. No covariance translation between domains

`transform()` raises `NotImplementedError` for a covariance whose target
domain or grid differs from its reference (`translation.py`, pointing at design
spec section 6). Noise blocks must therefore run on the observation's own
domain and grid; a block cannot, for example, fit on a coarser WDM grid and
have its covariance mapped back.

## 7. Smaller points

- `L1Data.from_cd1l` loaded the TTL tables unconditionally: 24 tables at ten
  samples per data sample, 25 GB for a one-year prefix, enough to exhaust a
  19 GB machine. Fixed in `0000cf5` on `rjrosati/enchilada@cd1l-orbit-arrays`:
  they now load only with `from_cd1l(..., load_ttl=True)`.
- `L1Data.from_cd1l` rejects the GALFOR brick: its `noise_estimates/XYZ` table
  is a one-point placeholder, shape (1, 1, 3, 3), and the reader requires at
  least two frequencies (`data.py:581`). The optional tables cannot be
  skipped, so `foreground_demo.py` reads that brick's XYZ with h5py. There is
  also no reader support for adding a signal brick to a noise brick.
- `L1Data` in the WDM domain requires inactive coefficients to be exactly zero;
  synthetic test data has to be multiplied by `grid.active_mask` first. This is
  reasonable, but the error message could name the fix.
