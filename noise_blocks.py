"""WDM XYZ instrument-noise blocks; register one per comparison Wheel.

Naming: <arms>Arm<links>NoiseBlock. Unequal arms follow the orbit's
time-dependent delays, so their covariance varies in time even with constant
link levels; equal arms use six equal, constant delays. Equal noise shares one
OMS and one acceleration level across links; unequal noise has per-link levels.
Constant levels are the default; time-dependent levels will add Drifting.

    SampledNoiseBlock               Eryn ensemble, coarse likelihood, publishing
    ├── EqualNoiseBlock             2 shared levels, stretch move
    │   ├── UnequalArmEqualNoiseBlock
    │   └── EqualArmEqualNoiseBlock
    └── UnequalNoiseBlock           per-link levels, stretch + LinkNoiseMove
        ├── UnequalArmUnequalNoiseBlock   12 levels, optional OMS ratio prior
        └── EqualArmUnequalNoiseBlock     9 levels, reverse OMS pairs tied (GLASS)

    PublishedCovarianceBlock        a fixed covariance, no sampling
    ├── FixedNoiseBlock             any sampled model at fixed amplitudes
    └── L1NoiseEstimateBlock        L1Data's XYZ noise estimates

Wheel keeps one covariance owner, so noise that adds to the instrument's is
sampled and published with it:

    StochasticNoiseBlock            an instrument model + StochasticComponents
    ├── GalacticForegroundBlock     foreground_blocks.py
    └── SGWBBlock                   sgwb_blocks.py
"""

import numpy as np
import wdm
from enchilada import Block, BlockState
from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer, UniformDistribution
from eryn.state import State

from noise_model import (
    BAND_HZ, CHANNELS, CROSS_WEIGHTS, FIDUCIAL, LINKS, PRIORS, UI, UJ, AnalysisRegion,
    check_xyz_wdm, column_delays, equal_arm_delays, instrument_bases, layer_nodes,
    noise_loglike, stochastic_response, unit_bases, wdm_filter_weights,
)
from noise_proposals import REVERSE_PAIRS, LinkNoiseMove, OMSPairPrior


REJECTED = -1e300  # log likelihood of an invalid covariance


# --- Sampled models --------------------------------------------------------------


class SampledNoiseBlock(Block):
    """Base for sampled instrument-noise models; register a subclass.

    Subclasses define the parameterization (``name``, ``parameter_names``,
    ``fiducial``, ``priors`` as log-ASD bounds, and ``unit_model`` returning
    unit-level covariances per parameter) and pass per-column delays.

    Construction folds the bases once and keeps only their coarse, pixel-major
    form. Each ``sample`` call reduces the new residual to coarse cross products
    and advances the block-owned Eryn ensemble; the published covariance uses
    walker 0 and is piecewise constant in time cells, exactly as fitted.
    ``chain`` holds the latest call's coordinates, (steps, walkers, parameters).
    """

    parameter_names: tuple
    fiducial: np.ndarray
    priors: np.ndarray

    def __init__(self, data, column_ltts, *, nwalkers=16, steps_per_call=100,
                 time_slices_crop=3, coarseQ=1, band_hz=BAND_HZ):
        check_xyz_wdm(data)
        ndim = len(self.parameter_names)
        if nwalkers < 2 * ndim:
            raise ValueError(f"The {ndim}-parameter stretch move requires at least "
                             f"{2 * ndim} walkers")
        self.grid, self.fs, self.t0 = data.wdm_grid, data.fs, data.t0
        self.nwalkers = nwalkers
        self.steps_per_call = steps_per_call
        self.sampler_state = self.chain = None
        self.region = AnalysisRegion.from_data(data, time_slices_crop=time_slices_crop,
                                               coarseQ=coarseQ, band_hz=band_hz)
        self.active_mask = self.region.active_mask
        fine = instrument_bases(self.region.layers, self.grid, data.dt, column_ltts,
                                self.unit_model)[:, UI, UJ]
        coarse = self.region.coarsen(fine)  # (components, 6, layers, cells)
        self.bases = np.ascontiguousarray(coarse.reshape(ndim, 6, -1).transpose(2, 0, 1))

    @staticmethod
    def unit_model(f, d):
        raise NotImplementedError

    @property
    def center(self):
        """Sampling coordinates the initial walkers scatter around."""
        return np.log(self.fiducial)

    def levels(self, rows):
        """Power multiplying each basis for every coordinate row."""
        return np.exp(2 * np.atleast_2d(rows))

    def parameters(self, row):
        """Published physical values of one coordinate row."""
        return dict(zip(self.parameter_names, map(float, np.exp(row))))

    def coordinates(self, values):
        """Coordinate row of physical values in ``parameter_names`` order."""
        return np.log(values)

    def state(self, parameters):
        """The BlockState this model publishes at the named physical parameters.

        ``parameters`` maps every name in ``parameter_names`` to its published
        value, as in the ledger; synthetic.simulate draws data from the result.
        """
        if set(parameters) != set(self.parameter_names):
            raise ValueError(f"{self.name} requires exactly the parameters "
                             f"{', '.join(self.parameter_names)}")
        values = np.array([parameters[name] for name in self.parameter_names], dtype=float)
        row = self.coordinates(values)
        if not np.isfinite(row).all():
            raise ValueError("Parameters must give finite sampling coordinates")
        return BlockState(covariance=self.covariance(row), parameters=self.parameters(row))

    def log_likelihood(self, data):
        """Return a vectorized log likelihood of coordinate rows for this residual."""
        if (data.wdm_grid, data.fs, data.t0, data.channel_names) != (
                self.grid, self.fs, self.t0, CHANNELS):
            raise ValueError("Residual grid, epoch and channels must match the block")
        w = np.stack([data.channel_data[ch][self.region.layers] for ch in CHANNELS])
        products = w[UI] * w[UJ] * CROSS_WEIGHTS[:, None, None]
        X = np.ascontiguousarray(self.region.coarsen(products).reshape(6, -1).T)
        weights = self.region.weights

        def loglike(rows):
            scores = noise_loglike(self.levels(rows), self.bases, X, weights)
            # Eryn turns a rejected -inf proposal into a NaN walker; keep scores finite.
            return np.where(np.isfinite(scores), scores, REJECTED)

        return loglike

    def covariance(self, logs):
        packed = np.tensordot(self.bases, np.exp(2 * np.asarray(logs)), axes=(1, 0))
        coarse = packed.T.reshape(6, len(self.region.layers), self.region.num_cells)
        return self.region.covariance(coarse)

    def _result(self, coords):
        row = coords[0, 0, 0]
        return BlockState(covariance=self.covariance(row), parameters=self.parameters(row))

    def initial_result(self, *, rng):
        center = self.center
        coords = center + rng.normal(0, 0.05, size=(1, self.nwalkers, 1, len(center)))
        coords[0, 0, 0] = center
        self.sampler_state = {"coords": coords.copy(), "iterations": 0}
        return self._result(coords)

    def make_prior(self):
        return ProbDistContainer({i: UniformDistribution(*b) for i, b in enumerate(self.priors)})

    def make_moves(self):
        return StretchMove(randomize_split=False)  # keep Wheel's RNG ownership

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        if self.sampler_state is None:
            raise ValueError("Call initial_result before sampling")
        sampler = EnsembleSampler(
            self.nwalkers,
            [len(self.parameter_names)],
            self.log_likelihood(conditional_residual),
            {"noise": self.make_prior()},
            branch_names=["noise"],
            nleaves_max=[1],
            nleaves_min=[1],
            vectorize=True,
            moves=self.make_moves(),
            tempering_kwargs={"betas": np.array([1.0]), "adaptive": False, "permute": False},
        )
        sampler.random_state = np.random.RandomState(int(rng.integers(2**32))).get_state()
        # Fresh State forces likelihood evaluation against this call's residual.
        state = State({"noise": self.sampler_state["coords"].copy()})
        final = sampler.run_mcmc(state, self.steps_per_call, progress=False)
        self.chain = sampler.get_chain()["noise"][:, 0, :, 0, :]
        coords = final.branches_coords["noise"]
        self.sampler_state = {
            "coords": coords.copy(),
            "iterations": self.sampler_state["iterations"] + self.steps_per_call,
        }
        return self._result(coords)


class EqualNoiseBlock(SampledNoiseBlock):
    """Shared OMS and acceleration levels on all six links; stretch move."""

    parameter_names = ("Soms_d", "Sa_a")
    fiducial = np.array(FIDUCIAL)
    priors = PRIORS
    unit_model = staticmethod(unit_bases)


class UnequalNoiseBlock(SampledNoiseBlock):
    """Per-link levels: OMS levels, then six test masses in LINKS order.

    Moves mix the stretch move 50/50 with LinkNoiseMove unless
    ``targeted_proposals=False``. The prior is a box on log ASDs, optionally
    with a soft reverse-link OMS ratio term (``oms_pair_sigma``).
    """

    oms_pair_sigma = None

    def __init__(self, data, column_ltts, *, nwalkers=32, targeted_proposals=True, **kwargs):
        self.targeted_proposals = targeted_proposals
        self.link_prior = OMSPairPrior(self.priors, self.oms_pair_sigma)
        super().__init__(data, column_ltts, nwalkers=nwalkers, **kwargs)

    def make_prior(self):
        return self.link_prior

    def make_moves(self):
        stretch = super().make_moves()
        if not self.targeted_proposals:
            return stretch
        return [(stretch, 0.5), (LinkNoiseMove(self.fiducial), 0.5)]


class UnequalArmEqualNoiseBlock(EqualNoiseBlock):
    """Two shared levels with the orbit's per-column directed delays.

    Translated from the unequal-arm model used in our CD1-L analyses.
    ``column_ltts`` defaults to ``column_delays(data)``.
    """

    name = "UnequalArmEqualNoiseBlock"

    def __init__(self, data, column_ltts=None, **kwargs):
        super().__init__(data, column_delays(data) if column_ltts is None else column_ltts,
                         **kwargs)


class EqualArmEqualNoiseBlock(EqualNoiseBlock):
    """Two shared levels with six equal, constant arms; constant in time."""

    name = "EqualArmEqualNoiseBlock"

    def __init__(self, data, armlength_s=None, **kwargs):
        delays = equal_arm_delays(data, armlength_s)
        self.armlength_s = float(delays[0, 0])
        super().__init__(data, delays, **kwargs)


class UnequalArmUnequalNoiseBlock(UnequalNoiseBlock):
    """Six OMS and six test-mass levels with the orbit's per-column delays.

    Uses C = M diag(S_link) M† (SEGWO cov.project_covariance). The test-mass
    level for link ij labels the mass in eta_ij's direct term.
    ``oms_log_power_ratio_sigma`` adds -0.5 (ln(P_ij/P_ji) / sigma)² per
    reverse OMS pair to the log prior.
    """

    name = "UnequalArmUnequalNoiseBlock"
    parameter_names = tuple(f"{kind}_{link}" for kind in ("Soms_d", "Sa_a") for link in LINKS)
    fiducial = np.repeat(FIDUCIAL, 6)
    priors = np.repeat(PRIORS, 6, axis=0)

    @staticmethod
    def unit_model(f, d):
        return unit_bases(f, d, per_link=True)

    def __init__(self, data, column_ltts=None, *, oms_log_power_ratio_sigma=None, **kwargs):
        self.oms_pair_sigma = oms_log_power_ratio_sigma
        super().__init__(data, column_delays(data) if column_ltts is None else column_ltts,
                         **kwargs)


class EqualArmUnequalNoiseBlock(UnequalNoiseBlock):
    """GLASS's equal-arm per-link instrument model, in TDI-2.

    At equal arms the real XYZ covariance depends on reverse-link OMS powers
    only through their sum, so each pair is tied, as GLASS does: three OMS
    levels (``Soms_d_12_21`` sets both links 12 and 21), then six test-mass
    levels in link order. This reproduces glass_noise_model.c
    generate_instrument_noise_model (Nchannel=3) times the TDI-2 factor
    4 sin²(2x) and an overall 4 (GLASS's normalization); GLASS's ``sacc_ij``
    is our ``Sa_a_ji``. We sample log ASDs with log-uniform priors, where
    GLASS steps in power with uniform priors. LinkNoiseMove swaps reverse-pair
    power for acceleration only.
    """

    name = "EqualArmUnequalNoiseBlock"
    parameter_names = (tuple(f"Soms_d_{LINKS[i]}_{LINKS[j]}" for i, j in REVERSE_PAIRS)
                       + tuple(f"Sa_a_{link}" for link in LINKS))
    fiducial = np.repeat(FIDUCIAL, (3, 6))
    priors = np.repeat(PRIORS, (3, 6), axis=0)

    @staticmethod
    def unit_model(f, d):
        bases = unit_bases(f, d, per_link=True)
        return np.concatenate([np.stack([bases[i] + bases[j] for i, j in REVERSE_PAIRS]),
                               bases[6:]])

    def __init__(self, data, armlength_s=None, **kwargs):
        delays = equal_arm_delays(data, armlength_s)
        self.armlength_s = float(delays[0, 0])
        super().__init__(data, delays, **kwargs)


# --- Instrument plus stochastic components -------------------------------------------

ISOTROPIC = np.array([1.0, 1.0, 1.0, -0.5, -0.5, -0.5])  # XX YY ZZ XY XZ YZ


class StochasticComponent:
    """A strain spectrum seen through the equal-arm TDI-2 response.

    Its covariance is ``pattern × fold(psd × stochastic_response)``: one
    variance per layer, spread over the packed XYZ elements and WDM columns by
    ``pattern``. Subclasses set ``parameter_names`` (published; unique within
    a block), ``priors`` (bounds) and ``fiducial`` (centre of the initial
    walkers), both in sampling coordinates, and implement ``psd``. Pass
    ``priors`` or ``fiducial`` to replace the defaults.
    """

    parameter_names: tuple
    priors: np.ndarray
    fiducial: np.ndarray

    def __init__(self, *, priors=None, fiducial=None):
        n = len(self.parameter_names)
        self.priors = np.array(type(self).priors if priors is None else priors, dtype=float)
        self.fiducial = np.array(type(self).fiducial if fiducial is None else fiducial,
                                 dtype=float)
        if self.priors.shape != (n, 2) or self.fiducial.shape != (n,):
            raise ValueError(f"{type(self).__name__} requires {n} prior bounds and fiducials")
        if not np.all((self.priors[:, 0] < self.fiducial) & (self.fiducial < self.priors[:, 1])):
            raise ValueError("Fiducial coordinates must lie inside the prior bounds")

    def psd(self, f, coords):
        """One-sided strain PSD: ``f`` (layers, nodes), ``coords`` (rows, n) -> (rows, layers, nodes)."""
        raise NotImplementedError

    def pattern(self, times_gps):
        """Packed XYZ pattern at WDM column times, (6, columns); isotropic by default."""
        return np.repeat(ISOTROPIC[:, None], len(times_gps), axis=1)

    def physical(self, coords):
        """Published values for one coordinate row, in ``parameter_names`` order."""
        return coords

    def coordinates(self, values):
        """Inverse of ``physical``."""
        return values


class JointPrior:
    """The instrument's prior on the leading coordinates and a box on the rest."""

    def __init__(self, instrument_prior, split, bounds):
        self.instrument_prior, self.split, self.bounds = instrument_prior, split, bounds
        self.key_order = list(range(split + len(bounds)))  # Eryn backend column metadata

    def logpdf(self, coords):
        shape = np.shape(coords)[:-1]
        coords = np.reshape(coords, (-1, len(self.key_order)))  # Eryn priors take 2-D rows
        rest = coords[:, self.split:]
        inside = np.all((rest >= self.bounds[:, 0]) & (rest <= self.bounds[:, 1]), axis=-1)
        instrument = self.instrument_prior.logpdf(coords[:, :self.split])
        return np.where(inside, instrument, -np.inf).reshape(shape)


class StochasticNoiseBlock(SampledNoiseBlock):
    """An instrument model and stochastic components, sampled and published together.

    ``instrument`` is any sampled instrument block: its analysis region, folded
    bases, parameters and prior are reused, its walkers are not.
    ``components`` are StochasticComponents; pass several to fit them jointly.
    Coordinates are the instrument's log ASDs, then each component's sampling
    coordinates, moved by the stretch move only (the instrument's link moves
    are not used). Components use the nominal ``ARMLENGTH_S`` response whatever
    the instrument's arms. Each spectrum is folded through a layer's filter at
    ``nodes`` frequencies; a few suffice when layers are narrow.
    """

    name = "StochasticNoiseBlock"

    def __init__(self, instrument, components, *, nwalkers=32, steps_per_call=100, nodes=64):
        # Adopts the instrument's folded bases rather than refolding them.
        self.instrument, self.components = instrument, tuple(components)
        if not self.components:
            raise ValueError("StochasticNoiseBlock requires at least one component")
        self.grid, self.fs, self.t0 = instrument.grid, instrument.fs, instrument.t0
        self.region, self.active_mask = instrument.region, instrument.active_mask
        self.parameter_names = instrument.parameter_names + tuple(
            name for c in self.components for name in c.parameter_names)
        ndim = len(self.parameter_names)
        if len(set(self.parameter_names)) != ndim:
            raise ValueError("Parameter names must be unique within a block")
        if nwalkers < 2 * ndim:
            raise ValueError(f"The {ndim}-parameter stretch move requires at least "
                             f"{2 * ndim} walkers")
        self.nwalkers, self.steps_per_call = nwalkers, steps_per_call
        self.sampler_state = self.chain = None
        self.split = len(instrument.parameter_names)
        self.bounds = np.vstack([c.priors for c in self.components])
        nf, nt, dt = (self.grid.num_frequency_divisions, self.grid.num_time_divisions,
                      1 / self.fs)
        f, weights = layer_nodes(self.region.layers, self.grid, dt, nodes)
        self.node_frequencies, self.node_weights = f, weights * stochastic_response(f)
        times = self.t0 + np.arange(nt) * nf * dt
        patterns = np.stack([self.region.coarsen(c.pattern(times)) for c in self.components])
        # (components, 6, cells) -> layer-major pixels, beside the instrument's bases
        tiled = np.tile(patterns.transpose(2, 0, 1), (len(self.region.layers), 1, 1))
        self.bases = np.ascontiguousarray(np.concatenate([instrument.bases, tiled], axis=1))

    @property
    def center(self):
        return np.concatenate([self.instrument.center, *(c.fiducial for c in self.components)])

    def _split(self, rows):
        """Each component with its columns of ``rows`` (..., ndim)."""
        stops = np.cumsum([len(c.parameter_names) for c in self.components])
        return zip(self.components, np.split(rows[..., self.split:], stops[:-1], axis=-1))

    def levels(self, rows):
        """(rows, layers, components): instrument powers, then each folded spectrum."""
        rows = np.atleast_2d(rows)
        out = np.empty((len(rows), len(self.region.layers), self.bases.shape[1]))
        out[:, :, :self.split] = np.exp(2 * rows[:, None, :self.split])
        with np.errstate(all="ignore"):  # invalid corners of a prior are rejected
            for i, (component, coords) in enumerate(self._split(rows)):
                spectrum = component.psd(self.node_frequencies, coords)
                out[:, :, self.split + i] = np.sum(spectrum * self.node_weights, axis=-1)
        return out

    def initial_result(self, *, rng):
        """Scatter walkers about the centre, redrawing any that fall outside
        the prior or where a template is undefined."""
        center, prior = self.center, self.make_prior()

        def invalid(rows):
            levels = self.levels(rows)
            return ~(np.isfinite(prior.logpdf(rows)) & np.isfinite(levels).all(axis=(1, 2))
                     & (levels >= 0).all(axis=(1, 2)))

        if invalid(center[None])[0]:
            raise ValueError("Fiducial coordinates must lie inside the prior and give "
                             "finite spectra")
        walkers = np.tile(center, (self.nwalkers, 1))
        redraw = np.arange(1, self.nwalkers)  # walker 0 stays at the centre
        for _ in range(100):
            walkers[redraw] = center + rng.normal(0, 0.05, size=(len(redraw), len(center)))
            redraw = redraw[invalid(walkers[redraw])]
            if not len(redraw):
                break
        else:
            raise ValueError("Could not place walkers with finite spectra near the fiducial")
        coords = walkers[None, :, None, :]
        self.sampler_state = {"coords": coords.copy(), "iterations": 0}
        return self._result(coords)

    def parameters(self, row):
        values = [np.exp(row[:self.split])]
        values += [component.physical(coords) for component, coords in self._split(row)]
        return dict(zip(self.parameter_names, map(float, np.concatenate(values))))

    def coordinates(self, values):
        return np.concatenate([np.log(values[:self.split]), *(
            component.coordinates(physical) for component, physical in self._split(values))])

    def covariance(self, row):
        bases = self.bases.reshape(len(self.region.layers), self.region.num_cells, -1, 6)
        return self.region.covariance(np.einsum("lqck,lc->klq", bases, self.levels(row)[0]))

    def make_prior(self):
        return JointPrior(self.instrument.make_prior(), self.split, self.bounds)


# --- Fixed covariances -------------------------------------------------------------


class PublishedCovarianceBlock(Block):
    """Publish one precomputed state on every call; consume no random numbers."""

    def _check_data(self, data):
        if (data.data_domain, data.wdm_grid, data.fs, data.t0, data.channel_names,
                data.tdi_generation, data.physical_observable) != self._conventions:
            raise ValueError("Residual grid, epoch and conventions must match the block")

    def initial_result(self, *, rng=None):
        return self._state

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        self._check_data(conditional_residual)
        return self._state


class FixedNoiseBlock(PublishedCovarianceBlock):
    """Hold a sampled model's ASDs fixed, e.g. at injection values.

    ``amplitudes`` are ASDs in the model's parameter order, defaulting to its
    fiducial values. L1Data carries no injection amplitudes, so the defaults
    are the nominal 15 pm/sqrt(Hz) and 3 fm/s²/sqrt(Hz), not file metadata.
    The ledger name is ``'Fixed' + model.name``.
    """

    def __init__(self, model, amplitudes=None):
        amplitudes = np.asarray(model.fiducial if amplitudes is None else amplitudes, dtype=float)
        if amplitudes.shape != (len(model.parameter_names),) or not np.all(
                np.isfinite(amplitudes) & (amplitudes > 0)):
            raise ValueError(f"amplitudes must contain {len(model.parameter_names)} "
                             f"positive finite ASDs for {model.name}")
        self.name = f"Fixed{model.name}"
        self.model = model
        self.active_mask = model.active_mask
        # The model already enforced WDM XYZ TDI-2 fractional-frequency data.
        self._conventions = ("wdm", model.grid, model.fs, model.t0, CHANNELS,
                             "2.0", "fractional_frequency")
        self._state = BlockState(covariance=model.covariance(np.log(amplitudes)),
                                 parameters=dict(zip(model.parameter_names, map(float, amplitudes))))


class L1NoiseEstimateBlock(PublishedCovarianceBlock):
    """Fold L1Data's one-sided XYZ Hz²/Hz noise estimates into real WDM covariance.

    The estimates are used as provided (currently the CD1-L noise_estimates
    table), on the same analysis region as the sampled blocks. Matrix entries
    are interpolated linearly in log frequency and in reception time.
    Frequency extrapolation is rejected; time extrapolation holds the nearest
    estimate. Imaginary CSDs describe correlations this real model omits.
    """

    name = "L1NoiseEstimateBlock"

    def __init__(self, data, *, time_slices_crop=3, coarseQ=1):
        check_xyz_wdm(data)
        if data.noise_estimates_xyz is None or data.laser_frequency_hz is None:
            raise ValueError("L1 XYZ noise estimates and laser_frequency_hz are required")
        region = AnalysisRegion.from_data(data, time_slices_crop=time_slices_crop,
                                          coarseQ=coarseQ)
        spectra = self._real_spectra(data)  # (times, frequencies, 3, 3)
        nf, nt = region.grid.num_frequency_divisions, region.grid.num_time_divisions
        offsets, weights = wdm_filter_weights(nf, nt, data.dt)
        table_f = data.noise_estimate_frequencies_hz
        index = np.arange(len(table_f))
        column_t = data.t0 + np.arange(nt) * nf * data.dt
        packed = np.empty((6, len(region.layers), nt))
        for n, f in enumerate(wdm.band_frequencies(nf, data.dt)[region.layers]):
            query = f + offsets
            if query.min() < table_f[0] or query.max() > table_f[-1]:
                raise ValueError("Noise-estimate frequencies must cover the WDM filter support")
            # Linear interpolation in log f followed by the filter average is
            # one weighted sum over table frequencies.
            position = np.interp(np.log(query), np.log(table_f), index)
            low = np.minimum(position.astype(int), len(table_f) - 2)
            upper = position - low
            kernel = (np.bincount(low, weights * (1 - upper), len(table_f))
                      + np.bincount(low + 1, weights * upper, len(table_f)))
            folded = np.einsum("f,tfij->tij", kernel, spectra)[:, UI, UJ]
            packed[:, n] = [np.interp(column_t, data.noise_estimate_sample_times_gps, entry)
                            for entry in folded.T]
        self.active_mask = region.active_mask
        self._conventions = (data.data_domain, data.wdm_grid, data.fs, data.t0,
                             data.channel_names, data.tdi_generation, data.physical_observable)
        covariance = region.covariance(region.coarsen(packed))
        covariance.check_compatible(data)
        self._state = BlockState(covariance=covariance,
                                 parameters={"source": "L1Data.noise_estimates_xyz"})

    @staticmethod
    def _real_spectra(data):
        """Fractional-frequency spectra, checked Hermitian and positive definite."""
        spectra = data.noise_estimates_xyz / data.laser_frequency_hz**2
        scale = np.max(np.abs(spectra))
        if not np.isfinite(spectra).all() or not np.allclose(
                spectra, spectra.swapaxes(-1, -2).conj(), rtol=1e-10, atol=scale * 1e-12):
            raise ValueError("XYZ noise estimates must be finite and Hermitian")
        # Taking the real symmetric part preserves positive definiteness.
        spectra = (spectra.real + spectra.real.swapaxes(-1, -2)) / 2
        if np.any(np.linalg.eigvalsh(spectra) <= 0):
            raise ValueError("XYZ noise estimates must be positive definite")
        return spectra
