"""Alternative WDM XYZ noise blocks for separate comparison campaigns."""

import numpy as np
import wdm
from enchilada import Block, BlockState

from noise_proposals import LinkNoiseMove, OMSPairPrior

from inst_noise_demo import (
    CHANNELS, FIDUCIAL, LINKS, PRIORS, UI, UJ, InstrumentNoiseBlock,
    packed_covariance, unit_bases,
)


class LinkNoiseBlock(InstrumentNoiseBlock):
    """Fit six OMS and six test-mass ASDs, stationary and separately variable.

    Uses C = M diag(S_link) M† (SEGWO cov.project_covariance), with the
    existing frozen-delay TDI-2 mixing. Link ij is received at i from j;
    the test-mass amplitude labels the mass in eta_ij's direct term.
    Only same-pixel real WDM XYZ covariance is retained, as in the demo.
    Targeted moves are enabled by default; the soft OMS ratio prior is opt-in.
    """

    name = 'LinkNoiseBlock'
    parameter_names = tuple(f'{kind}_{link}' for kind in ('Soms_d', 'Sa_a') for link in LINKS)
    fiducial = np.repeat(FIDUCIAL, 6)
    priors = np.repeat(PRIORS, 6, axis=0)

    @staticmethod
    def unit_model(f, d):
        return unit_bases(f, d, per_link=True)

    def __init__(self, data, column_ltts=None, *, nwalkers=32,
                 targeted_proposals=True, oms_log_power_ratio_sigma=None, **kwargs):
        self.targeted_proposals = targeted_proposals
        self.link_prior = OMSPairPrior(self.priors, oms_log_power_ratio_sigma)
        if nwalkers < 24:
            raise ValueError('The 12-parameter stretch move requires at least 24 walkers')
        if column_ltts is None:
            from inst_noise_demo import column_delays
            column_ltts = column_delays(data)
        super().__init__(data, column_ltts, nwalkers=nwalkers, **kwargs)

    def make_prior(self):
        return self.link_prior

    def make_moves(self):
        stretch = super().make_moves()
        if not self.targeted_proposals:
            return stretch
        return [(stretch, 0.5), (LinkNoiseMove(self.fiducial), 0.5)]


class FixedNoiseBlock(Block):
    """Publish a fixed covariance on every call; consume no random numbers."""

    def _check_data(self, data):
        if (data.data_domain, data.wdm_grid, data.fs, data.t0, data.channel_names,
            data.tdi_generation, data.physical_observable) != self._conventions:
            raise ValueError('Residual grid, epoch and conventions must match the block')

    def _bind(self, data):
        self._conventions = (data.data_domain, data.wdm_grid, data.fs, data.t0,
                             data.channel_names, data.tdi_generation, data.physical_observable)

    def initial_result(self, *, rng=None):
        return self._state

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        self._check_data(conditional_residual)
        return self._state


class InjectionNoiseBlock(FixedNoiseBlock):
    """Fix the shared OMS and acceleration ASDs to supplied injection values.

    L1Data currently carries no injection amplitudes: defaults are the demo's
    nominal (15 pm/sqrt(Hz), 3 fm/s²/sqrt(Hz)), not inferred file metadata.
    """

    name = 'InjectionNoiseBlock'

    def __init__(self, data, column_ltts=None, *, amplitudes=FIDUCIAL, **kwargs):
        amplitudes = np.asarray(amplitudes, dtype=float)
        if amplitudes.shape != (2,) or not np.all(np.isfinite(amplitudes) & (amplitudes > 0)):
            raise ValueError('amplitudes must contain two positive finite ASDs')
        if column_ltts is None:
            from inst_noise_demo import column_delays
            column_ltts = column_delays(data)
        model = InstrumentNoiseBlock(data, column_ltts, **kwargs)
        self.active_mask = model.active_mask
        self._bind(data)
        self._state = BlockState(covariance=model.covariance(np.log(amplitudes)),
                                 parameters=dict(zip(model.parameter_names, map(float, amplitudes))))


class CD1LNoiseEstimateBlock(FixedNoiseBlock):
    """Fold L1Data's one-sided XYZ Hz²/Hz spectra into real WDM covariance.

    Interpolate matrix entries linearly in log frequency and reception time.
    Frequency extrapolation is rejected. Time extrapolation holds the nearest
    available matrix (estimates need not span observation endpoints).
    Imaginary CSDs describe correlations omitted by this same-pixel real model.
    """

    name = 'CD1LNoiseEstimateBlock'

    def __init__(self, data, *, time_slices_crop=3, coarseQ=1):
        if (data.data_domain, data.channel_names, data.tdi_generation,
            data.physical_observable) != ('wdm', CHANNELS, '2.0', 'fractional_frequency'):
            raise ValueError('Block requires WDM XYZ TDI-2 fractional-frequency data')
        if data.noise_estimates_xyz is None or data.laser_frequency_hz is None:
            raise ValueError('CD1-L XYZ noise estimates and laser_frequency_hz are required')
        if not isinstance(coarseQ, (int, np.integer)) or coarseQ < 1:
            raise ValueError('coarseQ must be a positive integer')
        grid = data.wdm_grid
        nf, nt = grid.num_frequency_divisions, grid.num_time_divisions
        if not 0 <= 2 * time_slices_crop < nt:
            raise ValueError('Cropping must leave active columns')
        columns = np.arange(time_slices_crop, nt - time_slices_crop)
        frequencies = wdm.band_frequencies(nf, data.dt)
        layers = np.flatnonzero((frequencies >= 3e-4) & (frequencies <= 8e-3)
                                & (frequencies < data.fny))
        if not len(layers):
            raise ValueError('The analysis band has no WDM pixels')
        self.active_mask = np.zeros(grid.array_shape, dtype=bool)
        self.active_mask[np.ix_(layers, columns)] = True
        spectra = data.noise_estimates_xyz / data.laser_frequency_hz**2
        scale = np.max(np.abs(spectra))
        if not np.isfinite(spectra).all() or not np.allclose(
                spectra, spectra.swapaxes(-1, -2).conj(), rtol=1e-10, atol=scale * 1e-12):
            raise ValueError('XYZ noise estimates must be finite and Hermitian')
        # Taking the real symmetric part preserves positive definiteness.
        spectra = (spectra.real + spectra.real.swapaxes(-1, -2)) / 2
        if np.any(np.linalg.eigvalsh(spectra) <= 0):
            raise ValueError('XYZ noise estimates must be positive definite')
        phi2 = wdm.phi_centered(nf, nt, data.dt)**2
        keep = phi2 > 0
        weights = 0.5 * phi2[keep] / phi2.sum()
        offsets = (np.arange(nt) - nt // 2)[keep] / (nf * nt * data.dt)
        table_f = data.noise_estimate_frequencies_hz
        table_t = data.noise_estimate_sample_times_gps
        target_t = data.t0 + np.arange(nt) * nf * data.dt
        packed = np.empty((6, len(layers), nt))
        for layer_index, layer in enumerate(layers):
            query_f = frequencies[layer] + offsets
            if query_f.min() < table_f[0] or query_f.max() > table_f[-1]:
                raise ValueError('Noise-estimate frequencies must cover the WDM filter support')
            for k, (i, j) in enumerate(zip(UI, UJ)):
                folded = np.array([np.interp(np.log(query_f), np.log(table_f), row[:, i, j]) @ weights
                                   for row in spectra])
                packed[k, layer_index] = np.interp(target_t, table_t, folded)
        starts = np.arange(0, len(columns), coarseQ)
        sizes = np.diff(np.append(starts, len(columns)))
        coarse = np.add.reduceat(packed[..., columns], starts, axis=-1) / sizes
        packed[..., columns] = np.repeat(coarse, sizes, axis=-1)
        self._bind(data)
        covariance = packed_covariance(grid, layers, columns, packed)
        covariance.check_compatible(data)
        self._state = BlockState(covariance=covariance,
                                 parameters={'source': 'CD1-L noise_estimates/XYZ'})
