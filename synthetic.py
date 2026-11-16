"""Synthetic WDM observations from Enchilada block states.

A data set is the sum of the states' templates plus Gaussian noise drawn from
their covariances: independent WDM pixels, each with its own XYZ correlations.
Those are the statistics the noise blocks' likelihood assumes, so a pipeline
should recover what was put in. Errors in the assumptions themselves
(correlations between pixels, the WDM fold, the equal-arm stochastic response)
are invisible here; exposing them needs a time-domain simulation.

    reference = blank_observation(768, 2048, 5.0)      # or a conditioned brick
    block = SGWBBlock(UnequalArmEqualNoiseBlock(reference, delays), PowerLawSGWB())
    truth = block.state({"Soms_d": 1.5e-11, "Sa_a": 3e-15,
                         "powerlaw_log10_A": -8.45, "powerlaw_alpha": 2 / 3})
    data = simulate(reference, [truth], block.active_mask, rng=np.random.default_rng(1))
"""

from dataclasses import replace

import numpy as np
from enchilada import L1Data, WDMGrid

from noise_model import CHANNELS


def blank_observation(num_frequency_divisions, num_time_divisions, dt, start_time_gps=0.0):
    """An all-zero WDM XYZ TDI-2 observation: a grid and conventions to simulate on.

    It carries no orbit, so give unequal-arm blocks their delays and equal-arm
    blocks an ``armlength_s``.
    """
    grid = WDMGrid(num_frequency_divisions=num_frequency_divisions,
                   num_time_divisions=num_time_divisions)
    return L1Data(
        channel_names=CHANNELS,
        channel_data={ch: np.zeros(grid.array_shape) for ch in CHANNELS},
        sample_rate_hz=1 / dt,
        start_time_gps=start_time_gps,
        data_domain="wdm",
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
    )


def simulate(reference, states, mask=None, *, rng):
    """``reference`` with its channel data replaced by a synthetic realization.

    ``states`` are BlockStates on ``reference``'s WDM grid, for instance
    ``block.state(parameters)`` of a noise block or a signal block's published
    state. Every template is added. Every covariance adds an independent noise
    draw on the pixels of ``mask``, the pixels that covariance models (a noise
    block's ``active_mask``; published covariances hold placeholders
    elsewhere), so pass each noise source once. Outside the mask the data hold
    templates only.
    """
    if reference.data_domain != "wdm":
        raise ValueError("simulate requires a WDM-domain reference")
    names, grid = reference.channel_names, reference.wdm_grid
    values = np.zeros((len(names), *grid.array_shape))
    for state in states:
        if state.template is not None:
            if set(state.template) != set(names) or any(
                    state.template[ch].shape != grid.array_shape for ch in names):
                raise ValueError("Templates must cover the reference's channels and WDM grid")
            values += [state.template[ch] for ch in names]
        if state.covariance is not None:
            if mask is None:
                raise ValueError("A covariance needs the mask of the pixels it models")
            state.covariance.check_compatible(reference)
            matrices = np.empty((np.count_nonzero(mask), len(names), len(names)))
            for i, first in enumerate(names):
                for j, second in enumerate(names[i:], i):
                    matrices[:, i, j] = matrices[:, j, i] = state.covariance.pair(first, second)[mask]
            draws = rng.normal(size=matrices.shape[:2])
            values[:, mask] += np.einsum("pij,pj->ip", np.linalg.cholesky(matrices), draws)
    values *= grid.active_mask  # L1Data requires zeros at inactive coefficients
    return replace(reference, channel_data=dict(zip(names, values)))
