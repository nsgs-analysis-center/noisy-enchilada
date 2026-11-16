"""Small GLASS-inspired moves for our per-link log-ASD coordinates.

Reference: glass/noise/src/glass_noise_sampler.c. Coordinates are OMS levels
followed by six test-mass levels in link order. Unequal arms keep all six OMS
levels (12 parameters) under our log-uniform prior; equal arms tie each
reverse OMS pair, as GLASS does, leaving three (9 parameters).
"""

import numpy as np
from eryn.moves import MHMove

# Enchilada order: (12, 23, 31, 13, 32, 21).
REVERSE_PAIRS = ((0, 5), (1, 4), (2, 3))
ACC_CYCLE = np.array([1., 1., 1., -1., -1., -1.])


class LinkNoiseMove(MHMove):
    """Equal mixture of local log steps, reverse-pair power swaps and acc cycles.

    Power increments use fixed nominal powers and scales 0.001, 0.01, 0.1, 1.
    Negative-power proposals are rejected, never clipped or redrawn. The
    returned Hastings factor converts symmetric power moves to log-ASD space.
    With 12 coordinates, swaps act on OMS or acceleration pairs; with 9
    (reverse OMS tied) they act on acceleration pairs only.
    """

    def __init__(self, fiducial):
        super().__init__()
        self.center = np.log(np.asarray(fiducial))
        self.ndim = len(self.center)
        if self.ndim not in (9, 12):
            raise ValueError('LinkNoiseMove requires 9 (tied OMS) or 12 log ASDs')
        self.n_oms = self.ndim - 6

    def get_proposal(self, branches_coords, random, **kwargs):
        coords = branches_coords['noise']
        if coords.shape[-2:] != (1, self.ndim):
            raise ValueError(f'LinkNoiseMove requires one leaf with {self.ndim} log ASDs')
        q = coords.copy()
        factors = np.zeros(coords.shape[:2])
        for t, w in np.ndindex(factors.shape):
            old = coords[t, w, 0]
            kind = random.randint(3)
            scale = random.choice([0.001, 0.01, 0.1, 1.0])
            step = scale * random.normal()
            if kind == 0:
                q[t, w, 0, random.randint(self.ndim)] += step
                continue  # symmetric directly in log ASD
            power = np.exp(2 * (old - self.center))
            if kind == 1:
                i, j = REVERSE_PAIRS[random.randint(3)]
                # OMS or acceleration; tied OMS pairs have no split to swap.
                offset = self.n_oms * random.randint(2) if self.n_oms == 6 else self.n_oms
                power[offset + i] += step
                power[offset + j] -= step
            else:
                power[self.n_oms:] += step * ACC_CYCLE
            if np.any(power <= 0):
                factors[t, w] = -np.inf
                continue
            new = self.center + 0.5 * np.log(power)
            q[t, w, 0] = new
            # q(old|new)/q(new|old) = product(P_old/P_new).
            factors[t, w] = 2 * np.sum(old - new)
        return {'noise': q}, factors


class OMSPairPrior:
    """Box prior on log ASDs, optionally penalizing reverse-link OMS ratios.

    sigma is the standard deviation of ln(P_ij/P_ji), not ASD ratio.
    None retains the original independent log-uniform prior. For fixed sigma
    this log density is sufficient for MCMC; its normalization is omitted.
    """

    def __init__(self, bounds, sigma=None):
        self.bounds = np.asarray(bounds)
        self.key_order = list(range(len(self.bounds)))  # Eryn backend column metadata
        if sigma is not None and (not np.isfinite(sigma) or sigma <= 0):
            raise ValueError('OMS log-power-ratio sigma must be positive and finite')
        self.sigma = sigma

    def logpdf(self, logs):
        logs = np.asarray(logs)
        inside = np.all(np.isfinite(logs) & (logs >= self.bounds[:, 0])
                        & (logs <= self.bounds[:, 1]), axis=-1)
        penalty = np.zeros(logs.shape[:-1])
        if self.sigma is not None:
            for i, j in REVERSE_PAIRS:
                penalty -= 0.5 * (2 * (logs[..., i] - logs[..., j]) / self.sigma)**2
        return np.where(inside, penalty, -np.inf)
