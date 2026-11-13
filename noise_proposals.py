"""Small GLASS-inspired moves for our 12 log-ASD coordinates.

Reference: glass/sgwb-dev/noise/src/glass_noise_sampler.c. We preserve all
12 parameters and our log-uniform prior, rather than tying reverse-link OMS.
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
    """

    def __init__(self, fiducial):
        super().__init__()
        self.center = np.log(np.asarray(fiducial))

    def get_proposal(self, branches_coords, random, **kwargs):
        coords = branches_coords['noise']
        if coords.shape[-2:] != (1, 12):
            raise ValueError('LinkNoiseMove requires one leaf with 12 log ASDs')
        q = coords.copy()
        factors = np.zeros(coords.shape[:2])
        for t, w in np.ndindex(factors.shape):
            old = coords[t, w, 0]
            kind = random.randint(3)
            scale = random.choice([0.001, 0.01, 0.1, 1.0])
            step = scale * random.normal()
            if kind == 0:
                q[t, w, 0, random.randint(12)] += step
                continue  # symmetric directly in log ASD
            power = np.exp(2 * (old - self.center))
            if kind == 1:
                i, j = REVERSE_PAIRS[random.randint(3)]
                offset = 6 * random.randint(2)  # OMS or acceleration
                power[offset + i] += step
                power[offset + j] -= step
            else:
                power[6:] += step * ACC_CYCLE
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
