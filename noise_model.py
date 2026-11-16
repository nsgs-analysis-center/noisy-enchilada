"""TDI-2 XYZ instrument-noise physics and WDM machinery shared by the noise blocks.

Conventions: link ij carries light received at spacecraft i from j, in
Enchilada's LINKS order (12, 23, 31, 13, 32, 21). Covariances are real,
same-pixel XYZ matrices packed as XX, YY, ZZ, XY, XZ, YZ (indices UI, UJ).
A flat one-sided PSD S folds to WDM variance S/2.
"""

from dataclasses import dataclass, field

import numpy as np
import wdm
from enchilada import Covariance
from enchilada.data import LINKS
from numba import njit, prange

CHANNELS = ("X", "Y", "Z")
C_SI = 299792458.0
ARMLENGTH_S = 2.5e9 / C_SI  # nominal arm of the equal-arm stochastic response
BAND_HZ = (3e-4, 8e-3)
PRIORS = np.log([[6e-12, 2e-10], [1e-15, 2e-13]])  # log ASD bounds: OMS, acceleration
FIDUCIAL = (1.5e-11, 3e-15)  # OMS m/sqrt(Hz), acceleration m/s²/sqrt(Hz)
UI, UJ = [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]  # XX YY ZZ XY XZ YZ
CROSS_WEIGHTS = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])  # off-diagonal products count twice

__all__ = [
    "ARMLENGTH_S", "BAND_HZ", "CHANNELS", "C_SI", "FIDUCIAL", "LINKS", "PRIORS", "UI", "UJ",
    "AnalysisRegion", "check_xyz_wdm", "column_delays", "equal_arm_delays",
    "gaussian_loglike", "instrument_bases", "layer_nodes", "noise_loglike",
    "stochastic_response", "unit_bases", "wdm_filter_weights",
]


def check_xyz_wdm(data):
    """Require WDM-domain XYZ TDI-2 fractional-frequency data."""
    if (data.data_domain, data.channel_names, data.tdi_generation,
            data.physical_observable) != ("wdm", CHANNELS, "2.0", "fractional_frequency"):
        raise ValueError("Block requires WDM XYZ TDI-2 fractional-frequency data")


# --- Delays --------------------------------------------------------------------


def column_delays(data, stride=200):
    """Average Enchilada's directed delays in cells centred on WDM columns."""
    if data.light_travel_times_s is None:
        raise ValueError("Observation requires light-travel-time tables")
    times = data.light_travel_time_sample_times_gps[::stride]
    delays = data.light_travel_times_s[:, ::stride].T
    width = data.wdm_grid.num_frequency_divisions * data.dt
    nt = data.wdm_grid.num_time_divisions
    times = times - data.t0
    if times[0] > -width / 2 or times[-1] < (nt - 0.5) * width:
        raise ValueError("LTT table must cover the full WDM column cells")
    columns = np.floor(times / width + 0.5).astype(int)
    keep = (columns >= 0) & (columns < nt)
    counts = np.bincount(columns[keep], minlength=nt)
    if np.any(counts == 0):
        raise ValueError("LTT subsampling leaves empty WDM columns; reduce stride")
    return np.stack([
        np.bincount(columns[keep], weights=d[keep], minlength=nt) / counts for d in delays.T
    ], axis=1)


def equal_arm_delays(data, armlength_s=None):
    """Six equal, constant one-way delays (s) on every WDM column, shape (Nt, 6).

    The default is the mean of all six directed delays over the observation;
    pass ``armlength_s=2.5e9 / C_SI`` for GLASS's nominal LARM.
    """
    if armlength_s is None:
        armlength_s = column_delays(data).mean()
    armlength_s = float(armlength_s)
    if not np.isfinite(armlength_s) or armlength_s <= 0:
        raise ValueError("armlength_s must be a positive finite delay in seconds")
    return np.full((data.wdm_grid.num_time_divisions, 6), armlength_s)


# --- Spectral model ------------------------------------------------------------


def unit_bases(f, d, *, per_link=False):
    """Real XYZ TDI-2 covariances for unit OMS and acceleration levels.

    Port of lisatools._unequal_arm_fused. ``d`` holds the six link delays in
    LINKS order (``d_ij`` from spacecraft j to i); returns
    ``(2, *d.shape[:-1], 3, 3, len(f))``. With ``per_link=True``,
    return 12 components: six OMS, then six test masses in LINKS order.
    """
    D = np.exp(-2j * np.pi * d[..., None] * f)
    index = {link: i for i, link in enumerate(LINKS)}
    eta = np.zeros(D.shape[:-2] + (3, 6, f.size), complex)
    for ch in range(3):
        a1, a2, b1, b2 = (
            index[10 * ((link // 10 + ch - 1) % 3 + 1) + (link % 10 + ch - 1) % 3 + 1]
            for link in (13, 31, 12, 21)
        )
        a, b = D[..., a1, :] * D[..., a2, :], D[..., b1, :] * D[..., b2, :]
        eta[..., ch, a1, :] = (1 - b) * (1 - a * b)
        eta[..., ch, a2, :] = D[..., a1, :] * eta[..., ch, a1, :]
        eta[..., ch, b1, :] = (a - 1) * (1 - a * b)
        eta[..., ch, b2, :] = D[..., b1, :] * eta[..., ch, b1, :]
    # a test-mass noise enters its own link directly and the reverse link delayed
    rev = [index[int(str(link)[::-1])] for link in LINKS]
    tm = eta + eta[..., rev, :] * D[..., rev, :][..., None, :, :]
    oms = (2 * np.pi * f / C_SI) ** 2 * (1 + (2e-3 / f) ** 4)
    acc = (1 + (0.4e-3 / f) ** 2) * (1 + (f / 8e-3) ** 4) / (2 * np.pi * f * C_SI) ** 2
    if per_link:
        # C = M diag(S_link) M†, as in SEGWO's covariance projection.
        # Each test mass enters its own eta and the delayed reverse eta.
        return np.stack([
            (A[..., :, k, :][..., :, None, :] *
             A[..., :, k, :].conj()[..., None, :, :] * spectrum).real
            for A, spectrum in ((eta, oms), (tm, acc)) for k in range(6)
        ])
    return np.stack(
        [
            np.einsum("...ikf,...jkf,f->...ijf", A, A.conj(), s).real
            for A, s in ((eta, oms), (tm, acc))
        ]
    )


def wdm_filter_weights(nf, nt, dt):
    """Frequency offsets from a layer centre and weights of wdm's filter |phi|².

    ``sum(weights * S(f_layer + offsets))`` is the WDM variance of a one-sided
    PSD S; the weights sum to 1/2 (a flat PSD folds to S/2).
    """
    phi2 = wdm.phi_centered(nf, nt, dt) ** 2
    keep = phi2 > 0
    offsets = (np.arange(nt) - nt // 2)[keep] / (nf * nt * dt)
    return offsets, 0.5 * phi2[keep] / phi2.sum()


def instrument_bases(layers, grid, dt, column_ltts, unit_model=unit_bases):
    """Layer-calibrated WDM covariances: (components, 3, 3, layers, Nt).

    Fold at mean delays with wdm's actual filter, then calibrate the
    column-dependent band-centre values, as in the standalone sampler.
    """
    nf, nt = grid.num_frequency_divisions, grid.num_time_divisions
    offsets, weight = wdm_filter_weights(nf, nt, dt)
    f_mid = wdm.band_frequencies(nf, dt)[layers]
    ref = column_ltts.mean(axis=0)
    exact = np.stack([unit_model(f + offsets, ref) @ weight for f in f_mid], axis=-1)
    approx = 0.5 * unit_model(f_mid, ref)
    usable = np.abs(approx) > 1e-10 * np.abs(approx).max(axis=-1, keepdims=True)
    ratio = np.where(usable, exact / np.where(usable, approx, 1.0), 1.0)
    return 0.5 * np.moveaxis(unit_model(f_mid, column_ltts), 1, -1) * ratio[..., None]


def stochastic_response(f, armlength_s=ARMLENGTH_S):
    """Equal-arm TDI-2 X response to a strain PSD: 4 x² sin²x sin²2x, x = 2πfL/c.

    lisatools X2TDISens.stochastic_transform. An isotropic background has
    cross spectra XY = XZ = YZ = -XX/2 in this limit.
    """
    x = 2 * np.pi * armlength_s * f
    return 4 * x**2 * np.sin(x) ** 2 * np.sin(2 * x) ** 2


def layer_nodes(layers, grid, dt, num=64):
    """Quadrature nodes folding a smooth PSD through each layer's filter.

    Returns ``(f, weights)``, each (layers, nodes), with
    ``sum(weights * S(f), axis=-1)`` the WDM variance of a one-sided PSD S.
    Consecutive filter bins are merged into at most ``num`` nodes at their
    weighted mean frequency, as in the standalone sampler; with no more than
    ``num`` bins the fold is exact.
    """
    nf, nt = grid.num_frequency_divisions, grid.num_time_divisions
    offsets, weight = wdm_filter_weights(nf, nt, dt)
    pieces = np.array_split(np.arange(len(offsets)), min(num, len(offsets)))
    node_weight = np.array([weight[p].sum() for p in pieces])
    node_offset = np.array([offsets[p] @ weight[p] for p in pieces]) / node_weight
    f = wdm.band_frequencies(nf, dt)[layers][:, None] + node_offset
    return f, np.broadcast_to(node_weight, f.shape)


# --- Analysis region -----------------------------------------------------------


@dataclass(frozen=True, eq=False)
class AnalysisRegion:
    """Fitted WDM pixels: band layers × cropped columns, averaged in time cells.

    Cells hold ``coarseQ`` consecutive retained columns (the last may be
    shorter). Coarse pixels are flattened layer-major, matching ``weights``.
    """

    grid: object
    layers: np.ndarray
    columns: np.ndarray
    starts: np.ndarray
    sizes: np.ndarray
    active_mask: np.ndarray = field(init=False, repr=False)
    weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        mask = np.zeros(self.grid.array_shape, dtype=bool)
        mask[np.ix_(self.layers, self.columns)] = True
        object.__setattr__(self, "active_mask", mask)
        object.__setattr__(self, "weights", np.tile(self.sizes, len(self.layers)).astype(float))

    @classmethod
    def from_data(cls, data, *, time_slices_crop=3, coarseQ=1, band_hz=BAND_HZ):
        if not isinstance(coarseQ, (int, np.integer)) or coarseQ < 1:
            raise ValueError("coarseQ must be a positive integer")
        grid = data.wdm_grid
        nt = grid.num_time_divisions
        if not 0 <= 2 * time_slices_crop < nt:
            raise ValueError("Cropping must leave active columns")
        f = wdm.band_frequencies(grid.num_frequency_divisions, data.dt)
        layers = np.flatnonzero((f >= band_hz[0]) & (f <= band_hz[1]) & (f < data.fny))
        if not len(layers):
            raise ValueError("The analysis band has no WDM pixels")
        columns = np.arange(time_slices_crop, nt - time_slices_crop)
        starts = np.arange(0, len(columns), coarseQ)
        sizes = np.diff(np.append(starts, len(columns)))
        return cls(grid, layers, columns, starts, sizes)

    @property
    def num_cells(self):
        return len(self.starts)

    def coarsen(self, values):
        """Average ``values[..., columns]`` within cells: (..., Nt) -> (..., cells)."""
        return np.add.reduceat(values[..., self.columns], self.starts, axis=-1) / self.sizes

    def covariance(self, coarse):
        """Publish packed (6, layers, cells) values on the full grid.

        Enchilada requires positive variance on every physical WDM pixel: unit
        variance outside the region is a placeholder no block likelihood reads.
        """
        pairs = {}
        cells = np.ix_(self.layers, self.columns)
        for k, (i, j) in enumerate(zip(UI, UJ)):
            values = (self.grid.active_mask.astype(float) if i == j
                      else np.zeros(self.grid.array_shape))
            values[cells] = np.repeat(coarse[k], self.sizes, axis=-1)
            pairs[CHANNELS[i], CHANNELS[j]] = values
        return Covariance(channel_names=CHANNELS, pairs=pairs)


# --- Likelihood kernels ----------------------------------------------------------

TILE = 256  # pixels per parallel work item; a tile of 12-component bases fits in L2


@njit(cache=True)
def gaussian_loglike(C, X):
    """One real XYZ pixel; X contains doubled cross products."""
    xx, yy, zz, xy, xz, yz = C
    cof = (yy * zz - yz**2, xx * zz - xz**2, xx * yy - xy**2,
           xz * yz - zz * xy, xy * yz - yy * xz, xy * xz - xx * yz)
    det = xx * cof[0] + xy * cof[3] + xz * cof[4]
    if not (xx > 0 and cof[2] > 0 and det > 0):  # also rejects NaN
        return -np.inf
    quad = 0.0
    for k in range(6):
        quad += cof[k] * X[k]
    return -0.5 * (quad / det + np.log(det))


@njit(cache=True)
def _tile_loglike(levels, B, X, weights, tile, out):
    """Accumulate one tile of pixels into ``out`` (walkers,)."""
    C = np.empty(6)
    cells = B.shape[0] // levels.shape[1]
    for p in range(tile * TILE, min(B.shape[0], (tile + 1) * TILE)):
        layer = p // cells
        for w in range(levels.shape[0]):
            for k in range(6):
                C[k] = 0.0
            for c in range(levels.shape[2]):
                level = levels[w, layer, c]
                for k in range(6):
                    C[k] += level * B[p, c, k]
            out[w] += weights[p] * gaussian_loglike(C, X[p])


@njit(cache=True)
def _loglike_serial(levels, B, X, weights):
    partial = np.zeros(((B.shape[0] + TILE - 1) // TILE, levels.shape[0]))
    for tile in range(partial.shape[0]):
        _tile_loglike(levels, B, X, weights, tile, partial[tile])
    return partial.sum(axis=0)


@njit(cache=True, parallel=True)
def _loglike_parallel(levels, B, X, weights):
    partial = np.zeros(((B.shape[0] + TILE - 1) // TILE, levels.shape[0]))
    for tile in prange(partial.shape[0]):
        _tile_loglike(levels, B, X, weights, tile, partial[tile])
    return partial.sum(axis=0)


PARALLEL_MIN_TILES = 3  # measured crossover: fewer tiles run faster serially


def noise_loglike(levels, B, X, weights):
    """Weighted log likelihood per walker, without allocating covariances.

    ``levels`` (walkers, components) are powers; ``B`` (pixels, components, 6)
    and ``X`` (pixels, 6) are pixel-major so each pixel's bases are loaded
    once for all walkers. Levels that depend on the layer are passed as
    (walkers, layers, components), with pixels layer-major and the same
    number of pixels in every layer. Tile partial sums are added in a fixed
    order, so the serial and parallel paths and any thread count give
    identical results.
    """
    levels = np.ascontiguousarray(levels, dtype=float)
    if levels.ndim == 2:
        levels = levels[:, None]
    if B.shape[0] % levels.shape[1]:
        raise ValueError("Pixels must divide evenly among the layers of levels")
    tiles = (B.shape[0] + TILE - 1) // TILE
    kernel = _loglike_parallel if tiles >= PARALLEL_MIN_TILES else _loglike_serial
    return kernel(levels, B, X, weights)
