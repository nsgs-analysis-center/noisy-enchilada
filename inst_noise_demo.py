"""Unequal-arm instrument noise demo. Run: uv run python inst_noise_demo.py"""

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import wdm
from enchilada import Block, BlockResult, DataCovariance, L1Data, NumericOrbit, Wheel, transform
from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer, UniformDistribution
from eryn.state import State
from numba import njit
from scipy import signal

CHANNELS = ("X", "Y", "Z")
LINKS = (12, 23, 31, 13, 32, 21)  # ij: received at i from j; keep both directions
C_SI = 299792458.0
PRIORS = np.log([[6e-12, 2e-10], [1e-15, 2e-13]])
FIDUCIAL = (1.5e-11, 3e-15)
UI, UJ = [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]  # XX YY ZZ XY XZ YZ

# --- Missing Enchilada helpers: issues #4 (reader) and #3 (directed LTTs) -------


def read_cd1l(path, *, duration_s=None):
    """XYZ TDI-2, in fractional frequency; let Enchilada load the orbit."""
    with h5py.File(path, "r") as f:
        sampling = f["tdis/sampling"].attrs
        fs, t0 = float(sampling["fs"]), float(sampling["t0"])
        count = int(sampling["size"]) if duration_s is None else int(np.ceil(duration_s * fs))
        if not 0 < count <= sampling["size"]:
            raise ValueError("Requested duration exceeds the brick or is non-positive")
        data = {ch: f[f"tdis/{ch}2"][:count] / f.attrs["laser_frequency"] for ch in CHANNELS}
    return L1Data(
        channel_names=CHANNELS, channel_data=data, sample_rate_hz=fs,
        start_time_gps=t0, tdi_generation="2.0", physical_observable="fractional_frequency",
        orbit_ephemeris=NumericOrbit.from_hdf5(path),
    )


def read_ltts(path, stride=200):
    """Absolute times and six directed delays in LINKS order, all in seconds."""
    with h5py.File(path, "r") as f:
        sampling = f["ltts/sampling"].attrs
        t0, dt = float(sampling["t0"]), float(sampling["dt"])
        delays = np.stack([f[f"ltts/ltt_{link}"][::stride] for link in LINKS], axis=1)
    return t0 + np.arange(len(delays)) * dt * stride, delays


def column_delays(times, delays, data):
    """Average tabulated delays in cells centred on WDM time columns."""
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


def load_observation(path, nf=768, nt=256, dt=5.0):
    """Reference conditioning: high-pass, resample, trim 200 h, Tukey taper."""
    trim = int(200 * 3600 / dt)
    raw = read_cd1l(path, duration_s=(nf * nt + 2 * trim) * dt)
    down = int(round(dt * raw.fs))
    if down < 1 or not np.isclose(down, dt * raw.fs):
        raise ValueError("dt must be an integer multiple of the input interval")
    xyz = np.stack([raw.channel_data[ch] for ch in CHANNELS])
    xyz = signal.sosfiltfilt(signal.butter(2, 2e-5, "highpass", fs=raw.fs, output="sos"), xyz)
    xyz = signal.resample_poly(xyz, 1, down, axis=-1, window=("kaiser", 31.0), padtype="line")
    xyz = xyz[:, trim:trim + nf * nt] * signal.windows.tukey(nf * nt, 4.0 / nt)
    data = replace(raw, channel_data=dict(zip(CHANNELS, xyz)), num_time_samples=nf * nt,
                   sample_rate_hz=1 / dt, start_time_gps=raw.t0 + trim * dt)
    data = transform(data, "wdm", num_frequency_divisions=nf, num_time_divisions=nt)
    return data, column_delays(*read_ltts(path), data)


# --- Instrument model: from noise_sampler_minimal.py, with wdm's filter -------


def unit_bases(f, d):
    """Real XYZ TDI-2 covariances for unit OMS and acceleration levels.

    Port of lisatools._unequal_arm_fused. ``d`` holds the six link delays in
    LINKS order (``d_ij`` from spacecraft j to i); returns
    ``(2, *d.shape[:-1], 3, 3, len(f))``.
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
    return np.stack(
        [
            np.einsum("...ikf,...jkf,f->...ijf", A, A.conj(), s).real
            for A, s in ((eta, oms), (tm, acc))
        ]
    )


def instrument_bases(layers, grid, dt, column_ltts):
    """Layer-calibrated WDM covariances: (2, 3, 3, layers, Nt).

    Fold at mean delays with wdm's actual filter, then calibrate the
    column-dependent band-centre values, as in the standalone sampler.
    A flat one-sided PSD folds to S/2 in the wdm normalization.
    """
    nf, nt = grid.num_frequency_divisions, grid.num_time_divisions
    phi2 = wdm.phi_centered(nf, nt, dt) ** 2
    keep = phi2 > 0
    weight = 0.5 * phi2[keep] / phi2.sum()
    offsets = (np.arange(nt) - nt // 2)[keep] / (nf * nt * dt)
    f_mid = wdm.band_frequencies(nf, dt)[layers]
    ref = column_ltts.mean(axis=0)
    exact = np.stack([unit_bases(f + offsets, ref) @ weight for f in f_mid], axis=-1)
    approx = 0.5 * unit_bases(f_mid, ref)
    usable = np.abs(approx) > 1e-10 * np.abs(approx).max(axis=-1, keepdims=True)
    ratio = np.where(usable, exact / np.where(usable, approx, 1.0), 1.0)
    return 0.5 * np.moveaxis(unit_bases(f_mid, column_ltts), 1, -1) * ratio[..., None]


@njit(cache=True)
def gaussian_loglike(C, X):
    """One real XYZ pixel; X contains doubled cross products."""
    xx, yy, zz, xy, xz, yz = C
    cof = (yy * zz - yz**2, xx * zz - xz**2, xx * yy - xy**2,
           xz * yz - zz * xy, xy * yz - yy * xz, xy * xz - xx * yz)
    det = xx * cof[0] + xy * cof[3] + xz * cof[4]
    if xx <= 0 or cof[2] <= 0 or det <= 0:
        return -np.inf
    quad = 0.0
    for k in range(6):
        quad += cof[k] * X[k]
    return -0.5 * (quad / det + np.log(det))


@njit(cache=True)
def noise_loglike(rows, B, X, weights):
    """Fused walker/pixel loop; no full covariance allocation per proposal."""
    scores = np.zeros(len(rows))
    C = np.empty(6)
    for i in range(len(rows)):
        oms, acc = np.exp(2 * rows[i, 0]), np.exp(2 * rows[i, 1])
        for p in range(X.shape[1]):
            for k in range(6):
                C[k] = oms * B[0, k, p] + acc * B[1, k, p]
            scores[i] += weights[p] * gaussian_loglike(C, X[:, p])
    return scores

# --- Noise block: fixed configuration; walkers and results belong to Wheel ----


class InstrumentNoiseBlock(Block):
    """
    Two-parameter instrument-only covariance, 0.3–8 mHz
    Translated from the unequal-arm model we used in CD1-L analyses.
    Assumes all links have equal and stationary Sacc and Soms
    """

    name = "InstrumentNoiseBlock"

    def __init__(self, data, column_ltts, *, nwalkers=16, steps_per_call=100, time_slices_crop=3, coarseQ=1):
        if (data.data_domain, data.channel_names, data.tdi_generation,
                data.physical_observable) != ("wdm", CHANNELS, "2.0", "fractional_frequency"):
            raise ValueError("Block requires WDM XYZ TDI-2 fractional-frequency data")
        self.grid, self.fs, self.t0 = data.wdm_grid, data.fs, data.t0
        self.nwalkers = nwalkers
        self.steps_per_call = steps_per_call
        self.time_slices_crop = time_slices_crop
        if not isinstance(coarseQ, (int, np.integer)) or coarseQ < 1:
            raise ValueError("coarseQ must be a positive integer")
        f = wdm.band_frequencies(self.grid.num_frequency_divisions, data.dt)
        self.layers = np.flatnonzero((f >= 3e-4) & (f <= 8e-3) & (f < data.fny))
        self.active_mask = np.zeros(self.grid.array_shape, dtype=bool)
        if not 0 <= 2 * time_slices_crop < self.grid.num_time_divisions:
            raise ValueError("Cropping must leave active columns")
        stop = self.grid.num_time_divisions - time_slices_crop
        self.columns = np.arange(time_slices_crop, stop)
        self.active_mask[np.ix_(self.layers, self.columns)] = True
        if not self.active_mask.any():
            raise ValueError("The analysis band has no WDM pixels after trimming")
        self.bases = instrument_bases(self.layers, self.grid, data.dt, column_ltts)[:, UI, UJ]
        self.starts = np.arange(0, len(self.columns), coarseQ)
        self.sizes = np.diff(np.append(self.starts, len(self.columns)))
        fine = self.bases[..., self.columns]
        coarse = np.add.reduceat(fine, self.starts, axis=-1) / self.sizes
        self.coarse_bases = np.ascontiguousarray(coarse.reshape(2, 6, -1))
        self.weights = np.tile(self.sizes, len(self.layers)).astype(float)
        # Publish the same piecewise-constant model used by the coarse fit.
        self.bases[..., self.columns] = np.repeat(coarse, self.sizes, axis=-1)

    def log_likelihood(self, data):
        if (data.wdm_grid, data.fs, data.t0, data.channel_names) != (
                self.grid, self.fs, self.t0, CHANNELS):
            raise ValueError("Residual grid, epoch and channels must match the block")
        w = np.stack([data.channel_data[ch][self.layers] for ch in CHANNELS])
        w = w[..., self.columns]
        products = w[UI] * w[UJ] * np.array([1, 1, 1, 2, 2, 2])[:, None, None]
        X = np.add.reduceat(products, self.starts, axis=-1) / self.sizes
        X = np.ascontiguousarray(X.reshape(6, -1))

        def loglike(rows):
            return noise_loglike(np.atleast_2d(rows), self.coarse_bases, X, self.weights)

        return loglike

    def covariance(self, logs):
        packed = np.einsum("b,beft->eft", np.exp(2 * logs), self.bases)
        matrix = np.zeros((*self.grid.array_shape, 3, 3))
        for k, (i, j) in enumerate(zip(UI, UJ)):
            matrix[self.layers, :, i, j] = matrix[self.layers, :, j, i] = packed[k]
        return DataCovariance(
            covariance_matrix=matrix, channel_names=CHANNELS, sample_rate_hz=self.fs,
            start_time_gps=self.t0, num_time_samples=self.grid.num_time_samples,
            data_domain="wdm", wdm_grid=self.grid, active_mask=self.active_mask,
        )

    def _result(self, coords, iterations):
        # Publish a fixed walker's draw, and preserve the entire ensemble.
        logs = coords[0, 0, 0]
        return BlockResult(
            noise_covariance=self.covariance(logs),
            model_parameters=dict(zip(("Soms_d", "Sa_a"), map(float, np.exp(logs)))),
            sampler_state={"coords": coords.copy(), "iterations": iterations},
        )

    def initial_result(self, *, rng):
        center = np.log(FIDUCIAL)
        coords = center + rng.normal(0, 0.05, size=(1, self.nwalkers, 1, 2))
        coords[0, 0, 0] = center
        return self._result(coords, 0)

    def sample(self, conditional_residual, noise_covariance, current_block_result, *, rng):
        prior = ProbDistContainer({i: UniformDistribution(*b) for i, b in enumerate(PRIORS)})
        sampler = EnsembleSampler(
            self.nwalkers,
            [2],
            self.log_likelihood(conditional_residual),
            {"noise": prior},
            branch_names=["noise"],
            nleaves_max=[1],
            nleaves_min=[1],
            vectorize=True,
            moves=StretchMove(randomize_split=False),  # avoid Eryn's global RNG shuffle
            tempering_kwargs={
                "betas": np.array([1.0]),
                "adaptive": False,
                "permute": False
                },
        )
        sampler.random_state = np.random.RandomState(int(rng.integers(2**32))).get_state()
        # Fresh State forces likelihood evaluation against this call's residual.
        state = State({"noise": current_block_result.sampler_state["coords"].copy()})
        final = sampler.run_mcmc(state, self.steps_per_call, progress=False)
        return self._result(
            final.branches_coords["noise"],
            current_block_result.sampler_state["iterations"] + self.steps_per_call,
        )


if __name__ == "__main__":
    NOISE_FILE = Path(__file__).with_name("NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5")
    NF, NT, DT = 768, 256, 5.0
    data, delays = load_observation(NOISE_FILE, NF, NT, DT)
    block = InstrumentNoiseBlock(data, delays, coarseQ=16)
    wheel = Wheel(data, random_seed=42)
    initial = block.initial_result(rng=np.random.default_rng(42))
    wheel.add(block, initial_block_result=initial, data_domain="wdm")
    for cycle in range(10):
        wheel.run(1)
        print("Adding to the ledger:", wheel.ledger['InstrumentNoiseBlock'].model_parameters)
