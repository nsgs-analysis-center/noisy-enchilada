#!/usr/bin/env python
"""Minimal, self-contained WDM noise sampler for the mojito L1 bricks.

Needs eryn, numpy, scipy, h5py, numba and WDMWaveletTransforms -- no lisatools.
Data conditioning, noise model and likelihood follow noise_sampler.py:
unequal-arm TDI-2 instrument with per-column link delays, tanh galactic
foreground with a tabulated modulation, "layer_calibrated" WDM PSDs, and an
optional coarse-WDM likelihood.

WDM coefficients and covariances follow the lisatools conventions (the
WDMWaveletTransforms output is rescaled by sqrt(dt) and re-signed per pixel),
so log L is directly comparable with lisatools'.

    python noise_sampler_minimal.py                                  # lite instrument
    python noise_sampler_minimal.py --mode foreground --grid full
    python noise_sampler_minimal.py --mode foreground --grid two-years --coarse-Q 150
"""

import argparse
import os

import h5py
import numpy as np
from eryn.backends import HDFBackend
from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
from eryn.moves.tempering import make_ladder
from eryn.priors import ProbDistContainer, UniformDistribution
from eryn.state import State
from scipy import signal
from WDMWaveletTransforms.transform_freq_funcs import phitilde_vec_norm
from WDMWaveletTransforms.wavelet_transforms import transform_wavelet_freq_time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
NOISE_FILE = os.path.join(ROOT, "NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5")
GALFOR_FILE = os.path.join(ROOT, "GALFOR_731d_2.5s_L1.h5")
MODULATION_FILE = os.path.join(HERE, "..", "modulation_unequal.dat")

C_SI, L_SI = 299792458.0, 2.5e9
DT, NF, MIN_FREQ, MAX_FREQ = 5.0, 768, 3e-4, 8e-3
HIGHPASS, TRIM, TAPER_WAVELETS, LTT_STRIDE, NODES = 2e-5, 200 * 3600.0, 2, 200, 64
LINKS = (12, 23, 31, 13, 32, 21)
UI, UJ = [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]  # symmetric 3x3 elements XX YY ZZ XY XZ YZ
GRIDS = {"lite": 256, "full": 1024, "two-years": None}

# Uniform prior support in the sampling basis, as in lisatools.globalfit.stock.erebor.noise.
PRIORS = (
    np.log((6e-12, 2e-10)),  # ln Soms_d
    np.log((1e-15, 2e-13)),  # ln Sa_a
    np.log10((1e-47, 1e-41)),  # log10 amp
    np.log10((1e-5, 1e-2)),  # log10 fk
    (1e-3, 5.0),  # alpha
    np.log10((1e-5, 1e-2)),  # log10 f_1
    np.log10((1e-5, 1e-2)),  # log10 f_2
)
LOG10 = (True, True, False, True, True)
LABELS = ("ln Soms_d", "ln Sa_a", "log10 amp", "log10 fk", "alpha", "log10 f_1", "log10 f_2")
# Welch-Satterthwaite fiducials for the coarse cell weights
PSD_FIDUCIAL = (1.5e-11, 3e-15)
GALFOR_FIDUCIAL = (3.26651613e-44, 2.09278117e-03, 1.18300266, 1.14556409e-03, 3.38095297e-04)


# --- data --------------------------------------------------------------------


def read_xyz(path):
    """``(xyz (3, N) in Doppler units, t0, fs)`` from a mojito L1 brick."""
    with h5py.File(path, "r") as f:
        samp = f["tdis/sampling"].attrs
        xyz = np.stack([f["tdis"][c][:] for c in ("X2", "Y2", "Z2")])
        return xyz * (1.0 / f.attrs["laser_frequency"]), float(samp["t0"]), float(samp["fs"])


def tukey(n, alpha):
    """lisatools.utils.utility.windowfun("tukey", n, alpha)."""
    w = np.zeros(n)
    r = 2.0 * np.arange(n) / (alpha * (n - 1))
    l1, l2 = int(np.rint(alpha * (n - 1) / 2)), int(np.rint((n - 1) * (1 - alpha / 2)))
    w[:l1] = 0.5 * (1 + np.cos(np.pi * (r[:l1] - 1)))
    w[l1:l2] = 1.0
    w[l2 : n - 1] = 0.5 * (1 + np.cos(np.pi * (r[l2 : n - 1] - 2 / alpha + 1)))
    return w


def load_wdm(noise_file, galfor_file, nt):
    """Condition like the global-fit engine, then WDM-transform.

    Returns ``(w (3, layers, Nt), layers, data_t0)``; ``nt=None`` takes the
    largest even grid the conditioned record covers.
    """
    xyz, t0, fs = read_xyz(noise_file)
    if galfor_file:
        xyz = xyz + read_xyz(galfor_file)[0]
    xyz = signal.sosfiltfilt(signal.butter(2, HIGHPASS, "highpass", fs=fs, output="sos"), xyz)
    xyz = signal.resample_poly(
        xyz, 1, int(round(DT * fs)), axis=-1, window=("kaiser", 31.0), padtype="line"
    )
    trim = int(TRIM / DT)
    xyz, data_t0 = xyz[:, trim:-trim], t0 + trim * DT
    if nt is None:
        nt = (xyz.shape[1] - 1) // NF // 2 * 2
    n = NF * nt
    xyz = xyz[:, :n] * tukey(n, 2.0 * TAPER_WAVELETS / nt)
    layers = np.arange(int(np.ceil(MIN_FREQ * 2 * NF * DT)), int(MAX_FREQ * 2 * NF * DT) + 1)
    w = np.stack([transform_wavelet_freq_time(x, NF, nt).T[layers] for x in xyz])
    # lisatools convention: sqrt(dt) scale and (-1)^(m+n) sign per pixel
    return w * np.sqrt(DT) * (-1.0) ** np.add.outer(layers, np.arange(nt)), layers, data_t0


# --- noise model -------------------------------------------------------------


def link_delays(noise_file, data_t0):
    """``(t, ltts (n, 6))``: the brick's /ltts every LTT_STRIDE samples, t on the data clock."""
    with h5py.File(noise_file, "r") as f:
        t0, dt = (float(f["ltts/sampling"].attrs[k]) for k in ("t0", "dt"))
        ltts = np.stack([f["ltts"][f"ltt_{link}"][::LTT_STRIDE] for link in LINKS], axis=1)
    return t0 + np.arange(len(ltts)) * dt * LTT_STRIDE - data_t0, ltts


def column_delays(t, ltts, nt):
    """``(Nt, 6)`` mean delays inside each WDM time column (centred on n * Nf * dt)."""
    col = np.floor(t / (NF * DT) + 0.5).astype(int)
    keep = (col >= 0) & (col < nt)
    sums = np.stack([np.bincount(col[keep], ltts[keep, i], nt) for i in range(6)], axis=1)
    return sums / np.bincount(col[keep], minlength=nt)[:, None]


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
        [np.einsum("...ikf,...jkf,f->...ijf", A, A.conj(), s).real for A, s in ((eta, oms), (tm, acc))]
    )


def fold_frequencies(layers, nt):
    """WDM fold: ``E[w_mn^2] = sum_j weight[j] * S(f[m, j])`` for a one-sided PSD ``S``.

    lisatools normalization, so a flat ``S`` folds to ``0.5 * S``.
    """
    j = np.arange(-nt // 2, nt // 2)
    phi = 2.0 / NF * phitilde_vec_norm(NF, nt, 4.0)
    weight = NF / (4.0 * nt) * phi[np.abs(j)] ** 2
    return (layers[:, None] * (nt // 2) + j) / (NF * nt * DT), weight


def instrument_bases(layers, nt, column_ltts, cell_ltts):
    """``(2, 3, 3, layers, cells)`` unit instrument covariances, "layer_calibrated".

    Each cell takes its layer-centre value at its own delays, corrected per
    (basis, element, layer) by one exact fold at the run-mean delays.
    """
    f_fold, weight = fold_frequencies(layers, nt)
    f_mid = layers / (2 * NF * DT)
    ref = column_ltts.mean(axis=0)
    exact = np.stack([unit_bases(f, ref) @ weight for f in f_fold], axis=-1)
    approx = weight.sum() * unit_bases(f_mid, ref)
    usable = np.abs(approx) > 1e-10 * np.abs(approx).max(axis=-1, keepdims=True)
    ratio = np.where(usable, exact / np.where(usable, approx, 1.0), 1.0)
    return weight.sum() * np.moveaxis(unit_bases(f_mid, cell_ltts), 1, -1) * ratio[..., None]


def galfor_nodes(layers, nt):
    """64 window-weighted frequency nodes per layer and their parameter-free factors.

    Returns ``(f, log f, f^(-7/3) * TDI-2 X response * node weight)``, each ``(layers, 64)``.
    """
    f, weight = fold_frequencies(layers, nt)
    pieces = np.array_split(np.flatnonzero(weight > 0), NODES)
    f = np.stack([f[:, p] @ weight[p] / weight[p].sum() for p in pieces], axis=1)
    x = 2 * np.pi * L_SI / C_SI * f
    fixed = f ** (-7 / 3) * 4 * x**2 * np.sin(x) ** 2 * np.sin(2 * x) ** 2
    return f, np.log(f), fixed * np.array([weight[p].sum() for p in pieces])


def galfor_column(params, nodes):
    """WDM variance per layer of the tanh foreground through the TDI-2 X response."""
    amp, fk, alpha, f1, f2 = params
    f, log_f, fixed = nodes
    shape = np.exp(-np.exp(alpha * (log_f - np.log(f1)))) * (1 + np.tanh((fk - f) / f2))
    return 0.5 * amp * np.sum(shape * fixed, axis=-1)


def modulation(path, t):
    """Tabulated (t, XX, YY, ZZ, XY, XZ, YZ) modulation at mission times ``t``: ``(6, len(t))``."""
    g = np.loadtxt(path)
    g = g if g.shape[1] == 7 else g.T
    return np.array([np.interp(t, g[:, 0], g[:, i]) for i in range(1, 7)])


# --- likelihood --------------------------------------------------------------


def gaussian_loglike(C, X, weight):
    """``-1/2 sum weight * (tr(C^-1 X) + ln det C)`` over symmetric 3x3 pixels.

    ``C`` and ``X`` hold the XX YY ZZ XY XZ YZ elements, with the off-diagonals
    of ``X`` doubled. The caller rejects non-finite scores.
    """
    xx, yy, zz, xy, xz, yz = C
    cof = (
        yy * zz - yz * yz,
        xx * zz - xz * xz,
        xx * yy - xy * xy,
        xz * yz - zz * xy,
        xy * yz - yy * xz,
        xy * xz - xx * yz,
    )
    det = xx * cof[0] + xy * cof[3] + xz * cof[4]
    quad = sum(k * x for k, x in zip(cof, X))
    with np.errstate(divide="ignore", invalid="ignore"):
        return -0.5 * np.sum(weight * (quad / det + np.log(det)))


def build(
    mode,
    grid="lite",
    coarse_Q=1,
    ws=True,
    noise_file=NOISE_FILE,
    galfor_file=GALFOR_FILE,
    modulation_file=MODULATION_FILE,
    psd_fiducial=PSD_FIDUCIAL,
    likelihood_backend="numba",
    threads=1,
    fastmath=False,
):
    """Load the data, build the noise model; returns ``(loglike, parts)``.

    ``loglike`` maps ``(N, ndim)`` sampling-basis rows to ``(N,)``. The default
    Numba CPU kernel fuses pixel arithmetic; ``likelihood_backend="numpy"``
    selects the reference below. ``threads`` and ``fastmath`` apply to Numba
    only; fastmath allows reassociation and fused multiply-adds.
    """
    foreground = mode == "foreground"
    w, layers, data_t0 = load_wdm(noise_file, galfor_file if foreground else None, GRIDS[grid])
    nt = w.shape[-1]
    t_cols = np.arange(nt) * NF * DT
    t_ltt, ltts = link_delays(noise_file, data_t0)
    column_ltts = column_delays(t_ltt, ltts, nt)

    starts = np.arange(0, nt, coarse_Q)
    sizes = np.minimum(coarse_Q, nt - starts)

    def cell_mean(a):
        return np.add.reduceat(a, starts, axis=-1) / sizes

    # symmetric 3x3 quantities keep only their XX YY ZZ XY XZ YZ elements
    X = w[UI] * w[UJ] * np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])[:, None, None]
    weight = 1.0
    if coarse_Q == 1:
        cell_ltts = column_ltts
    else:
        t_cells = cell_mean(t_cols)
        cell_ltts = np.stack([np.interp(t_cells, t_ltt, ltts[:, i]) for i in range(6)], axis=1)
        X = cell_mean(X)
    B = instrument_bases(layers, nt, column_ltts, cell_ltts)[:, UI, UJ]

    nodes = M = None
    if foreground:
        nodes = galfor_nodes(layers, nt)
        M_fine = modulation(modulation_file, t_cols + data_t0)
        M = cell_mean(M_fine)

    if coarse_Q > 1:
        if ws:
            # cell dof moment-matched to the fiducial fine diagonal, per channel
            s = np.asarray(psd_fiducial) ** 2
            diag = np.repeat(s[0] * B[0, :3] + s[1] * B[1, :3], sizes, axis=-1)
            if foreground:
                diag = diag + galfor_column(GALFOR_FIDUCIAL, nodes)[:, None] * M_fine[:3, None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                dof = np.minimum(cell_mean(diag) ** 2 * sizes / cell_mean(diag**2), sizes)
            weight = np.where((dof > 0).all(axis=0), dof.mean(axis=0), 0.0)
        else:
            weight = sizes.astype(float)

    def loglike(x):
        x = np.atleast_2d(x)
        out = np.empty(len(x))
        for i, row in enumerate(x):
            C = np.exp(2 * row[0]) * B[0]
            C += np.exp(2 * row[1]) * B[1]
            if foreground:
                params = [10**v if lg else v for v, lg in zip(row[2:], LOG10)]
                C += galfor_column(params, nodes)[:, None] * M[:, None, :]
            out[i] = gaussian_loglike(C, X, weight)
        return np.where(np.isfinite(out), out, -1e300)

    if likelihood_backend == "numba":
        from noise_likelihood_numba import make_likelihood

        loglike = make_likelihood(B, X, weight, M, nodes, threads=threads, fastmath=fastmath)

    parts = dict(w=w, layers=layers, data_t0=data_t0, B=B, M=M, X=X, weight=weight, nodes=nodes)
    return loglike, parts


# --- sampler -----------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("instrument", "foreground"), default="instrument")
    p.add_argument("--grid", choices=tuple(GRIDS), default="lite")
    p.add_argument("--coarse-Q", type=int, default=1, help="average Q WDM time columns (1 = exact)")
    p.add_argument("--bartlett", action="store_true", help="coarse weights = cell sizes, not WS")
    p.add_argument("--nwalkers", type=int, default=16)
    p.add_argument("--ntemps", type=int, default=4)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--likelihood-backend", choices=("numba", "numpy"), default="numba")
    p.add_argument("--threads", type=int, default=1, help="Numba likelihood threads (default: 1)")
    p.add_argument("--fastmath", action="store_true", help="Numba reassociation and fused multiply-adds")
    p.add_argument("--out", help="eryn HDF5 backend (resumed if it exists)")
    p.add_argument("--noise-file", default=NOISE_FILE)
    p.add_argument("--galfor-file", default=GALFOR_FILE)
    p.add_argument("--modulation", default=MODULATION_FILE)
    args = p.parse_args()

    loglike, _ = build(
        args.mode, args.grid, args.coarse_Q, not args.bartlett, args.noise_file, args.galfor_file, args.modulation,
        likelihood_backend=args.likelihood_backend, threads=args.threads, fastmath=args.fastmath,
    )
    ndim = 2 if args.mode == "instrument" else 7
    priors = {"noise": ProbDistContainer({i: UniformDistribution(*PRIORS[i]) for i in range(ndim)})}
    betas = (
        np.array([1.0]) if args.ntemps == 1 else make_ladder(10 * ndim, ntemps=args.ntemps, Tmax=np.inf)
    )
    sampler = EnsembleSampler(
        args.nwalkers,
        [ndim],
        loglike,
        priors,
        branch_names=["noise"],
        nleaves_max=[1],
        nleaves_min=[1],
        vectorize=True,
        tempering_kwargs=dict(betas=betas, permute=False),
        moves=StretchMove(live_dangerously=True),
        num_repeats_in_model=5,
        backend=HDFBackend(args.out or f"noise_minimal_{args.mode}_{args.grid}_cQ{args.coarse_Q}.h5"),
    )
    if sampler.backend.iteration:
        state = sampler.backend.get_last_sample()
    else:
        state = State({"noise": priors["noise"].rvs(size=(args.ntemps, args.nwalkers, 1))})
    sampler.run_mcmc(state, args.iterations, progress=True)

    chain = sampler.get_chain()["noise"][:, 0, :, 0, :]
    lo, med, hi = np.percentile(chain[len(chain) // 2 :].reshape(-1, ndim), [5, 50, 95], axis=0)
    for name, m, a, b in zip(LABELS, med, lo, hi):
        print(f"{name:10s} {m:12.5f}  [{a:.5f}, {b:.5f}]")


if __name__ == "__main__":
    main()
