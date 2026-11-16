"""Choosing the WDM tiling and the coarse-graining Q for the template cases.

Run: uv run python q_scan.py        (about 2 minutes; ~6 GB peak footprint)

Applies the criteria of the WDM paper draft (../wdm-paper: draft2 Secs. "Optimal
choice of WDM tiling" and "Welch-like coarse-graining", tiling_heuristic.md,
ragged_coarse_graining.md, ws_wishart_route1.py) to rosati_littenberg_cases.py's
setup: one year, ΔT = 12 h, 0.1-8 mHz, the truth models of its three detection
cases. Everything is exact for the model, given independent WDM pixels (how
synthetic.simulate draws): no sampling.

For each Q, with cells of Q retained columns per layer:

* information: the Godambe information G = H J^-1 H of the coarse likelihood,
  where H is its expected Hessian and J the variance of its score under the
  true (fine) covariances. sigma_true = sqrt(diag G^-1) is the actual scatter
  of the estimate; D = n - tr(F_fine^-1 G) the parameter directions' worth of
  information lost relative to the pixel likelihood (ragged note, Sec. 2.1).
* coverage: sigma_post = sqrt(diag H^-1) is the width the coarse posterior
  reports. sigma_post / sigma_true < 1 means intervals that are too narrow.
  Computed for cell weights nu = cell size (what the blocks use) and for the
  Route-1 Welch-Satterthwaite nu_eff = size * d / <tr((Cbar^-1 C_n)^2)>.
* leakage: Delta_dep per channel from Result R3 for time-only blocks,
  sum over fine cells of [C_f^4 eps_f^4 + Q^2 C_t^4 eps_t^4] / (2 Q^3)
  (valid once Q exceeds the ~8-cell reach of the time profile).
* cost: likelihood terms, layers x cells.
"""

import argparse

import numpy as np

import rosati_littenberg_cases as cases
from foreground_blocks import GalacticForeground
from noise_blocks import StochasticNoiseBlock, UnequalArmEqualNoiseBlock
from noise_model import UI, UJ

C_F, C_T = 0.208, 0.639  # Meyer window constants (draft2 Eq. error-per-pixel)
QS = (1, 2, 4, 8, 16, 32, 50, 100, 181, 362, 724)
FD_STEP = 1e-5  # central-difference step in sampling coordinates


def unpack(packed):
    """(..., 6) XX YY ZZ XY XZ YZ -> (..., 3, 3)."""
    out = np.empty(packed.shape[:-1] + (3, 3))
    for k, (i, j) in enumerate(zip(UI, UJ)):
        out[..., i, j] = out[..., j, i] = packed[..., k]
    return out


def model(name, reference, delays, table):
    """Fine (Q = 1) block at a detection case's truth; its covariance and derivatives."""
    kind, _, _, _, levels, _ = cases.CASES[name]
    spec = cases.TEMPLATES[kind]
    template = spec["cls"](priors=spec["priors"], fiducial=spec["injection"])
    foreground = GalacticForeground(*table, priors=cases.CONFUSION_PRIORS)
    instrument = UnequalArmEqualNoiseBlock(reference, delays, band_hz=cases.WIDE_BAND_HZ)
    block = StochasticNoiseBlock(instrument, [foreground, template], nodes=cases.NODES)
    truth = {**levels, **cases.CONFUSION, **dict(zip(template.parameter_names, spec["injection"]))}
    row = block.coordinates(np.array([truth[n] for n in block.parameter_names]))
    layers, cols = len(block.region.layers), block.region.num_cells
    bases = block.bases.reshape(layers, cols, -1, 6)

    def covariance(levels):
        return unpack(np.einsum("lnck,lc->lnk", bases, levels))

    C = covariance(block.levels(row)[0])
    dC = []
    for k in range(len(row)):
        step = np.zeros_like(row)
        step[k] = FD_STEP
        dlevels = (block.levels(row + step)[0] - block.levels(row - step)[0]) / (2 * FD_STEP)
        dC.append(covariance(dlevels))
    return block, C, np.stack(dC)


def trace_products(X, Y, weight):
    """sum over pixels of weight * tr(X_a Y_b) for stacks X, Y of shape (K, ..., 3, 3)."""
    K = X.shape[0]
    Xw = (X * weight[None, ..., None, None]).reshape(K, -1)
    return Xw @ np.swapaxes(Y, -1, -2).reshape(K, -1).T


def scan(C, dC):
    """Fisher of the pixel likelihood, and per Q the coarse H, J for both weightings."""
    Cinv = np.linalg.inv(C)
    M = Cinv[None] @ dC
    F = 0.5 * trace_products(M, M, np.ones(C.shape[:2]))
    out = {}
    cols = C.shape[1]
    for Q in QS:
        starts = np.arange(0, cols, Q)
        sizes = np.diff(np.append(starts, cols)).astype(float)
        cell_of = np.repeat(np.arange(len(starts)), sizes.astype(int))
        Cbar = np.add.reduceat(C, starts, axis=1) / sizes[None, :, None, None]
        dCbar = np.add.reduceat(dC, starts, axis=2) / sizes[None, None, :, None, None]
        Cbinv = np.linalg.inv(Cbar)
        G = Cbinv[None] @ dCbar
        A = G @ Cbinv[None]  # Cbar^-1 dCbar_a Cbar^-1
        B = A[:, :, cell_of] @ C[None]  # per fine pixel: A_a(cell) C_n
        ratio = Cbinv[:, cell_of] @ C  # Cbar^-1 C_n
        trM2 = np.einsum("lnij,lnji->ln", ratio, ratio)
        nu_ws = sizes * 3 / (np.add.reduceat(trM2, starts, axis=1) / sizes)
        result = {}
        for label, nu in (("size", np.broadcast_to(sizes, nu_ws.shape)), ("WS", nu_ws)):
            H = 0.5 * trace_products(G, G, nu)
            pixel_weight = (nu**2 / (2 * sizes**2))[:, cell_of]
            J = trace_products(B, B, pixel_weight)
            result[label] = (H, J)
        out[Q] = dict(result, cells=len(starts), nu_ws_ratio=float(np.median(nu_ws / sizes)),
                      nu_ws_min=float((nu_ws / sizes).min()))
    return F, out


def epsilons(C):
    """eps_f, eps_t of ln S_XX on the fine grid (central differences, edges dropped)."""
    lnS = np.log(C[..., 0, 0])
    eps_f = 0.5 * (lnS[2:, 1:-1] - lnS[:-2, 1:-1])
    eps_t = 0.5 * (lnS[1:-1, 2:] - lnS[1:-1, :-2])
    return eps_f, eps_t


def tiling(block, C):
    """draft2 Eq. rms_optimal_tiling and rms_optimal_loss on the fitted region."""
    eps_f, eps_t = epsilons(C)
    dT = cases.NF * cases.DT
    dF = 1 / (2 * dT)
    T_obs = cases.NT * dT
    df_rms = np.sqrt(np.mean((eps_f / dF) ** 2))
    dt_rms = np.sqrt(np.mean((eps_t / dT) ** 2))
    nt_star = T_obs * np.sqrt(2 * C_T * dt_rms / (C_F * df_rms))
    loss_star = C_F * C_T / 2 * df_rms * dt_rms * eps_f.size
    loss_now = 0.5 * np.sum(C_F**2 * eps_f**2 + C_T**2 * eps_t**2)
    f = (block.region.layers[1:-1] / (2 * dT))
    bands = []
    for lo, hi in ((1e-4, 3e-4), (3e-4, 1e-3), (1e-3, 3e-3), (3e-3, 8e-3)):
        keep = (f >= lo) & (f < hi)
        a = np.sqrt(np.mean((eps_f[keep] / dF) ** 2))
        b = np.sqrt(np.mean((eps_t[keep] / dT) ** 2))
        bands.append((lo, hi, np.sqrt(C_F / (2 * C_T) * a / b) / 3600, np.median(np.abs(eps_f[keep])),
                      np.median(np.abs(eps_t[keep]))))
    return T_obs / nt_star / 3600, loss_now, loss_star, bands, eps_f, eps_t


def leakage(eps_f, eps_t, Q):
    if Q == 1:  # pixel Whittle over w (draft2 Eq. error-per-pixel), per channel
        return 0.5 * np.sum(C_F**2 * eps_f**2 + C_T**2 * eps_t**2)
    return np.sum(C_F**4 * eps_f**4 + Q**2 * C_T**4 * eps_t**4) / (2 * Q**3)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cases", nargs="*", default=["det-powerlaw", "det-lognormal", "det-pt"])
    args = parser.parse_args()
    reference, delays = cases.observation()
    table = np.loadtxt(cases.MODULATION_FILE)
    table = (table[:, 0], table[:, 1:])
    for name in args.cases:
        block, C, dC = model(name, reference, delays, table)
        names = block.parameter_names
        dT_star, loss_now, loss_star, bands, eps_f, eps_t = tiling(block, C)
        print(f"\n=== {name}: {len(block.region.layers)} layers x {C.shape[1]} columns")
        print(f"tiling: ΔT* = {dT_star:.1f} h (now 12 h); diagonal-approximation KL per channel "
              f"{loss_now:.0f} nats now, {loss_star:.0f} at ΔT*")
        for lo, hi, local, ef, et in bands:
            print(f"   {lo*1e3:4.1f}-{hi*1e3:4.1f} mHz: local ΔT* {local:8.1f} h, "
                  f"median |eps_f| {ef:.3f}, |eps_t| {et:.1e}")
        F, out = scan(C, dC)
        Finv = np.linalg.inv(F)
        sigma_fine = np.sqrt(np.diag(Finv))
        shown = [k for k, n in enumerate(names)
                 if n in ("Soms_d", "Sa_a", "galfor_amp", "galfor_fk")
                 or n.endswith(("log10_A", "log10_Ap", "log10_fstar", "log10_fp"))]
        print("sigma_true / sigma_fine (information), and sigma_post / sigma_true (coverage) "
              "for weights = size | WS:")
        print(f"{'Q':>5} {'terms':>7} {'D lost':>7} {'leak':>9} {'nu_WS/Q':>8}  "
              + "  ".join(f"{names[k][:14]:>23}" for k in shown))
        for Q, r in out.items():
            H, J = r["size"]
            Hi = np.linalg.inv(H)
            Ginv = Hi @ J @ Hi
            D = len(names) - np.trace(Finv @ np.linalg.inv(Ginv))
            sigma_true = np.sqrt(np.diag(Ginv))
            Hw, Jw = r["WS"]
            Hwi = np.linalg.inv(Hw)
            sigma_true_ws = np.sqrt(np.diag(Hwi @ Jw @ Hwi))
            cells = [f"{sigma_true[k] / sigma_fine[k]:5.3f} {np.sqrt(Hi[k, k]) / sigma_true[k]:5.3f}"
                     f"|{np.sqrt(Hwi[k, k]) / sigma_true_ws[k]:5.3f}" for k in shown]
            print(f"{Q:5d} {len(block.region.layers) * r['cells']:7d} {D:7.3f} "
                  f"{leakage(eps_f, eps_t, Q):9.2e} {r['nu_ws_ratio']:8.3f}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
