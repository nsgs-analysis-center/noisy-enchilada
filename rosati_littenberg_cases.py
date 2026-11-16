"""Case studies from Rosati & Littenberg (2024), arXiv:2410.17180, in this setup.

Run: uv run python rosati_littenberg_cases.py [case ...]

One year at 5 s on a WDM grid with ΔT = Nf·dt = 12 h, fitted in cells of 100
time pixels (50 days; the paper used twelve month-long STFTs). Synthetic data
are drawn at full time resolution with synthetic.simulate on the NOISE brick's
grid and orbit. Results go to paper_cases/.

The default cases are an injection recovery and an upper limit for each of the
paper's three templates, all with synthetic non-stationary confusion noise
analysed with its time dependence. Named extras drop the confusion noise
(-none), analyse it as stationary, or use the CD1-L bricks.

What follows the paper: the power-law injection (log10 A = -8.45,
alpha = 2/3); the shapes of the other two templates, which are the values its
upper limits are conditioned on; and instrument levels and prior ranges read
off its corner plots. The log-normal and phase-transition injection amplitudes
are ours: an order of magnitude in energy density above the paper's limits.

What differs: XYZ with the orbit's unequal arms rather than equal-arm AET; the
confusion modulation is a known table rather than a free amplitude per STFT;
the confusion spectrum is the tanh model, onto which the paper's injection
(Cornish & Robson's six-month fit) is mapped without its beta·f·sin(kappa·f)
term; and the CD1-L NOISE + GALFOR bricks stand in for the GLASS residuals,
with no Gaussianity veto.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from enchilada import L1Data, Wheel
from scipy.stats import gaussian_kde

from foreground_blocks import GalacticForeground
from foreground_demo import add_signal
from inst_noise_demo import TRIM_S, condition
from noise_blocks import StochasticNoiseBlock, UnequalArmEqualNoiseBlock
from noise_model import BAND_HZ
from sgwb_blocks import LogNormalSGWB, PhaseTransitionSGWB, PowerLawSGWB, SGWBBlock
from sgwb_demo import inject
from synthetic import simulate

HERE = Path(__file__).parent
NOISE_FILE = HERE / "NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5"
GALFOR_FILE = HERE / "GALFOR_731d_2.5s_L1.h5"
MODULATION_FILE = HERE / "modulation_unequal.dat"
OUTPUT = HERE / "paper_cases"

NF, NT, DT, COARSE_Q = 8640, 730, 5.0, 100  # ΔT = 12 h, one year, 50-day cells
WIDE_BAND_HZ = (1e-4, 8e-3)  # the paper conditions its narrow templates at 0.2 mHz
NODES = 8  # layers are 11.6 µHz wide: the spectra are smooth across one

# Instrument ASDs (m/sqrt(Hz), m/s²/sqrt(Hz)) read off the paper's corner plots.
LOUD = {"Soms_d": 1.5e-11, "Sa_a": 3e-15}  # power-law figures
QUIET = {"Soms_d": 7.0e-12, "Sa_a": 2.4e-15}  # log-normal and phase-transition figures

# Synthetic confusion noise: Cornish & Robson's six-month fit, A = 9e-45,
# alpha = 0.133, gamma = 917, f_k = 2.58 mHz (the injection lines of the
# paper's confusion-shape figure). A f^(-7/3) exp(-f^alpha) (1 + tanh(gamma (f_k - f)))
# is the tanh model with amp = 2A, f_1 = 1 Hz and f_2 = 1/gamma.
CONFUSION = {"galfor_amp": 1.8e-44, "galfor_fk": 2.58e-3, "galfor_alpha": 0.133,
             "galfor_f_1": 1.0, "galfor_f_2": 1 / 917}
CONFUSION_PRIORS = [[-47, -41], [-5, -2], [1e-3, 5.0], [-5, 1], [-5, -2]]  # f_1 reaches 1 Hz

# Template, prior box (read off the paper's corner-plot axes), injection, a
# quiet start for upper limits, and the paper's conditioning values for the
# quoted limit. The power law is Radler LDC1-6's.
TEMPLATES = {
    "powerlaw": dict(cls=PowerLawSGWB, priors=[[-14, -3], [-1, 3.3]],
                     injection=(-8.45, 2 / 3), quiet=(-12.5, 2 / 3),
                     amplitude=0, condition={1: 2 / 3}),
    "lognormal": dict(cls=LogNormalSGWB, priors=[[-8, -2], [-5, -2], [-1, 0.7]],
                      injection=(-2.5, np.log10(2e-4), np.log10(0.3)),
                      quiet=(-6.5, np.log10(2e-4), np.log10(0.3)),
                      amplitude=0, condition={1: np.log10(2e-4), 2: np.log10(0.3)}),
    "phase transition": dict(cls=PhaseTransitionSGWB,
                             priors=[[0.1, 0.7], [0.0, 0.7], [-14, -6], [-5, -2]],
                             injection=(0.15, 0.65, -10.4, np.log10(2e-4)),
                             quiet=(0.15, 0.65, -13.0, np.log10(2e-4)),
                             amplitude=2, condition={0: 0.15, 1: 0.65, 3: np.log10(2e-4)}),
}
PIVOT_HZ = 3e-3  # where log10 Omega is quoted for the power law

# name: (template, SGWB injected, confusion in the data, analysis, instrument, paper's limit)
CASES = {
    "det-powerlaw": ("powerlaw", True, "synthetic", "modulated", LOUD, None),
    "det-lognormal": ("lognormal", True, "synthetic", "modulated", QUIET, None),
    "det-pt": ("phase transition", True, "synthetic", "modulated", QUIET, None),
    "ul-powerlaw": ("powerlaw", False, "synthetic", "modulated", LOUD, -9.76),
    "ul-lognormal": ("lognormal", False, "synthetic", "modulated", QUIET, -3.00),
    "ul-pt": ("phase transition", False, "synthetic", "modulated", QUIET, -11.36),
    "det-powerlaw-none": ("powerlaw", True, None, None, LOUD, None),
    "det-powerlaw-stationary": ("powerlaw", True, "synthetic", "stationary", LOUD, None),
    "det-powerlaw-bricks": ("powerlaw", True, "bricks", "modulated", None, None),
    "ul-powerlaw-none": ("powerlaw", False, None, None, LOUD, -10.41),
    "ul-lognormal-none": ("lognormal", False, None, None, QUIET, -3.40),
    "ul-pt-none": ("phase transition", False, None, None, QUIET, -11.46),
}
DEFAULT_CASES = [name for name in CASES if name.count("-") == 1]


def observation(signal_file=None):
    """Conditioned WDM observation of the NOISE brick (plus a signal brick) and its delays."""
    raw = L1Data.from_cd1l(str(NOISE_FILE), duration_s=NF * NT * DT + 2 * TRIM_S)
    return condition(add_signal(raw, signal_file) if signal_file else raw, NF, NT, DT)


def conditional_upper_limit(samples, bounds, amplitude, condition, level=0.95):
    """Upper limit on one coordinate with others fixed, from a KDE of the samples."""
    grid = np.linspace(*bounds[amplitude], 2000)
    free = [amplitude] + [i for i in range(samples.shape[1]) if i not in condition
                          and i != amplitude]
    points = np.empty((samples.shape[1], len(grid)))
    points[amplitude] = grid
    for index, value in condition.items():
        points[index] = value
    if len(free) > 1:  # marginalize what is neither the amplitude nor conditioned
        raise ValueError("Every coordinate but the amplitude must be conditioned")
    cdf = np.cumsum(gaussian_kde(samples.T)(points))
    return float(grid[np.searchsorted(cdf, level * cdf[-1])])


def run_case(name, reference, bricks, delays, table, args):
    kind, injected, confusion, analysis, levels, paper = CASES[name]
    spec = TEMPLATES[kind]
    injection = spec["injection"]
    start = injection if injected else spec["quiet"]
    template = spec["cls"](priors=spec["priors"], fiducial=start)
    rng = np.random.default_rng(args.seed)

    def instrument(data, coarseQ, band_hz):
        block = UnequalArmEqualNoiseBlock(data, delays, coarseQ=coarseQ, band_hz=band_hz)
        if levels is not None:
            block.fiducial = np.array([levels["Soms_d"], levels["Sa_a"]])
        return block

    def foreground(modulated):
        return GalacticForeground(*(table if modulated else ()), priors=CONFUSION_PRIORS,
                                  fiducial=GalacticForeground().coordinates(
                                      np.array(list(CONFUSION.values()))))

    truth = None
    if confusion == "bricks":  # real noise and foreground; only the background is drawn
        band, components = BAND_HZ, [GalacticForeground(*table), template]
        helper = SGWBBlock(instrument(bricks, 1, band), PowerLawSGWB(), nodes=NODES)
        data = inject(bricks, helper, injection, rng)
    else:
        band = WIDE_BAND_HZ
        truth = dict(levels)
        parts = []
        if confusion:
            truth.update(CONFUSION)
            parts.append(foreground(True))
        if injected:
            truth.update(zip(template.parameter_names, injection))
            parts.append(template)
        fine = instrument(reference, 1, band)
        source = StochasticNoiseBlock(fine, parts, nodes=NODES) if parts else fine
        data = simulate(reference, [source.state(truth)], source.active_mask, rng=rng)
        components = ([foreground(analysis == "modulated")] if confusion else []) + [template]

    block = StochasticNoiseBlock(instrument(data, COARSE_Q, band), components, nodes=NODES,
                                 nwalkers=args.walkers, steps_per_call=args.steps)
    block.name = name
    wheel = Wheel(data, random_seed=args.seed)
    wheel.add(block, initial_block_state=block.initial_result(rng=rng), data_domain="wdm")
    chains = []
    for _ in range(args.cycles):
        wheel.run(1)
        chains.append(block.chain[::args.thin])
    kept = np.concatenate(chains[args.burn:])  # (steps, walkers, parameters)
    halves = np.array_split(kept, 2)
    drift = np.abs(halves[0].mean(axis=(0, 1)) - halves[1].mean(axis=(0, 1))) / kept.std(axis=(0, 1))
    samples = kept.reshape(-1, kept.shape[-1])
    names = block.parameter_names
    low, median, high = np.percentile(samples, [5, 50, 95], axis=0)
    result = {"case": name, "parameters": {
        n: {"median": float(m), "5%": float(a), "95%": float(b)}
        for n, a, m, b in zip(names, low, median, high)},
        "max_half_chain_drift_sigma": float(drift.max()), "samples": len(samples)}
    sgwb = samples[:, -len(template.parameter_names):]
    if truth is not None:
        row = block.coordinates(np.array([truth.get(n, np.nan) for n in names]))
        result["truth"] = {n: (None if np.isnan(v) else float(v)) for n, v in zip(names, row)}
    if kind == "powerlaw":
        pivot = sgwb[:, 0] + sgwb[:, 1] * np.log10(PIVOT_HZ / template.fref)
        result["log10_omega_3mHz"] = dict(zip(("5%", "median", "95%"),
                                              map(float, np.percentile(pivot, [5, 50, 95]))))
        result["log10_omega_3mHz"]["truth"] = (
            injection[0] + injection[1] * np.log10(PIVOT_HZ / template.fref) if injected else None)
    if not injected:
        result["upper_limit_95"] = conditional_upper_limit(
            sgwb, np.array(spec["priors"]), spec["amplitude"], spec["condition"])
        result["paper_upper_limit_95"] = paper
    OUTPUT.mkdir(exist_ok=True)
    np.savez_compressed(OUTPUT / f"{name}.npz", chain=kept, names=np.array(names))
    (OUTPUT / f"{name}.json").write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cases", nargs="*", choices=[*CASES, []], default=DEFAULT_CASES)
    parser.add_argument("--walkers", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000, help="Eryn steps per Wheel cycle")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--burn", type=int, default=4, help="cycles discarded")
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    reference, delays = observation()
    bricks = None
    if "det-powerlaw-bricks" in args.cases:
        bricks, _ = observation(GALFOR_FILE)
    table = np.loadtxt(MODULATION_FILE)
    table = (table[:, 0], table[:, 1:])

    for name in args.cases:
        result = run_case(name, reference, bricks, delays, table, args)
        print(json.dumps(result), flush=True)
