"""Stochastic gravitational-wave backgrounds: lisatools' spectral templates.

Each template is a strain spectrum S(f) = 3 H0² Ω(f) / (4 π² f³) with
H0 = 70 km/s/Mpc, seen through the equal-arm TDI-2 response as a stationary,
isotropic background (XY = XZ = YZ = -XX/2):

    PowerLawSGWB           log10_A, alpha: Ω = A (f / fref)^alpha, fref = 25 Hz
    LogNormalSGWB          log10_A, log10_fstar, log10_D: scalar-induced
                           (Pi & Sasaki 2020, arXiv:2005.12306)
    PhaseTransitionSGWB    rb, b, log10_Ap, log10_fp: double broken power law
                           (arXiv:2209.13277)

The spectra are ported from lisatools.stochastic and sampled in lisatools'
own parameters. Priors and fiducials default to GLASS's
(glass_noise_model.h default priors, default_sgwb_injection).

Wheel keeps one covariance owner, so the block samples a template together
with an instrument model:

    block = SGWBBlock(UnequalArmEqualNoiseBlock(data, delays), PowerLawSGWB())
"""

import numpy as np
from scipy.special import erfc

from noise_blocks import StochasticComponent, StochasticNoiseBlock

H0_SI = 70.0 * 3.24078e-20  # 1/s
HSCALE = 3 * H0_SI**2 / (4 * np.pi**2)  # Ω(f) / f³ to strain PSD
CGOR0 = 1.6e-5 / 0.7**2  # c_g Ω_r,0 for h = 0.7


class PowerLawSGWB(StochasticComponent):
    """Ω(f) = 10^log10_A (f / fref)^alpha; fref defaults to lisatools' 25 Hz."""

    parameter_names = ("powerlaw_log10_A", "powerlaw_alpha")
    priors = np.array([[-22.0, -7.0], [-2.0, 2.0]])
    fiducial = np.array([-8.45, 2 / 3])

    def __init__(self, *, fref=25.0, **kwargs):
        super().__init__(**kwargs)
        self.fref = fref

    def psd(self, f, coords):
        log10_A, alpha = coords.T[:, :, None, None]
        return HSCALE / f**3 * 10.0**log10_A * (f / self.fref) ** alpha


class LogNormalSGWB(StochasticComponent):
    """Scalar-induced background of a log-normal curvature peak of width D.

    Pi & Sasaki eq. (3.29); for D >= 9 the closed form cancels
    catastrophically and lisatools uses the asymptote of eq. (3.33).
    """

    parameter_names = ("lognormal_log10_A", "lognormal_log10_fstar", "lognormal_log10_D")
    priors = np.array([[-5.0, 0.0], [-5.0, 0.0], [-2.0, 1.0]])
    fiducial = np.array([-2.5, np.log10(2e-3), 0.0])

    def psd(self, f, coords):
        log10_A, log10_fstar, log10_D = coords.T[:, :, None, None]
        D = 10.0**log10_D
        D2 = D * D
        ft = f / 10.0**log10_fstar
        logft = np.log(ft)
        logK = logft + 1.5 * D2
        half_log32 = 0.5 * np.log(1.5)
        sqrtpi = np.sqrt(np.pi)
        with np.errstate(over="ignore", invalid="ignore"):  # D >= 9 rows are replaced below
            t1 = (4 / 5 / sqrtpi * ft**3 * np.exp(9 * D2 / 4) / D
                  * ((logK**2 + 0.5 * D2) * erfc((logK + half_log32) / D)
                     - D / sqrtpi * np.exp(-((logK + half_log32) ** 2) / D2)
                     * (logK - half_log32)))
            t2 = (0.0659 / D2 * ft**2 * np.exp(D2)
                  * np.exp(-((logft + D2 - 0.5 * np.log(4 / 3)) ** 2) / D2))
            t3 = (np.sqrt(2) / 3 / sqrtpi * ft ** (-4) * np.exp(8 * D2) / D
                  * np.exp(-(logft**2) / (2 * D2))
                  * erfc((4 * D2 - logft + np.log(4)) / (D * np.sqrt(2))))
            shape = np.where(D < 9, t1 + t2 + t3, 0.783e-3)
        return HSCALE / f**3 * CGOR0 * 10.0 ** (2 * log10_A) * shape


class PhaseTransitionSGWB(StochasticComponent):
    """Double broken power law peaking at fp with amplitude Ap.

    The spectrum is undefined where b + 4 - m + m s² <= 0 (s = f / fp, m the
    rb-weighted slope), which happens below the peak when rb⁴ (5 - b) > 4; the
    block rejects those corners of the prior. GLASS's default injection, the
    fiducial here, sits on that boundary and far below the instrument noise.
    """

    parameter_names = ("phase_transition_rb", "phase_transition_b",
                       "phase_transition_log10_Ap", "phase_transition_log10_fp")
    priors = np.array([[0.1, 10.0], [0.0, 8.0], [-22.0, -4.0], [-5.0, 0.0]])
    fiducial = np.array([1.0, 1.0, -18.0, -2.5])

    def psd(self, f, coords):
        rb, b, log10_Ap, log10_fp = coords.T[:, :, None, None]
        rb4 = rb**4
        m = (9 * rb4 + b) / (rb4 + 1)
        s = f / 10.0**log10_fp
        shape = (s**9 * ((1 + rb4) / (rb4 + s**4)) ** ((9 - b) / 4)
                 * ((b + 4) / (b + 4 - m + m * s * s)) ** ((b + 4) / 2))
        return HSCALE / f**3 * 10.0**log10_Ap * shape


class SGWBBlock(StochasticNoiseBlock):
    """An instrument model with one SGWB template, sampled jointly.

    The ledger name is the template's class name + 'Block', e.g.
    ``PowerLawSGWBBlock``.
    """

    def __init__(self, instrument, template, **kwargs):
        self.name = f"{type(template).__name__}Block"
        super().__init__(instrument, [template], **kwargs)
