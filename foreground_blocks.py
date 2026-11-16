"""Galactic confusion foreground: the modulated hyperbolic-tangent model.

The strain spectrum is lisatools' HyperbolicTangentGalacticForeground,

    S(f) = amp/2 · f^(-7/3) · exp(-(f/f_1)^alpha) · (1 + tanh((fk - f)/f_2)),

seen through the equal-arm TDI-2 response. A table of the six XYZ covariance
elements against time carries the annual modulation and the channel
correlations, as in noise_sampler_minimal.py's foreground mode; without a
table the foreground is stationary and isotropic.

Wheel keeps one covariance owner, so the block samples the foreground together
with an instrument model:

    foreground = GalacticForeground.from_file("modulation_unequal.dat")
    block = GalacticForegroundBlock(UnequalArmEqualNoiseBlock(data, delays), foreground)
"""

import numpy as np

from noise_blocks import StochasticComponent, StochasticNoiseBlock

LOG10 = np.array([True, True, False, True, True])  # sampled in log10: all but alpha
FIDUCIAL = (3.26651613e-44, 2.09278117e-03, 1.18300266, 1.14556409e-03, 3.38095297e-04)


class GalacticForeground(StochasticComponent):
    """Hyperbolic-tangent spectrum with an optional tabulated time modulation.

    Coordinates are log10 amp, log10 fk, alpha, log10 f_1, log10 f_2 (Hz), with
    lisatools' prior support (GALFOR_PRIOR_RANGE); the published values are
    physical. ``modulation`` holds XX, YY, ZZ, XY, XZ, YZ at
    ``modulation_times_gps``, shape (times, 6), and is interpolated linearly to
    WDM column centres, which the table must cover.
    """

    parameter_names = ("galfor_amp", "galfor_fk", "galfor_alpha", "galfor_f_1", "galfor_f_2")
    priors = np.array([[-47, -41], [-5, -2], [1e-3, 5.0], [-5, -2], [-5, -2]])
    fiducial = np.where(LOG10, np.log10(FIDUCIAL), FIDUCIAL)

    def __init__(self, modulation_times_gps=None, modulation=None, **kwargs):
        super().__init__(**kwargs)
        if (modulation_times_gps is None) != (modulation is None):
            raise ValueError("Pass modulation times and values together, or neither")
        if modulation is not None:
            modulation_times_gps = np.asarray(modulation_times_gps, dtype=float)
            modulation = np.asarray(modulation, dtype=float)
            if (modulation.shape != (len(modulation_times_gps), 6)
                    or not np.all(np.diff(modulation_times_gps) > 0)
                    or not np.isfinite(modulation).all()):
                raise ValueError("Modulation requires increasing times and finite "
                                 "(times, 6) XX YY ZZ XY XZ YZ values")
        self.modulation_times_gps, self.modulation = modulation_times_gps, modulation

    @classmethod
    def from_file(cls, path, **kwargs):
        """Read a text table of time, XX, YY, ZZ, XY, XZ, YZ (rows or columns)."""
        table = np.loadtxt(path)
        table = table if table.shape[1] == 7 else table.T
        return cls(table[:, 0], table[:, 1:], **kwargs)

    def physical(self, coords):
        return np.where(LOG10, 10.0 ** np.asarray(coords), coords)

    def coordinates(self, values):
        with np.errstate(divide="ignore", invalid="ignore"):  # the block rejects non-finite rows
            return np.where(LOG10, np.log10(values), values)

    def psd(self, f, coords):
        amp, fk, alpha, f_1, f_2 = self.physical(coords).T[:, :, None, None]
        return (0.5 * amp * f ** (-7 / 3) * np.exp(-((f / f_1) ** alpha))
                * (1 + np.tanh((fk - f) / f_2)))

    def pattern(self, times_gps):
        if self.modulation is None:
            return super().pattern(times_gps)
        t = self.modulation_times_gps
        if times_gps[0] < t[0] or times_gps[-1] > t[-1]:
            raise ValueError("Modulation table must cover the WDM columns")
        return np.array([np.interp(times_gps, t, element) for element in self.modulation.T])


class GalacticForegroundBlock(StochasticNoiseBlock):
    """An instrument model with the galactic foreground, sampled jointly.

    With ``UnequalArmEqualNoiseBlock`` and a modulation table this is the
    standalone sampler's foreground mode: two instrument log ASDs, then the
    five foreground coordinates. ``foreground`` defaults to the stationary,
    isotropic model.
    """

    name = "GalacticForegroundBlock"

    def __init__(self, instrument, foreground=None, **kwargs):
        super().__init__(instrument, [foreground or GalacticForeground()], **kwargs)
