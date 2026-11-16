"""Foreground and SGWB blocks: spectra, WDM fold, joint likelihood and Wheel lifecycle."""
import unittest

import numpy as np
import wdm
from enchilada import WDMGrid, Wheel

from foreground_blocks import GalacticForeground, GalacticForegroundBlock
from noise_blocks import (
    ISOTROPIC, REJECTED, StochasticNoiseBlock, UnequalArmEqualNoiseBlock, UnequalArmUnequalNoiseBlock,
)
from noise_model import (
    FIDUCIAL, TILE, UI, UJ, _loglike_parallel, _loglike_serial, layer_nodes, noise_loglike,
    stochastic_response,
)
from sgwb_blocks import LogNormalSGWB, PhaseTransitionSGWB, PowerLawSGWB, SGWBBlock
from test_noise_model import campaign, matrix, reference_score

# lisatools.stochastic *.specific_Sh_function and X2TDISens.stochastic_transform
# (LISAanalysistools, 2026-11-15) at these frequencies and coordinates.
F = np.array([3e-4, 1e-3, 3e-3, 8e-3])
LISATOOLS = (
    (GalacticForeground, GalacticForeground.fiducial,
     [4.416954849252959e-36, 1.3919094580551141e-37, 5.147970018591869e-42,
      7.981588089053346e-59]),
    (PowerLawSGWB, [-8.45, 2 / 3],
     [2.6936867988303967e-38, 1.6229184303045848e-39, 1.2502986566701315e-40,
      1.2679050236754566e-41]),
    (LogNormalSGWB, [-2.5, np.log10(2e-3), 0.0],
     [8.342154545804307e-38, 9.228786910153784e-39, 4.274650512586766e-40,
      4.8221686330714766e-42]),
    (LogNormalSGWB, [-3.0, -2.7, -0.5],
     [1.778209033041081e-38, 1.9716549856047225e-39, 2.071345779850621e-40,
      6.530004655693558e-48]),
    (LogNormalSGWB, [-2.5, -3.0, 0.98],  # D >= 9: asymptote
     [3.703212200196991e-39, 9.99867294053187e-41, 3.703212200196989e-42,
      1.952865808697631e-43]),
    (PhaseTransitionSGWB, [1.0, 1.0, -18.0, -2.5],
     [4.692103584890884e-48, 1.5334652604177796e-47, 1.4324543623383068e-47,
      7.107808481418032e-50]),
    (PhaseTransitionSGWB, [0.4, 2.5, -10.0, -2.9],
     [5.819567925177801e-39, 3.527080285393465e-38, 4.0765665447308595e-40,
      7.922427640746379e-43]),
)
RESPONSE = [2.412479917423255e-10, 3.295545067423389e-07, 0.0002315835946679004,
            0.06436216598872062]
# Coordinates loud enough to matter beside the fiducial instrument noise.
LOUD = {GalacticForeground: [-40.5, -2.4, 1.2, -2.6, -3.1], PowerLawSGWB: [-6.0, 0.5],
        LogNormalSGWB: [-1.2, -2.6, -0.2], PhaseTransitionSGWB: [1.0, 3.0, -7.5, -2.6]}


def modulation_table(data):
    """A smooth, anisotropic, positive semidefinite XYZ pattern across the observation."""
    t = data.t0 + np.linspace(-1, data.Tobs + 1, 9)
    phase = 2 * np.pi * (t - t[0]) / (t[-1] - t[0])
    isotropic = np.full((3, 3), -0.5) + 1.5 * np.eye(3)
    v, w = np.array([1.0, -1.0, 0.0]), np.array([0.5, 0.5, -1.0])
    matrices = ((1 + 0.3 * np.sin(phase))[:, None, None] * isotropic
                + (0.2 * (1 + np.cos(phase)))[:, None, None] * np.outer(v, v)
                + (0.3 * (1 + np.sin(2 * phase)))[:, None, None] * np.outer(w, w))
    return t, matrices[:, UI, UJ]


def blocks(data, delays, **kwargs):
    """One block per component, and one block holding all four."""
    def instrument():
        return UnequalArmEqualNoiseBlock(data, delays, **kwargs)
    foreground = GalacticForeground(*modulation_table(data))
    templates = [PowerLawSGWB(), LogNormalSGWB(), PhaseTransitionSGWB()]
    return ([GalacticForegroundBlock(instrument(), foreground)]
            + [SGWBBlock(instrument(), template) for template in templates]
            + [StochasticNoiseBlock(instrument(), [foreground, *templates])])


def loud(block):
    return np.concatenate([np.log(FIDUCIAL), *(LOUD[type(c)] for c in block.components)])


class StochasticTests(unittest.TestCase):
    def test_spectra_reproduce_lisatools(self):
        np.testing.assert_allclose(stochastic_response(F), RESPONSE, rtol=1e-13)
        for template, coords, expected in LISATOOLS:
            # Two rows and two layers exercise the walker and node broadcasting.
            actual = template().psd(np.stack([F, F[::-1]]), np.array([coords, coords]))
            self.assertEqual(actual.shape, (2, 2, 4))
            np.testing.assert_allclose(actual[1, 0], expected, rtol=1e-13)
            np.testing.assert_allclose(actual[0, 1], expected[::-1], rtol=1e-13)
        self.assertAlmostEqual(PowerLawSGWB(fref=3e-3).psd(np.full((1, 1), 3e-3),
                                                        np.array([[-11.0, 1.7]]))[0, 0, 0]
                               / PowerLawSGWB().psd(np.full((1, 1), 3e-3),
                                                    np.array([[-11.0, 0.0]]))[0, 0, 0], 1.0)

    def test_nodes_fold_like_wdm(self):
        component = GalacticForeground()

        def psd(f):
            f = np.maximum(f, 1e-10)
            return component.psd(f[None], component.fiducial[None])[0] * stochastic_response(f)

        # 25 filter bins are their own nodes (exact); 385 are merged into 64.
        for nt, rtol in ((32, 2e-12), (512, 1e-3)):
            grid = WDMGrid(num_frequency_divisions=128, num_time_divisions=nt)
            layers = np.arange(1, 10)
            f, weights = layer_nodes(layers, grid, 5.0)
            self.assertEqual(f.shape, (9, min(64, 3 * nt // 4 + 1)))
            np.testing.assert_allclose(weights.sum(axis=-1), 0.5, rtol=1e-13)
            expected = wdm.band_averaged_variance(128, nt, 5.0, psd)[layers, 0]
            np.testing.assert_allclose(np.sum(weights * psd(f), axis=-1), expected, rtol=rtol)

    def test_layer_levels_in_kernel(self):
        rng = np.random.default_rng(6)
        layers, cells = 7, 3 * TILE // 4 + 5  # tiles straddle layers
        B = np.abs(rng.normal(size=(layers * cells, 4, 6)))
        B[:, :, :3] += 10
        X, weights = np.abs(rng.normal(size=(len(B), 6))), rng.uniform(1, 4, len(B))
        levels = np.abs(rng.normal(size=(5, layers, 4)))
        actual = _loglike_serial(levels, B, X, weights)
        np.testing.assert_array_equal(actual, _loglike_parallel(levels, B, X, weights))
        by_layer = sum(noise_loglike(levels[:, n], *(a[n * cells:(n + 1) * cells]
                                                     for a in (B, X, weights)))
                       for n in range(layers))
        np.testing.assert_allclose(actual, by_layer, rtol=1e-13)
        np.testing.assert_array_equal(  # constant levels need no layer axis
            noise_loglike(levels[:, 0], B, X, weights),
            noise_loglike(np.repeat(levels[:, :1], layers, axis=1), B, X, weights))
        with self.assertRaisesRegex(ValueError, "divide evenly"):
            noise_loglike(levels[:, :4], B, X, weights)

    def test_covariance_is_instrument_plus_folded_spectrum(self):
        data, instrument, _, delays = campaign()
        grid, layers = data.wdm_grid, instrument.region.layers
        nf, nt = grid.num_frequency_divisions, grid.num_time_divisions
        times = data.t0 + np.arange(nt) * nf * data.dt
        for block in blocks(data, delays)[:4]:
            component, row = block.components[0], loud(block)

            def psd(f, component=component, row=row):
                f = np.maximum(f, 1e-10)
                return component.psd(f[None], row[None, 2:])[0] * stochastic_response(f)

            folded = wdm.band_averaged_variance(nf, nt, data.dt, psd)[layers, 0]
            expected = np.zeros((*grid.array_shape, 3, 3))
            pattern = component.pattern(times)
            for k, (i, j) in enumerate(zip(UI, UJ)):
                expected[layers, :, i, j] = expected[layers, :, j, i] = folded[:, None] * pattern[k]
            expected[~block.active_mask] = 0
            actual = matrix(block.covariance(row)) - matrix(instrument.covariance(row[:2]))
            scale = np.abs(expected).max()
            self.assertGreater(scale, 1e-3 * np.abs(matrix(instrument.covariance(row[:2]))
                                                    [block.active_mask]).max())
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12 * scale)
        stationary = GalacticForegroundBlock(instrument).components[0]
        np.testing.assert_array_equal(stationary.pattern(times),
                                      np.repeat(ISOTROPIC[:, None], nt, axis=1))

    def test_likelihood_matches_published_covariance(self):
        data, _, _, delays = campaign()
        delays *= np.linspace(0.98, 1.02, len(delays))[:, None]
        for q in (1, 5):
            for block in blocks(data, delays, coarseQ=q):
                rows = np.stack([block.center, loud(block)])
                expected = [-0.5 * reference_score(block.covariance(row), data, block.active_mask)
                            for row in rows]
                np.testing.assert_allclose(block.log_likelihood(data)(rows), expected, rtol=2e-12)

    def test_wheel_lifecycle_and_published_parameters(self):
        data, _, _, delays = campaign()
        names = ["GalacticForegroundBlock", "PowerLawSGWBBlock", "LogNormalSGWBBlock",
                 "PhaseTransitionSGWBBlock", "StochasticNoiseBlock"]
        for block, name in zip(blocks(data, delays, coarseQ=5), names):
            self.assertEqual(block.name, name)
            block.steps_per_call = 3
            initial = block.initial_result(rng=np.random.default_rng(1))
            wheel = Wheel(data, random_seed=2)
            wheel.add(block, initial_block_state=initial, data_domain="wdm")
            wheel.run(2)
            self.assertEqual(block.sampler_state["iterations"], 6)
            self.assertEqual(block.chain.shape, (3, block.nwalkers, len(block.parameter_names)))
            np.testing.assert_array_equal(block.chain[-1], block.sampler_state["coords"][0, :, 0])
            self.assertFalse(np.array_equal(block.sampler_state["coords"][0, :, 0],
                                            np.tile(block.center, (block.nwalkers, 1))))
            parameters = wheel.ledger[block.name].parameters
            self.assertEqual(tuple(parameters), block.parameter_names)
            row = block.sampler_state["coords"][0, 0, 0]
            self.assertEqual(parameters["Soms_d"], np.exp(row[0]))
            if name == "GalacticForegroundBlock":  # published physical, sampled in log10
                self.assertEqual(parameters["galfor_amp"], 10 ** row[2])
                self.assertEqual(parameters["galfor_alpha"], row[4])
            if name == "PowerLawSGWBBlock":
                self.assertEqual(parameters["powerlaw_log10_A"], row[2])
            prior = block.make_prior()
            self.assertTrue(np.isfinite(prior.logpdf(block.sampler_state["coords"])).all())

    def test_prior_and_invalid_corners(self):
        data, _, _, delays = campaign()
        block = SGWBBlock(UnequalArmEqualNoiseBlock(data, delays), PhaseTransitionSGWB())
        prior = block.make_prior()
        inside, outside, bad_level = block.center, block.center.copy(), block.center.copy()
        outside[2], bad_level[0] = 11.0, 0.0
        np.testing.assert_array_equal(np.isfinite(prior.logpdf([inside, outside, bad_level])),
                                      [True, False, False])
        # rb = 10, b = 1: m > b + 4, so the spectrum is undefined below the peak.
        corner = np.concatenate([np.log(FIDUCIAL), [10.0, 1.0, -7.5, -2.0]])
        self.assertTrue(np.isfinite(prior.logpdf(corner)))
        self.assertEqual(block.log_likelihood(data)(corner)[0], REJECTED)
        # The instrument's prior is kept, including the soft OMS ratio term.
        links = UnequalArmUnequalNoiseBlock(data, delays, oms_log_power_ratio_sigma=0.5)
        joint = SGWBBlock(links, PowerLawSGWB())
        self.assertEqual(len(joint.parameter_names), 14)
        row = joint.center.copy()
        row[0] += 0.3
        np.testing.assert_allclose(joint.make_prior().logpdf(row),
                                   links.make_prior().logpdf(row[:12]))
        self.assertLess(joint.make_prior().logpdf(row), joint.make_prior().logpdf(joint.center))

    def test_configuration_errors(self):
        data, instrument, _, _ = campaign()
        t, table = modulation_table(data)
        with self.assertRaisesRegex(ValueError, "cover"):
            GalacticForegroundBlock(instrument, GalacticForeground(t[:5], table[:5]))
        with self.assertRaisesRegex(ValueError, "together"):
            GalacticForeground(t)
        with self.assertRaisesRegex(ValueError, "increasing"):
            GalacticForeground(t[::-1], table)
        with self.assertRaisesRegex(ValueError, "unique"):
            StochasticNoiseBlock(instrument, [PowerLawSGWB(), PowerLawSGWB()])
        with self.assertRaisesRegex(ValueError, "at least one"):
            StochasticNoiseBlock(instrument, [])
        with self.assertRaisesRegex(ValueError, "14 walkers"):
            GalacticForegroundBlock(instrument, nwalkers=12)
        with self.assertRaisesRegex(ValueError, "inside"):
            PowerLawSGWB(priors=[[-12, -9], [-2, 2]])
        with self.assertRaisesRegex(ValueError, "2 prior bounds"):
            PowerLawSGWB(priors=[[-12, -9]])
        # A wider band adds layers; fewer quadrature nodes barely move a smooth fold.
        wide = UnequalArmEqualNoiseBlock(data, np.full((32, 6), 8.3), band_hz=(3e-4, 2e-2))
        self.assertGreater(len(wide.region.layers), len(instrument.region.layers))
        coarse, fine = (SGWBBlock(wide, PowerLawSGWB(), nodes=n) for n in (5, 64))
        self.assertEqual(coarse.node_frequencies.shape[1], 5)
        np.testing.assert_allclose(coarse.levels(coarse.center), fine.levels(fine.center),
                                   rtol=0.05)
        custom = PowerLawSGWB(priors=[[-12, -8], [-1, 1]], fiducial=[-10, 0])
        np.testing.assert_array_equal(SGWBBlock(instrument, custom).center[2:], [-10, 0])
        np.testing.assert_array_equal(PowerLawSGWB.priors, [[-22, -7], [-2, 2]])


if __name__ == "__main__":
    unittest.main()
