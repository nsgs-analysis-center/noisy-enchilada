"""Synthetic observations: block states at named parameters and draws from them."""
import unittest

import numpy as np
from enchilada import BlockState, Wheel, transform

from foreground_blocks import GalacticForeground
from noise_blocks import StochasticNoiseBlock, UnequalArmEqualNoiseBlock
from noise_model import CHANNELS
from sgwb_blocks import PowerLawSGWB, SGWBBlock
from synthetic import blank_observation, simulate
from test_noise_model import matrix

# Each term dominates somewhere in the test band: acceleration and foreground
# at low frequency, the background in the middle, OMS at the top.
TRUTH = {"Soms_d": 1.2e-11, "Sa_a": 4e-15, "galfor_amp": 5e-44, "galfor_fk": 3e-3,
         "galfor_alpha": 1.2, "galfor_f_1": 2.5e-3, "galfor_f_2": 8e-4,
         "powerlaw_log10_A": -7.8, "powerlaw_alpha": 0.5}


def setup(coarseQ=1):
    reference = blank_observation(128, 32, 5.0, start_time_gps=10000.0)
    delays = np.tile([8.1, 8.2, 8.3, 8.4, 8.5, 8.6], (32, 1))
    instrument = UnequalArmEqualNoiseBlock(reference, delays, coarseQ=coarseQ)
    block = StochasticNoiseBlock(instrument, [GalacticForeground(), PowerLawSGWB()])
    return reference, instrument, block


def whitened(data, covariance, mask):
    """Per-pixel v C^-1 v / 3; averages to one when the data match the covariance."""
    values = np.stack([data.channel_data[ch][mask] for ch in CHANNELS], axis=-1)
    solved = np.linalg.solve(matrix(covariance)[mask], values[..., None])[..., 0]
    return np.einsum("pi,pi->p", values, solved) / 3


class SyntheticTests(unittest.TestCase):
    def test_state_at_named_parameters(self):
        _, instrument, block = setup()
        state = block.state(TRUTH)
        self.assertEqual(list(state.parameters), list(block.parameter_names))
        np.testing.assert_allclose(list(state.parameters.values()),
                                   [TRUTH[name] for name in block.parameter_names], rtol=1e-12)
        row = block.coordinates(np.array([TRUTH[name] for name in block.parameter_names]))
        np.testing.assert_allclose(row[:4], [np.log(1.2e-11), np.log(4e-15), np.log10(5e-44),
                                             np.log10(3e-3)])
        np.testing.assert_array_equal(row[[4, 7, 8]], [1.2, -7.8, 0.5])
        np.testing.assert_array_equal(matrix(state.covariance), matrix(block.covariance(row)))
        # Instrument-only models take their ASDs by name too.
        levels = {"Soms_d": 1.2e-11, "Sa_a": 4e-15}
        np.testing.assert_allclose(matrix(instrument.state(levels).covariance),
                                   matrix(instrument.covariance(np.log([1.2e-11, 4e-15]))))
        for bad in ({**TRUTH, "extra": 1.0}, {k: v for k, v in TRUTH.items() if k != "Sa_a"}):
            with self.assertRaisesRegex(ValueError, "exactly the parameters"):
                block.state(bad)
        with self.assertRaisesRegex(ValueError, "finite sampling coordinates"):
            block.state({**TRUTH, "galfor_amp": -1.0})

    def test_draw_has_the_block_covariance(self):
        reference, _, block = setup()
        truth, mask = block.state(TRUTH), block.active_mask
        data = simulate(reference, [truth], mask, rng=np.random.default_rng(3))
        for ch in CHANNELS:
            self.assertFalse(data.channel_data[ch][~mask].any())
        self.assertEqual((data.wdm_grid, data.t0, data.fs),
                         (reference.wdm_grid, reference.t0, reference.fs))
        # Many realizations: the whitened power is 1 and so is every XYZ correlation.
        draws = [simulate(reference, [truth], mask, rng=np.random.default_rng(seed))
                 for seed in range(40)]
        power = np.concatenate([whitened(d, truth.covariance, mask) for d in draws])
        self.assertLess(abs(power.mean() - 1), 5 * np.sqrt(2 / 3 / len(power)))
        values = np.stack([[d.channel_data[ch][mask] for ch in CHANNELS] for d in draws])
        sample = np.einsum("dip,djp->pij", values, values) / len(draws)
        expected = matrix(truth.covariance)[mask]
        scale = np.sqrt(np.einsum("pii,pjj->pij", expected, expected))
        self.assertLess(np.abs((sample - expected) / scale).mean(), 2 / np.sqrt(len(draws)))
        # The stochastic terms matter: the instrument alone does not whiten these data.
        alone = block.instrument.state({"Soms_d": 1.2e-11, "Sa_a": 4e-15}).covariance
        self.assertGreater(whitened(draws[0], alone, mask).mean(), 1.5)
        self.assertFalse(np.array_equal(draws[0].channel_data["X"], draws[1].channel_data["X"]))
        np.testing.assert_array_equal(data.channel_data["X"], draws[3].channel_data["X"])
        self.assertTrue(np.isfinite(transform(data, "time").channel_data["X"]).all())

    def test_truth_is_preferred_and_runs_in_a_wheel(self):
        reference, _, block = setup(coarseQ=5)
        data = simulate(reference, [block.state(TRUTH)], block.active_mask,
                        rng=np.random.default_rng(4))
        truth = block.coordinates(np.array([TRUTH[name] for name in block.parameter_names]))
        # OMS, acceleration (raised: lowering it is barely visible on this small
        # grid), foreground amplitude and background amplitude, in sampling units.
        rows = [truth]
        for index, shift in ((0, 0.3), (0, -0.3), (1, 0.7), (2, 0.7), (2, -0.7),
                             (7, 0.5), (7, -0.5)):
            rows.append(truth.copy())
            rows[-1][index] += shift
        scores = block.log_likelihood(data)(np.array(rows))
        self.assertTrue((scores[0] - scores[1:] > 5).all())
        foreground = GalacticForeground(fiducial=truth[2:7])
        fit = StochasticNoiseBlock(UnequalArmEqualNoiseBlock(
            reference, np.tile([8.1, 8.2, 8.3, 8.4, 8.5, 8.6], (32, 1)), coarseQ=5),
            [foreground, PowerLawSGWB(fiducial=truth[7:])], steps_per_call=3)
        wheel = Wheel(data, random_seed=5)
        wheel.add(fit, initial_block_state=fit.initial_result(rng=np.random.default_rng(6)),
                  data_domain="wdm")
        wheel.run(2)
        self.assertEqual(fit.sampler_state["iterations"], 6)

    def test_templates_and_errors(self):
        reference, instrument, _ = setup()
        grid = reference.wdm_grid
        rng = np.random.default_rng(8)
        signal = {ch: 1e-21 * rng.normal(size=grid.array_shape) * grid.active_mask
                  for ch in CHANNELS}
        states = [BlockState(template=signal), BlockState(template=signal)]
        clean = simulate(reference, states, rng=rng)
        np.testing.assert_array_equal(clean.channel_data["Y"], 2 * signal["Y"])
        noise = instrument.state({"Soms_d": 1.5e-11, "Sa_a": 3e-15})
        noisy = simulate(reference, [*states, noise], instrument.active_mask, rng=rng)
        outside = ~instrument.active_mask
        np.testing.assert_array_equal(noisy.channel_data["Z"][outside], 2 * signal["Z"][outside])
        self.assertFalse(np.array_equal(noisy.channel_data["Z"], 2 * signal["Z"]))
        with self.assertRaisesRegex(ValueError, "mask"):
            simulate(reference, [noise], rng=rng)
        with self.assertRaisesRegex(ValueError, "Templates"):
            simulate(reference, [BlockState(template={"X": signal["X"]})], rng=rng)
        with self.assertRaisesRegex(ValueError, "WDM"):
            simulate(transform(reference, "time"), [noise], instrument.active_mask, rng=rng)
        equal = SGWBBlock(UnequalArmEqualNoiseBlock(reference, np.full((32, 6), 8.3)),
                          PowerLawSGWB())
        self.assertEqual(equal.name, "PowerLawSGWBBlock")


if __name__ == "__main__":
    unittest.main()
