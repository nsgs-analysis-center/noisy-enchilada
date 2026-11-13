"""Alternative noise models: normalization, link projection and Wheel lifecycle."""
import unittest
from dataclasses import replace

import numpy as np
from enchilada import Wheel

from inst_noise_demo import FIDUCIAL, LINKS, unit_bases
from noise_blocks import InjectionNoiseBlock, CD1LNoiseEstimateBlock, LinkNoiseBlock
from test_inst_noise_demo import campaign, matrix, reference_score


class AlternativeNoiseTests(unittest.TestCase):
    def test_injection_matches_sampled_model_and_is_fixed(self):
        data, model, initial, delays = campaign()
        block = InjectionNoiseBlock(data, delays)
        state = block.initial_result()
        np.testing.assert_allclose(matrix(state.covariance), matrix(initial.covariance))
        rng = np.random.default_rng(1)
        before = repr(rng.bit_generator.state)
        self.assertIs(block.sample(data, None, state, rng=rng), state)
        self.assertEqual(before, repr(rng.bit_generator.state))
        wheel = Wheel(data)
        wheel.add(block, initial_block_state=state, data_domain='wdm')
        wheel.run(2)
        np.testing.assert_array_equal(matrix(wheel.covariance), matrix(state.covariance))
        with self.assertRaisesRegex(ValueError, 'epoch'):
            block.sample(replace(data, start_time_gps=data.t0+1), None, state, rng=rng)

    def test_estimate_fold_scaling_and_time_interpolation(self):
        data, _, _, _ = campaign()
        # Nonzero cross terms; imaginary antisymmetric CSD cancels in real pixels.
        base = np.array([[4, .2+.1j, -.1j], [.2-.1j, 3, .3], [.1j, .3, 2]]) * 1e-40
        t = data.t0 + np.array([0, data.Tobs])
        spectra = np.stack([np.tile(base, (8, 1, 1)), np.tile(2*base, (8, 1, 1))])
        data = replace(data, noise_estimates_xyz=spectra * 100,
                       laser_frequency_hz=10., noise_estimate_start_time_gps=t[0],
                       noise_estimate_sample_interval_s=t[1]-t[0],
                       noise_estimate_min_frequency_hz=1e-5, noise_estimate_max_frequency_hz=1.)
        block = CD1LNoiseEstimateBlock(data)
        state = block.initial_result()
        values = matrix(state.covariance)
        for column in (3, 10, 28):
            expected = .5 * base.real * (1 + column / 32)
            np.testing.assert_allclose(values[block.active_mask[:, column], column],
                                       np.broadcast_to(expected, values[block.active_mask[:, column], column].shape))
        wheel = Wheel(data)
        wheel.add(block, initial_block_state=state, data_domain='wdm')
        wheel.run(2)
        with self.assertRaisesRegex(ValueError, 'required'):
            CD1LNoiseEstimateBlock(campaign()[0])
        with self.assertRaisesRegex(ValueError, 'positive definite'):
            CD1LNoiseEstimateBlock(replace(data, noise_estimates_xyz=-spectra))

    def test_link_bases_sum_to_shared_noise(self):
        f = np.geomspace(3e-4, 8e-3, 40)
        d = np.array([8.1, 8.2, 8.3, 8.4, 8.5, 8.6])
        per_link = unit_bases(f, d, per_link=True)
        shared = unit_bases(f, d)
        np.testing.assert_allclose(per_link[:6].sum(axis=0), shared[0], rtol=2e-13, atol=1e-35)
        np.testing.assert_allclose(per_link[6:].sum(axis=0), shared[1], rtol=2e-13, atol=1e-20)
        # OMS in link 12 must not equal the directed reverse link 21.
        self.assertFalse(np.array_equal(per_link[LINKS.index(12)], per_link[LINKS.index(21)]))

    def test_link_likelihood_and_sampling(self):
        data, _, _, delays = campaign()
        block = LinkNoiseBlock(data, delays, steps_per_call=3, coarseQ=5)
        state = block.initial_result(rng=np.random.default_rng(1))
        logs = np.log(np.repeat(FIDUCIAL, 6)) + np.linspace(-.2, .2, 12)
        expected = -.5 * reference_score(block.covariance(logs), data, block.active_mask)
        np.testing.assert_allclose(block.log_likelihood(data)(logs)[0], expected, rtol=2e-12)
        wheel = Wheel(data, random_seed=2)
        wheel.add(block, initial_block_state=state, data_domain='wdm')
        wheel.run(2)
        self.assertEqual(block.sampler_state['iterations'], 6)
        self.assertEqual(len(wheel.ledger[block.name].parameters), 12)
        with self.assertRaisesRegex(ValueError, '24 walkers'):
            LinkNoiseBlock(data, delays, nwalkers=4)


if __name__ == '__main__':
    unittest.main()
