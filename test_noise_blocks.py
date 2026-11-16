"""Alternative noise models: normalization, link projection and Wheel lifecycle."""
import unittest
from dataclasses import replace

import numpy as np
from enchilada import Wheel

from noise_blocks import (
    EqualArmEqualNoiseBlock, EqualArmUnequalNoiseBlock, FixedNoiseBlock,
    L1NoiseEstimateBlock, UnequalArmEqualNoiseBlock, UnequalArmUnequalNoiseBlock,
)
from noise_model import C_SI, FIDUCIAL, LINKS, unit_bases
from test_noise_model import campaign, matrix, reference_score


def glass_xyz(f, L, soms, sacc):
    """Transcription of GLASS generate_instrument_noise_model, Nchannel=3.

    glass/noise/src/glass_noise_model.c; soms and sacc map GLASS link labels
    to power levels (GLASS units: m²/Hz and m²/s⁴/Hz).
    """
    x = 2 * np.pi * f * L  # f / fstar
    o = {k: v * (2*np.pi/C_SI)**2 * f**2 * (1 + (2e-3/f)**4) for k, v in soms.items()}
    a = {k: v / (2*np.pi*C_SI)**2 / f**2 * (1 + (.4e-3/f)**2) * (1 + (f/8e-3)**4)
         for k, v in sacc.items()}
    c2 = np.cos(x)**2
    C = np.zeros((3, 3, len(f)))
    C[0, 0] = (o[12]+o[21]+o[13]+o[31])*.25 + a[12]+a[13] + (a[21]+a[31])*c2
    C[1, 1] = (o[23]+o[32]+o[21]+o[12])*.25 + a[23]+a[21] + (a[32]+a[12])*c2
    C[2, 2] = (o[31]+o[13]+o[32]+o[23])*.25 + a[31]+a[32] + (a[13]+a[23])*c2
    C[0, 1] = (o[12]+o[21])*.5 + 2*(a[12]+a[21])
    C[0, 2] = (o[13]+o[31])*.5 + 2*(a[13]+a[31])
    C[1, 2] = (o[32]+o[23])*.5 + 2*(a[32]+a[23])
    tdi = -8 * np.sin(x)**2
    for i in range(3):
        C[i, i] *= -2 * tdi
    for i, j in ((0, 1), (0, 2), (1, 2)):
        C[i, j] *= tdi * np.cos(x)
        C[j, i] = C[i, j]
    return C / 4


class AlternativeNoiseTests(unittest.TestCase):
    def test_equal_arm_unequal_noise_reproduces_glass(self):
        f = np.geomspace(3e-4, 8e-3, 30)
        L = 2.5e9 / C_SI
        rng = np.random.default_rng(3)
        logs = np.log(EqualArmUnequalNoiseBlock.fiducial) + rng.uniform(-.5, .5, 9)
        bases = EqualArmUnequalNoiseBlock.unit_model(f, np.full(6, L))
        ours = np.einsum('b,bijf->ijf', np.exp(2 * logs), bases)
        power = np.exp(2 * logs)
        soms = {}
        for (a, b), level in zip(((12, 21), (23, 32), (31, 13)), power[:3]):
            soms[a] = soms[b] = level
        # GLASS labels the acceleration of our link ij as ji.
        sacc = {int(str(link)[::-1]): level for link, level in zip(LINKS, power[3:])}
        tdi2 = 4 * np.sin(4 * np.pi * f * L)**2
        np.testing.assert_allclose(ours, 4 * tdi2 * glass_xyz(f, L, soms, sacc), rtol=1e-10)
        # Reverse OMS links are indistinguishable at equal arms, hence the tie.
        per_link = unit_bases(f, np.full(6, L), per_link=True)
        for i, j in ((0, 5), (1, 4), (2, 3)):
            np.testing.assert_allclose(per_link[i], per_link[j], rtol=1e-12)

    def test_equal_arm_blocks_are_constant_and_consistent(self):
        data, _, _, _ = campaign()
        shared = EqualArmEqualNoiseBlock(data, 8.3, nwalkers=6, steps_per_call=3)
        links = EqualArmUnequalNoiseBlock(data, 8.3, steps_per_call=3, coarseQ=5)
        self.assertEqual(len(links.parameter_names), 9)
        values = matrix(shared.covariance(np.log(FIDUCIAL)))
        fitted = values[shared.region.layers][:, shared.region.columns]
        np.testing.assert_allclose(fitted, np.broadcast_to(fitted[:, :1], fitted.shape), rtol=1e-12)
        np.testing.assert_allclose(matrix(links.covariance(np.log(links.fiducial))), values,
                                   rtol=1e-12)
        logs = np.log(links.fiducial) + np.linspace(-.2, .2, 9)
        expected = -.5 * reference_score(links.covariance(logs), data, links.active_mask)
        np.testing.assert_allclose(links.log_likelihood(data)(logs)[0], expected, rtol=2e-12)
        for block in (shared, links):
            wheel = Wheel(data, random_seed=4)
            wheel.add(block, initial_block_state=block.initial_result(rng=np.random.default_rng(5)),
                      data_domain='wdm')
            wheel.run(2)
            self.assertEqual(list(wheel.ledger[block.name].parameters), list(block.parameter_names))
        with self.assertRaisesRegex(ValueError, '18 walkers'):
            EqualArmUnequalNoiseBlock(data, 8.3, nwalkers=16)
        with self.assertRaises(TypeError):  # tied pairs have no ratio to constrain
            EqualArmUnequalNoiseBlock(data, 8.3, oms_log_power_ratio_sigma=.5)
        with self.assertRaisesRegex(ValueError, 'light-travel-time'):
            EqualArmEqualNoiseBlock(data)
        with self.assertRaisesRegex(ValueError, 'positive finite'):
            EqualArmEqualNoiseBlock(data, -1.)

    def test_injection_matches_sampled_model_and_is_fixed(self):
        data, model, initial, delays = campaign()
        block = FixedNoiseBlock(UnequalArmEqualNoiseBlock(data, delays))
        self.assertEqual(block.name, 'FixedUnequalArmEqualNoiseBlock')
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

    def test_fixed_link_model_uses_its_parameter_order(self):
        data, _, initial, delays = campaign()
        model = UnequalArmUnequalNoiseBlock(data, delays)
        # Equal link levels at the fiducial reproduce the shared-noise covariance.
        state = FixedNoiseBlock(model).initial_result()
        np.testing.assert_allclose(matrix(state.covariance), matrix(initial.covariance),
                                   rtol=1e-12)
        self.assertEqual(list(state.parameters), list(model.parameter_names))
        amplitudes = np.repeat(FIDUCIAL, 6) * np.linspace(.8, 1.2, 12)
        state = FixedNoiseBlock(model, amplitudes).initial_result()
        np.testing.assert_allclose(matrix(state.covariance),
                                   matrix(model.covariance(np.log(amplitudes))))
        with self.assertRaisesRegex(ValueError, '12 positive'):
            FixedNoiseBlock(model, FIDUCIAL)

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
        block = L1NoiseEstimateBlock(data)
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
            L1NoiseEstimateBlock(campaign()[0])
        with self.assertRaisesRegex(ValueError, 'positive definite'):
            L1NoiseEstimateBlock(replace(data, noise_estimates_xyz=-spectra))

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
        block = UnequalArmUnequalNoiseBlock(data, delays, steps_per_call=3, coarseQ=5)
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
            UnequalArmUnequalNoiseBlock(data, delays, nwalkers=4)


if __name__ == '__main__':
    unittest.main()
