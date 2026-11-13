"""Detailed balance, soft OMS constraint, and sampler integration."""
import unittest
from copy import deepcopy

import numpy as np
from enchilada import Wheel

from noise_blocks import LinkNoiseBlock
from noise_proposals import ACC_CYCLE, LinkNoiseMove, OMSPairPrior, REVERSE_PAIRS
from test_inst_noise_demo import campaign


class FixedDraws:
    """Select one proposal component with a prescribed signed increment."""

    def __init__(self, kind, step, family=0):
        self.ints = iter([kind, 0, family] if kind == 1 else [kind, 0])
        self.step = step

    def randint(self, n):
        return next(self.ints)

    def choice(self, values):
        return 1.

    def normal(self):
        return self.step


class ProposalTests(unittest.TestCase):
    def test_power_moves_are_reversible_with_correct_jacobian(self):
        center = np.linspace(-34., -25., 12)
        # Reverse pairs have the same nominal scale, as in LinkNoiseBlock.
        center[:6] = -25.
        center[6:] = -34.
        move = LinkNoiseMove(np.exp(center))
        old = (center + np.linspace(-.05, .05, 12))[None, None, None, :]
        for kind, family in ((1, 0), (1, 1), (2, 0)):
            proposed, factor = move.get_proposal({'noise': old}, FixedDraws(kind, .1, family))
            new = proposed['noise']
            change = np.exp(2 * (new-center)) - np.exp(2 * (old-center))
            expected = np.zeros(12)
            if kind == 1:
                i, j = REVERSE_PAIRS[0]
                expected[6*family+i], expected[6*family+j] = .1, -.1
            else:
                expected[6:] = .1 * ACC_CYCLE
            np.testing.assert_allclose(change.ravel(), expected, atol=2e-14)
            # Analytic log Jacobian ratio for symmetric power proposals.
            np.testing.assert_allclose(factor, 2*(old-new).sum(axis=(-1, -2)))
            reverse, reverse_factor = move.get_proposal(proposed, FixedDraws(kind, -.1, family))
            np.testing.assert_allclose(reverse['noise'], old, atol=1e-14)
            np.testing.assert_allclose(reverse_factor, -factor, atol=1e-14)

    def test_invalid_power_step_is_rejected_without_clipping(self):
        move = LinkNoiseMove(np.ones(12))
        old = np.zeros((1, 1, 1, 12))
        proposed, factor = move.get_proposal({'noise': old}, FixedDraws(1, 2.))
        np.testing.assert_array_equal(proposed['noise'], old)
        self.assertEqual(factor[0, 0], -np.inf)

    def test_soft_constraint_allows_unequal_oms_and_respects_bounds(self):
        bounds = np.tile([-2., 2.], (12, 1))
        logs = np.zeros((3, 12))
        logs[1, 0] = .25  # log power ratio = .5
        logs[2, 0] = 3.
        prior = OMSPairPrior(bounds, sigma=.5)
        np.testing.assert_allclose(prior.logpdf(logs), [0., -.5, -np.inf])
        np.testing.assert_allclose(OMSPairPrior(bounds).logpdf(logs), [0., 0., -np.inf])
        for sigma in (0., -1., np.inf, np.nan):
            with self.assertRaises(ValueError):
                OMSPairPrior(bounds, sigma)

    def test_targeted_moves_and_constraint_through_wheel(self):
        data, _, _, delays = campaign()
        global_rng = deepcopy(np.random.get_state())
        wheels, blocks = [], []
        for _ in range(2):
            block = LinkNoiseBlock(data, delays, steps_per_call=10,
                                   oms_log_power_ratio_sigma=.5)
            initial = block.initial_result(rng=np.random.default_rng(1))
            wheel = Wheel(data, random_seed=7)
            wheel.add(block, initial_block_state=initial, data_domain='wdm')
            wheels.append(wheel)
            blocks.append(block)
        wheels[0].run(2)
        wheels[1].run(1)
        wheels[1].run(1)
        np.testing.assert_array_equal(blocks[0].sampler_state['coords'],
                                      blocks[1].sampler_state['coords'])
        np.testing.assert_array_equal(global_rng[1], np.random.get_state()[1])
        self.assertEqual(global_rng[2:], np.random.get_state()[2:])
        self.assertIs(blocks[0].make_prior(), blocks[0].link_prior)
        self.assertTrue(np.isfinite(blocks[0].link_prior.logpdf(
            blocks[0].sampler_state['coords'])).all())
        baseline = LinkNoiseBlock(data, delays, targeted_proposals=False)
        self.assertNotIsInstance(baseline.make_moves(), list)


if __name__ == '__main__':
    unittest.main()
