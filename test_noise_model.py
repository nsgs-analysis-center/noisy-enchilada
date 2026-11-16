"""Numerical and lifecycle checks; run with python -m unittest -q."""

import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import wdm
from enchilada import L1Data, WDMGrid, Wheel, transform

from noise_blocks import UnequalArmEqualNoiseBlock
from noise_model import (
    CHANNELS,
    FIDUCIAL,
    LINKS,
    UI,
    UJ,
    column_delays,
    gaussian_loglike,
    TILE,
    _loglike_parallel,
    _loglike_serial,
    instrument_bases,
    unit_bases,
)


def matrix(covariance):
    return np.stack([np.stack([covariance.pair(a, b) for b in CHANNELS], axis=-1)
                     for a in CHANNELS], axis=-2)


def reference_score(covariance, data, mask):
    cov = matrix(covariance)[mask]
    values = np.stack([data.channel_data[ch][mask] for ch in CHANNELS], axis=-1)
    return (np.einsum("pi,pi->", values, np.linalg.solve(cov, values[..., None])[..., 0])
            + np.linalg.slogdet(cov)[1].sum())


def campaign():
    grid = WDMGrid(num_frequency_divisions=128, num_time_divisions=32)
    data = L1Data(
        channel_names=CHANNELS,
        channel_data={ch: np.zeros(grid.array_shape) for ch in CHANNELS},
        sample_rate_hz=0.2,
        start_time_gps=10000.0,
        data_domain="wdm",
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
    )
    delays = np.tile([8.1, 8.2, 8.3, 8.4, 8.5, 8.6], (32, 1))
    block = UnequalArmEqualNoiseBlock(data, delays, nwalkers=6, steps_per_call=5)
    initial = block.initial_result(rng=np.random.default_rng(12))
    covariance = initial.covariance
    mask = block.active_mask
    factors = np.linalg.cholesky(matrix(covariance)[mask])
    draws = np.random.default_rng(13).normal(size=(mask.sum(), 3))
    values = np.zeros((*grid.array_shape, 3))
    values[mask] = np.einsum("...ij,...j->...i", factors, draws)
    data = replace(data, channel_data={ch: values[..., i] for i, ch in enumerate(CHANNELS)})
    return data, block, initial, delays


class InstrumentTests(unittest.TestCase):
    def test_likelihood_matches_enchilada_covariance(self):
        data, block, initial, delays = campaign()
        # Include a changing arm geometry, not just constant LTTs.
        delays *= np.linspace(0.995, 1.005, len(delays))[:, None]
        block = UnequalArmEqualNoiseBlock(data, delays)
        for amplitudes in (FIDUCIAL, (3e-11, 6e-15), (6.1e-12, 1.9e-13)):
            logs = np.log(amplitudes)
            covariance = block.covariance(logs)
            expected = -0.5 * (
                reference_score(covariance, data, block.active_mask)
            )
            np.testing.assert_allclose(block.log_likelihood(data)(logs)[0], expected, rtol=2e-12)
        self.assertIsNone(initial.template)
        self.assertFalse(block.active_mask[[0, -1]].any())
        self.assertFalse(block.active_mask[:, [0, 1, -2, -1]].any())
        # Inactive data cannot influence this block's likelihood.
        changed = {ch: values.copy() for ch, values in data.channel_data.items()}
        unused = data.wdm_grid.active_mask & ~block.active_mask
        for values in changed.values():
            values[unused] = 1.0
        np.testing.assert_array_equal(
            block.log_likelihood(replace(data, channel_data=changed))(logs),
            block.log_likelihood(data)(logs),
        )

    def test_coarse_likelihood_matches_published_covariance(self):
        data, _, _, delays = campaign()
        delays *= np.linspace(0.98, 1.02, len(delays))[:, None]
        for q in (1, 5, 100):  # includes a short final cell and one oversized cell
            block = UnequalArmEqualNoiseBlock(data, delays, coarseQ=q)
            self.assertEqual(block.region.sizes.sum(), len(block.region.columns))
            logs = np.log(FIDUCIAL)
            cov = block.covariance(logs)
            expected = -0.5 * (reference_score(cov, data, block.active_mask))
            np.testing.assert_allclose(block.log_likelihood(data)([logs, logs]),
                                       [expected, expected], rtol=2e-12)
        for q in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                UnequalArmEqualNoiseBlock(data, delays, coarseQ=q)

    def test_fold_agrees_with_wdm_backend(self):
        data, block, _, delays = campaign()
        nf, nt = data.wdm_grid.num_frequency_divisions, data.wdm_grid.num_time_divisions
        # Test all diagonal and cross-channel elements through v^T C v,
        # which is a nonnegative scalar PSD accepted by wdm.
        vectors = np.vstack((np.eye(3), [1, 1, 0], [1, 0, 1], [0, 1, 1]))
        layers = block.region.layers
        bases = instrument_bases(layers, data.wdm_grid, data.dt, delays)[:, UI, UJ]
        for basis in range(2):
            for vector in vectors:

                def psd(f):
                    cov = unit_bases(np.maximum(f.ravel(), 1e-10), delays[0])[basis]
                    return np.einsum("i,ijf,j->f", vector, cov, vector).reshape(f.shape)

                expected = wdm.band_averaged_variance(nf, nt, data.dt, psd)[layers]
                weights = np.array(
                    [
                        vector[0] ** 2,
                        vector[1] ** 2,
                        vector[2] ** 2,
                        2 * vector[0] * vector[1],
                        2 * vector[0] * vector[2],
                        2 * vector[1] * vector[2],
                    ]
                )
                actual = np.einsum("e,eft->ft", weights, bases[basis])
                np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)

    def test_transform_scaling(self):
        rng = np.random.default_rng(42)
        data = L1Data(
            channel_names=CHANNELS,
            channel_data={ch: rng.normal(size=8 * 16) for ch in CHANNELS},
            sample_rate_hz=0.2,
            tdi_generation="2.0",
            physical_observable="fractional_frequency",
        )
        transformed = transform(data, "wdm", num_frequency_divisions=8)
        reconstructed = transform(transformed, "time")
        for ch in CHANNELS:
            np.testing.assert_allclose(
                transformed.channel_data[ch],
                wdm.forward_time(data.channel_data[ch], 8, 16, data.dt),
                atol=1e-14,
            )
            np.testing.assert_allclose(
                np.sum(transformed.channel_data[ch] ** 2),
                data.dt * np.sum(data.channel_data[ch] ** 2),
                rtol=1e-14,
            )
            np.testing.assert_allclose(
                reconstructed.channel_data[ch], data.channel_data[ch], atol=1e-14
            )

    def test_wheel_continuation_and_rng_ownership(self):
        data, block, initial, _ = campaign()
        saved = deepcopy(block.sampler_state)
        global_rng = np.random.get_state()
        wheels = [Wheel(data, random_seed=7) for _ in range(2)]
        blocks = [UnequalArmEqualNoiseBlock(data, campaign()[3], nwalkers=6, steps_per_call=5)
                  for _ in wheels]
        for wheel, owned in zip(wheels, blocks):
            state = owned.initial_result(rng=np.random.default_rng(12))
            wheel.add(owned, initial_block_state=state, data_domain="wdm")
        wheels[0].run(2)
        wheels[1].run(1)
        wheels[1].run(1)
        a, b = [wheel.ledger[block.name] for wheel in wheels]
        np.testing.assert_array_equal(blocks[0].sampler_state["coords"], blocks[1].sampler_state["coords"])
        np.testing.assert_array_equal(block.sampler_state["coords"], saved["coords"])
        self.assertEqual(a.parameters, b.parameters)
        self.assertEqual(blocks[0].sampler_state["iterations"], 10)
        self.assertEqual(block.sampler_state["iterations"], 0)
        after = np.random.get_state()
        np.testing.assert_array_equal(global_rng[1], after[1])
        self.assertEqual(global_rng[0], after[0])
        self.assertEqual(global_rng[2:], after[2:])
        np.testing.assert_array_equal(
            list(a.parameters.values()), np.exp(blocks[0].sampler_state["coords"][0, 0, 0])
        )

    def test_residual_is_refreshed_and_epoch_checked(self):
        data, block, initial, _ = campaign()
        changed = replace(data, channel_data={ch: 20 * x for ch, x in data.channel_data.items()})
        block.sample(data, None, initial, rng=np.random.default_rng(7))
        a = deepcopy(block.sampler_state)
        block.initial_result(rng=np.random.default_rng(12))
        block.sample(changed, None, initial, rng=np.random.default_rng(7))
        b = block.sampler_state
        self.assertFalse(np.array_equal(a["coords"], b["coords"]))
        with self.assertRaisesRegex(ValueError, "epoch"):
            block.sample(
                replace(data, start_time_gps=data.t0 + 1),
                None,
                initial,
                rng=np.random.default_rng(7),
            )

    def test_serial_and_parallel_kernels_agree_exactly(self):
        rng = np.random.default_rng(5)
        pixels = 5 * TILE + 17  # includes a partial final tile
        B = np.abs(rng.normal(size=(pixels, 12, 6)))
        B[:, :, :3] += 10  # positive-definite pixels
        X, weights = np.abs(rng.normal(size=(pixels, 6))), rng.uniform(1, 4, pixels)
        levels = np.abs(rng.normal(size=(7, 12)))
        np.testing.assert_array_equal(_loglike_serial(levels[:, None], B, X, weights),
                                      _loglike_parallel(levels[:, None], B, X, weights))

    def test_non_positive_definite_covariance_rejected(self):
        self.assertEqual(
            gaussian_loglike(np.array([1.0, 1.0, 1.0, 2.0, 0.0, 0.0]), np.ones(6)), -np.inf
        )

    def test_cd1l_reader_and_directed_delays(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "brick.h5"
            t0 = 1e8
            with h5py.File(path, "w") as f:
                f.attrs["laser_frequency"] = 10.0
                sampling = f.create_group("tdis/sampling")
                sampling.attrs.update(fs=0.2, t0=t0, size=256)
                for i, ch in enumerate(CHANNELS):
                    f[f"tdis/{ch}2"] = (i + 1) * np.arange(256, dtype=float)
                sampling = f.create_group("orbits/sampling")
                sampling.attrs.update(t0=t0 - 100, dt=500.0, size=5)
                for i in range(3):
                    position = (
                        np.array([np.cos(i * 2 * np.pi / 3), np.sin(i * 2 * np.pi / 3), 0]) * 1e9
                    )
                    f[f"orbits/sc_position_{i + 1}"] = np.tile(position, (5, 1))
                sampling = f.create_group("ltts/sampling")
                sampling.attrs.update(t0=t0 - 100, dt=1.0, size=1501)
                for i, link in enumerate(LINKS):
                    f[f"ltts/ltt_{link}"] = np.full(1501, 8.0 + i / 10)
            data = L1Data.from_cd1l(str(path), duration_s=1200)
            self.assertEqual(data.N, 240)
            self.assertEqual(data.t0, t0)
            self.assertEqual(data.physical_observable, "fractional_frequency")
            np.testing.assert_allclose(data.channel_data["Y"], 0.2 * np.arange(240), rtol=1e-15)
            self.assertIsNotNone(data.spacecraft_positions_m)
            data = replace(data, channel_data={ch: x[16:-16] for ch, x in data.channel_data.items()},
                           num_time_samples=208, start_time_gps=t0 + 80)
            data = transform(data, "wdm", num_frequency_divisions=8)
            delays = data.light_travel_times_s.T
            expected = np.tile(8 + np.arange(6) / 10, (26, 1))
            np.testing.assert_allclose(column_delays(data, stride=2), expected, rtol=1e-14)
            self.assertNotEqual(
                delays[0, LINKS.index(12)], delays[0, LINKS.index(21)]
            )
            with self.assertRaisesRegex(ValueError, "spans"):
                column_delays(replace(data, light_travel_time_start_time_gps=t0 - 5000), stride=2)
            with self.assertRaisesRegex(ValueError, "empty"):
                column_delays(data, stride=100)


if __name__ == "__main__":
    unittest.main()
