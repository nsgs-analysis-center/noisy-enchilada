"""Stochastic background demo. Run: uv run python sgwb_demo.py

Adds a simulated isotropic background to the instrument-noise brick and
samples two instrument levels together with the template's parameters. The
templates live in sgwb_blocks.py; the conditioning is inst_noise_demo.py's.

The injection is a Gaussian draw from the block's own WDM covariance
(independent pixels, equal-arm response), so recovering it checks the sampler,
not the response model.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
from enchilada import Wheel

from inst_noise_demo import load_observation
from noise_blocks import UnequalArmEqualNoiseBlock
from noise_model import CHANNELS
from sgwb_blocks import LogNormalSGWB, PhaseTransitionSGWB, PowerLawSGWB, SGWBBlock


def inject(data, block, coordinates, rng):
    """Add a draw of the block's isotropic components to WDM data.

    ``coordinates`` are the components' sampling coordinates, in block order.
    """
    row = np.concatenate([block.instrument.center, coordinates])
    variance = block.levels(row)[0][:, block.split:].sum(axis=-1)  # XX, per layer
    layers = block.region.layers
    z = rng.normal(size=(3, len(layers), data.wdm_grid.num_time_divisions))
    # z - mean(z) has covariance 1 - 1/3 over channels: 3/2 of it is the isotropic pattern.
    draw = np.sqrt(1.5 * variance)[:, None] * (z - z.mean(axis=0))
    draw *= data.wdm_grid.active_mask[layers]
    channel_data = {ch: values.copy() for ch, values in data.channel_data.items()}
    for ch, values in zip(CHANNELS, draw):
        channel_data[ch][layers] += values
    return replace(data, channel_data=channel_data)


if __name__ == "__main__":
    NOISE_FILE = Path(__file__).with_name("NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5")
    NF, NT, DT = 768, 512 * 48, 2.5
    # Each template with its injected coordinates; walkers start around the
    # template's fiducial. The first two are GLASS's default injections; its
    # phase-transition default (log10_Ap = -18) is far below the instrument noise.
    INJECTIONS = {
        "power law": (PowerLawSGWB(), (-8.45, 2 / 3)),
        "log-normal": (LogNormalSGWB(), (-2.5, np.log10(2e-3), 0.0)),
        "phase transition": (PhaseTransitionSGWB(fiducial=(1.0, 2.0, -9.5, -2.5)),
                             (1.0, 2.0, -9.5, -2.5)),
    }
    TEMPLATE, INJECTION = INJECTIONS["power law"]

    data, delays = load_observation(NOISE_FILE, NF, NT, DT)
    block = SGWBBlock(UnequalArmEqualNoiseBlock(data, delays, coarseQ=32 * 32), TEMPLATE)
    data = inject(data, block, INJECTION, np.random.default_rng(7))

    wheel = Wheel(data, random_seed=42)
    initial = block.initial_result(rng=np.random.default_rng(42))
    wheel.add(block, initial_block_state=initial, data_domain="wdm")

    for cycle in range(10):
        wheel.run(1)
        print("Adding to the ledger: ", wheel.ledger[block.name].parameters)
