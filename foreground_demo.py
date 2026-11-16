"""Galactic foreground demo. Run: uv run python foreground_demo.py

Adds the GALFOR brick to the instrument-noise brick, conditions the sum, and
samples two instrument levels together with the five foreground parameters.
The model lives in foreground_blocks.py; the conditioning is
inst_noise_demo.py's.
"""

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
from enchilada import L1Data, Wheel

from foreground_blocks import GalacticForeground, GalacticForegroundBlock
from inst_noise_demo import TRIM_S, condition
from noise_blocks import UnequalArmEqualNoiseBlock
from noise_model import CHANNELS


def add_signal(raw, path):
    """Add a signal brick's fractional-frequency XYZ to raw time-domain data.

    Read directly: L1Data.from_cd1l rejects the GALFOR brick's one-point
    noise-estimate table.
    """
    with h5py.File(path, "r") as f:
        sampling = f["tdis/sampling"].attrs
        if (sampling["t0"], sampling["fs"]) != (raw.t0, raw.fs):
            raise ValueError("Bricks must share their start time and sample rate")
        signal = {ch: f[f"tdis/{ch}2"][:raw.N] / f.attrs["laser_frequency"] for ch in CHANNELS}
    if any(len(values) != raw.N for values in signal.values()):
        raise ValueError("The signal brick is shorter than the observation")
    return replace(raw, channel_data={ch: raw.channel_data[ch] + signal[ch] for ch in CHANNELS})


if __name__ == "__main__":
    HERE = Path(__file__).parent
    NOISE_FILE = HERE / "NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5"
    GALFOR_FILE = HERE / "GALFOR_731d_2.5s_L1.h5"
    MODULATION_FILE = HERE / "modulation_unequal.dat"
    NF, NT, DT = 768, 512 * 48, 2.5

    raw = L1Data.from_cd1l(str(NOISE_FILE), duration_s=NF * NT * DT + 2 * TRIM_S)
    data, delays = condition(add_signal(raw, GALFOR_FILE), NF, NT, DT)

    instrument = UnequalArmEqualNoiseBlock(data, delays, coarseQ=256)
    block = GalacticForegroundBlock(instrument, GalacticForeground.from_file(MODULATION_FILE))
    wheel = Wheel(data, random_seed=42)
    initial = block.initial_result(rng=np.random.default_rng(42))
    wheel.add(block, initial_block_state=initial, data_domain="wdm")

    for cycle in range(10):
        wheel.run(1)
        print("Adding to the ledger: ", wheel.ledger[block.name].parameters)
