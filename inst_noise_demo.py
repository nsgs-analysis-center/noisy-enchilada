"""Unequal-arm instrument noise demo. Run: uv run python inst_noise_demo.py

The model lives in noise_model.py and the blocks in noise_blocks.py; this
script conditions a CD1-L brick and runs a short Wheel campaign.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
from enchilada import L1Data, Wheel, transform
from scipy import signal

from noise_blocks import UnequalArmEqualNoiseBlock
from noise_model import CHANNELS, column_delays


TRIM_S = 200 * 3600  # dropped from each end after filtering


def load_observation(path, nf=768, nt=256, dt=5.0):
    """Read the start of a CD1-L brick and condition it onto the WDM grid."""
    raw = L1Data.from_cd1l(str(path), duration_s=nf * nt * dt + 2 * TRIM_S)
    return condition(raw, nf, nt, dt)


def condition(raw, nf=768, nt=256, dt=5.0):
    """Reference conditioning: high-pass, resample, trim 200 h, Tukey taper."""
    trim = int(TRIM_S / dt)
    down = int(round(dt * raw.fs))
    if down < 1 or not np.isclose(down, dt * raw.fs):
        raise ValueError("dt must be an integer multiple of the input interval")
    xyz = np.stack([raw.channel_data[ch] for ch in CHANNELS])
    xyz = signal.sosfiltfilt(signal.butter(2, 2e-5, "highpass", fs=raw.fs, output="sos"), xyz)
    xyz = signal.resample_poly(xyz, 1, down, axis=-1, window=("kaiser", 31.0), padtype="line")
    xyz = xyz[:, trim:trim + nf * nt] * signal.windows.tukey(nf * nt, 4.0 / nt)
    data = replace(raw, channel_data=dict(zip(CHANNELS, xyz)), num_time_samples=nf * nt,
                   sample_rate_hz=1 / dt, start_time_gps=raw.t0 + trim * dt)
    data = transform(data, "wdm", num_frequency_divisions=nf, num_time_divisions=nt)
    return data, column_delays(data)


if __name__ == "__main__":
    NOISE_FILE = Path(__file__).with_name("NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5")
    #NF, NT, DT = 768, 256, 5.0
    NF, NT, DT = 768, 512*48 , 2.5

    data, delays = load_observation(NOISE_FILE, NF, NT, DT)
    wheel = Wheel(data, random_seed=42)

    block = UnequalArmEqualNoiseBlock(data, delays, coarseQ=32*32, nwalkers=4)
    initial = block.initial_result(rng=np.random.default_rng(42))
    wheel.add(block, initial_block_state=initial, data_domain="wdm")

    for cycle in range(10):
        wheel.run(1)
        print("Adding to the ledger: ", wheel.ledger[block.name].parameters)
