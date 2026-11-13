"""Quick real-brick mixing check: .venv/bin/python diagnostics/autocorrelation.py."""
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inst_noise_demo as demo


def tau(chain):
    """Average walker ACFs, then use a self-consistent 5*tau window."""
    x = chain - chain.mean(axis=0)
    n = len(x)
    ft = np.fft.rfft(x, n=2 * n, axis=0)
    acf = np.fft.irfft(ft * ft.conj(), n=2 * n, axis=0)[:n]
    acf = (acf / acf[0]).mean(axis=1)
    cumulative = 2 * np.cumsum(acf, axis=0) - 1
    return np.array([cumulative[np.flatnonzero(np.arange(n) >= 5 * cumulative[:, p])[0], p]
                     for p in range(chain.shape[-1])])


class RecordingSampler(demo.EnsembleSampler):
    latest = None

    def run_mcmc(self, *args, **kwargs):
        result = super().run_mcmc(*args, **kwargs)
        RecordingSampler.latest = self
        return result


if __name__ == '__main__':
    demo.EnsembleSampler = RecordingSampler
    path = Path(demo.__file__).with_name('NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5')
    # Same frequency resolution and cadence as the current demo, shorter duration.
    data, delays = demo.load_observation(path, nf=768, nt=256, dt=2.5)
    results = []
    for walkers in (4, 16):
        for seed in (42, 123):
            start = time.perf_counter()
            block = demo.InstrumentNoiseBlock(data, delays, coarseQ=1024,
                                              nwalkers=walkers, steps_per_call=12000)
            initial = block.initial_result(rng=np.random.default_rng(seed))
            block.sample(data, None, initial, rng=np.random.default_rng(seed + 1))
            sampler = RecordingSampler.latest
            chain = sampler.get_chain()['noise'][2000:, 0, :, 0, :]
            result = dict(walkers=walkers, seed=seed, retained_steps=len(chain),
                          tau_log=tau(chain).tolist(), tau_amplitude=tau(np.exp(chain)).tolist(),
                          tau_first_half=tau(chain[:5000]).tolist(),
                          tau_second_half=tau(chain[5000:]).tolist(),
                          acceptance=float(np.mean(sampler.acceptance_fraction)),
                          amplitude_mean=np.exp(chain).mean(axis=(0, 1)).tolist(),
                          seconds=time.perf_counter() - start)
            results.append(result)
            print(json.dumps(result), flush=True)
    Path('/tmp/noisy-autocorrelation.json').write_text(json.dumps(results, indent=2))
