# retinal-adaptive-spike-coding

Simulation code for the manuscript

> **When do adaptive spike codes keep stimulus amplitude recoverable? A retinal-inspired level-crossing model**
> Mehdi Borjkhani, Hadi Borjkhani, Morteza A. Sharif, Dawid Borycki

The model treats ganglion-cell output as a refractory ON/OFF level-crossing encoder (an
asynchronous delta modulator). It compares, on identical stimuli:

| Key | Encoder | Decoder |
| --- | --- | --- |
| `F` | fixed threshold | recomputed from spikes |
| `AS` | additive spike-history threshold, θ = θ0 (1 + r / r½) | recomputed from spikes |
| `AM` | multiplicative spike-history threshold (Bohte 2012 style) | recomputed from spikes |
| `AI` | input-driven divisive gain, θ = θ0 ŝ / s0 | gain-informed, fixed-gain, rate-gain (learned) or per-spike (learned) |
| `F_split`, `AS_split` | separate ON and OFF units | per unit |
| `U` | uniform synchronous ADC | zero-order hold |

Spike times are transmitted and decoded at 1 ms resolution, and the code rate is
estimated at the same resolution. Every condition is scored on held-out realizations;
alignment lags are fixed on independent calibration runs.

## Install

    python -m pip install -r requirements.txt

Python 3.10 or later. Numba compiles the encoder loops on first use.

## Reproduce

    python experiments.py      # all simulations -> results/  (about 10 min on 2 cores)
    python figures.py          # Figures 1-9 and S1-S2 -> figures/*.pdf, *.png
    python stats.py            # every number quoted in the paper -> results/numbers.json

Run single experiments with `python experiments.py e4 e8`. Experiments:

| Name | Figure | Content |
| --- | --- | --- |
| `e1` | 1 | example contrast step |
| `e2` | 2 | frozen-stimulus rasters across contrasts |
| `e3` | 3 | rate-distortion sweeps, timing resolution |
| `e4` | 4, S1 | contrast sweep (10 seeds), learned decoders, split ON/OFF units |
| `e5` | 5 | scale invariance / identifiability |
| `e6` | 6 | contrast switching |
| `e7` | 7 | flash detection, paired bootstrap differences |
| `e8` | 8 | helps/hurts map (5 seeds, two objectives) |
| `e9` | 9 | synaptic failures, input noise |
| `e10` | S2 | controls: time-quantization desynchronization, step sizes, leak × architecture, stimulus bandwidth |
| `e11` | S1 | ridge-strength sensitivity of the learned decoders |

A clean run (`experiments.py` then `stats.py`) reproduces every value in `results/numbers.json` exactly.

## Files

- `core.py` — stimulus, front-end filter, encoders, decoders, channel, metrics, learned decoder
- `experiments.py` — simulation protocols
- `figures.py` — figure layout
- `stats.py` — collects the reported numbers

## Licence and citation

MIT licence (see `LICENSE`). If you use the code, please cite the article (see `CITATION.cff`).
