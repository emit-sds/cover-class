# specmix — EMIT spectral mixing simulator

Simulate realistic EMIT hyperspectral **mixtures** from pure-endmember spectral
libraries, to generate training data for **fractional cover unmixing** of 5
classes: PV, NPV, Soil, Snow+Ice, Water.

Training data with known fractions is infeasible to collect from real scenes, so
we build it: sample a class-presence pattern, mix within and between classes,
apply per-class brightness/glint augmentation, and add instrument noise. Each
call returns one `(spectra, fractions)` pair — `fractions` is the regression
target; binarizing to class presence is left to the caller.

## Install

```bash
pip install -e .            # core (numpy/scipy/sklearn/h5py/pyyaml/pandas/kmedoids)
pip install -e ".[torch]"   # + PyTorch, to use MixtureDataset with a DataLoader
pip install -e ".[viz]"     # + matplotlib, for the eval / experiment plots
```

## Usage

```python
import numpy as np
from specmix import MixtureSimulator

sim = MixtureSimulator()                 # packaged default config
rng = np.random.default_rng(0)
spectra, fractions = sim.simulate(rng)   # spectra: (207,)  fractions: (5,)
```

As a PyTorch dataset:

```python
from torch.utils.data import DataLoader
from specmix import MixtureDataset, numpy_collate

ds = MixtureDataset(epoch_size=100_000)
dl = DataLoader(ds, batch_size=256, num_workers=4, collate_fn=numpy_collate)
for X, F in dl:      # X: (B, 207) reflectance, F: (B, 5) fractions
    ...
```

Each item is an independent, reproducible mixture (per-item RNG seeded from
`(base_seed, worker_id, index)` — no cross-worker collisions).

## Training a model

`specmix` generates the data; the unmixing model that learns from it lives in
[`models/`](models/README.md) — one entrypoint trains a SpecTf encoder for
either **classification** (which classes are present) or **regression**
(per-class fractions), and two eval scripts score checkpoints on the simulation,
OOD, and Francisco validation sets. See its README for train/eval commands.

## Data root (endmember libraries)

The default config, mixture prior, and noise covariance **ship with the
package**. The large endmember **HDF5 libraries do not** — they are resolved
against a *data root*, in this order:

1. `MixtureSimulator(data_root="/path/to/root")`
2. `$SPECMIX_DATA_ROOT`
3. the current working directory (so running from the repo root, which contains
   `datasets/`, just works).

The config references them as e.g. `datasets/emit-endmembers/soil_soil.hdf5`,
resolved relative to that root.

## Custom config

```python
sim = MixtureSimulator("my_config.yaml")          # your own copy
sim = MixtureSimulator(config_dict)               # or an already-loaded dict
```

Start from `src/specmix/sim_config.yaml` (the packaged default) — it documents
every field (presence prior, intra/inter-class mixing, augmentation, noise,
subsampling).

## Self-tests

```bash
python -m specmix.simulator --selftest     # fraction/pattern/range invariants
python -m specmix.dataset                  # multi-worker uniqueness + reproducibility
python -m specmix.mixture_priors           # dump the presence prior
```

## Repo layout

```
src/specmix/          the installable simulator package
  simulator.py          MixtureSimulator (the 7-step pipeline)
  dataset.py            MixtureDataset + collate/presence helpers
  bands, augment, noise, subsample, mixture_priors   pipeline components
  sim_config.yaml       packaged default config
  data/                 bundled: mixture prior CSV + noise covariance CSV
models/               train + eval the unmixing model on simulated data (see models/README.md)
datasets/             external endmember libraries + validation sets (not bundled; see Data root)
```

## Simulator pipeline

`MixtureSimulator.simulate` builds one mixture in 7 steps (see `simulator.py`):

1. **presence pattern** — sample which classes are present from the prior;
2. **intra-class mix** — per present class, pick *k* endmembers and mix them
   (floor-then-Dirichlet weights);
3. **augment** — per-class albedo (brightness) scaling + water glint;
4. **inter-class fractions** — a Dirichlet mix across present classes,
   constrained so each present class ≥ `interclass_min_frac`;
5. **global illumination** — a whole-pixel brightness scalar (default OFF);
6. **noise** — white floor + brightness-scaled covariance noise;
7. **return** `(spectra (207,), fractions (n_classes,))`.

`sim_config.yaml` documents every field and is the place to change this
behavior; each step maps to one config block.
