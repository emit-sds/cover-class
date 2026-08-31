# models — fractional-cover unmixing model

Train and evaluate a [SpecTf](https://github.com/emit-sds) encoder that consumes
`specmix` simulated mixtures and predicts, for the 5 classes
(**PV, NPV, Soil, Snow+Ice, Water**), either:

- **classification** — which classes are present (multi-label, sigmoid head), or
- **regression** — the per-class fractions (softmax head).

There is one training entrypoint; the task is chosen by the config (or `--task`).
Training draws mixtures from `specmix.MixtureDataset` and validates on both a
cached simulation-val split and the real **OOD** expert-labeled set.

## Setup

Install the simulator package (from the repo root) and its Torch extra:

```bash
pip install -e ".[torch]"      # specmix + PyTorch (see the top-level README)
```

Run commands from the **repo root** so relative paths in the configs resolve
(`datasets/`, `src/specmix/sim_config.yaml`). On this Mac, training uses **MPS**
via `spectf.utils.get_device()`.

## Train

```bash
# classification (presence) — the config default
python -m models.train --train-config models/train_config.yaml

# regression (fractions)
python -m models.train --train-config models/train_config.yaml --task regression
```

Common overrides (all optional; each falls back to the config):

```bash
python -m models.train --train-config models/train_config.yaml \
    --task regression \
    --epochs 1000 --steps-per-epoch 200 \
    --sim-val-size 2000 --num-workers 8 \
    --wandb-mode offline --outdir models/runs
```

Each run writes to `outdir` (default `models/runs/`, gitignored):

- `model_epoch<N>.pth` — checkpoints (already the schedule-free **averaged**,
  deployable weights; no optimizer state to strip).
- `SpecTfEncoder_<timestamp>_<task>.md` / `.json` — metric tables.
- ROC / confusion / regression figures (PNG).

Per epoch, scalar metrics are logged to W&B for three regimes:

- **sim-val** at its own F1-optimal thresholds,
- **OOD** at the sim thresholds (transfer), and
- **OOD** at its own F1-optimal thresholds (the OOD threshold shifts a lot).

Set `wandb.mode: disabled` in the config (or `--wandb-mode disabled`) to run
without W&B.

## Evaluate

Score a saved checkpoint — no training, no W&B. Both eval scripts reuse
`engine.py`, so their numbers match the end-of-training report exactly.

**On the sim-val + OOD sets** (same configs as training):

```bash
python -m models.eval --train-config models/train_config.yaml \
    --weights models/runs/<run>/model_epoch1000.pth --outdir models/eval_out
```

Pass `--task` if the checkpoint's head differs from the config default.

**On the Francisco set** (in-situ pv/npv/soil fractions for real EMIT pixels):

```bash
python -m models.eval_francisco --train-config models/train_config.yaml \
    --weights models/runs/<run>/model_epoch1000.pth --task regression
```

Regression is the primary use (per-class MAE/RMSE/R² + a predicted-vs-true
figure on RAW, non-renormalized pv/npv/soil). Classification is reported with a
caveat: every Francisco plot contains all three classes, so there are no true
negatives — only TPR (recall) is meaningful. `--task both` runs whichever
apply to the checkpoint's head.

> Note: Francisco fractions are E(MC)² unmixing-model *output*, not measured
> field truth — "agreement with Francisco" means agreement with another model,
> not physical ground truth.

## Configuration

- **`train_config.yaml`** — task, batch size, sim-val size, presence threshold
  (`min_frac`), endmember train/val split, loss (focal α/γ), training loop
  (LR, epochs, steps/epoch, `resample_each_epoch`), W&B. Paths are resolved
  relative to the config file, so it is location-stable.
- **`model_config.yaml`** — SpecTf architecture (`spectf.model.SpecTfEncoder`).
  `dim_output` is set from the class count at runtime.

`resample_each_epoch: true` advances the simulator seed each epoch so the model
sees fresh mixtures every epoch (unique spectra = `epochs × steps × batch`).

## Layout

```
train.py            training entrypoint (--task classification|regression)
eval.py             score a .pth on sim-val + OOD
eval_francisco.py   score a .pth on the Francisco pv/npv/soil set
engine.py           shared primitives (model build/load, infer, sim/OOD metrics)
                      — used by train + both eval scripts so numbers never drift
metrics.py          F1-opt threshold, binary/regression metrics, figures (numpy+sklearn)
losses.py           FocalLoss (presence), FocalCategoricalCrossEntropy (fractions)
ood_dataset.py      load/remap the OOD expert-labeled set to the sim class axis
francisco_dataset.py load Francisco EMIT pixels + broadcast per-plot fractions
reporting.py        markdown/txt tables + PNG figure export
train_config.yaml   task + loop + W&B config
model_config.yaml   SpecTf architecture
ref/                original cover_class-based scripts (reference only; do not run)
```

## Class-axis alignment

The sim class axis is `[water, pv, npv, soil, snow+ice]`. The model's band
definition comes from `MixtureSimulator.wavelengths` (207 good bands),
guaranteeing it matches the simulated spectra and the OOD grid. OOD ambiguous
labels (`2`) are masked to NaN and excluded from metrics.
