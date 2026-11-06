"""Shared helpers for the manuscript reproduction scripts.

Everything the figure/table scripts need lives here: config + paths, the v2
(specmix) simulators, the OOD validation polygons, the two presence classifiers
(SpecTf label-query encoder, MLP-NPV), threshold calibration, and the Julia
E(MC)^2 runner. Pipeline stages under pipeline/ write to outputs/; the
figureN.py / tableN.py scripts only read from outputs/.
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

PAPER_DIR = Path(__file__).resolve().parent
REPO_DIR = PAPER_DIR.parent
CONFIG_PATH = PAPER_DIR / "configs" / "paper.yaml"

# specmix (v2 simulator) is imported from this repo's src/, not whatever copy
# happens to be pip-installed in the env (the spectf env points at
# cover-class-new, which lacks mixture_priors_v2.csv).
sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))


# ------------------------------------------------------------------ config --
def _resolve(p, base):
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else (base / p).resolve()


def load_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    base = CONFIG_PATH.parent
    for k in ("sim_config", "ood_path", "geojson_path", "outputs", "figures", "tables"):
        cfg[k] = _resolve(cfg[k], base)
    for k in ("model_config", "weights"):
        cfg["spectf"][k] = _resolve(cfg["spectf"][k], base)
    for k in ("julia", "julia_project", "julia_depot", "script"):
        cfg["sma"][k] = _resolve(cfg["sma"][k], base)
    return cfg


CFG = load_config()
CLASSES = list(CFG["classes"])
OUT = CFG["outputs"]
FIG_DIR = CFG["figures"]
TAB_DIR = CFG["tables"]

# Manuscript class palette (EMIT-FC plot_scene.CLASS_COLOR_MAP).
CLASS_COLORS = {
    "water": "#5B9BD5",
    "pv": "#6FAE5C",
    "npv": "#D8B463",
    "soil": "#A9764F",
    "snow+ice": "#7FC3C9",
}
CLASS_LABELS = {"soil": "Soil", "pv": "PV", "npv": "NPV",
                "snow+ice": "Snow/ice", "water": "Water"}
# Row order used by the manuscript tables.
TABLE_ORDER_PRESENCE = ["npv", "pv", "snow+ice", "soil", "water"]
TABLE_ORDER_MAE = ["soil", "pv", "npv", "snow+ice", "water"]


def soften(hex_color, factor=0.12):
    """Blend toward white, as EMIT-FC get_class_colors does for line/map plots."""
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(c + (255 - c) * factor):02X}" for c in rgb)


def out_path(*parts):
    p = OUT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def fig_path(name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR / name


def tab_path(name):
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    return TAB_DIR / name


# --------------------------------------------------------------- simulator --
def make_simulator(split):
    from specmix import MixtureSimulator
    return MixtureSimulator(str(CFG["sim_config"]), split=split,
                            split_frac=CFG["split_frac"], split_seed=CFG["split_seed"])


def make_dataset(split, base_seed, epoch_size):
    from specmix import MixtureDataset
    return MixtureDataset(str(CFG["sim_config"]), epoch_size=epoch_size,
                          base_seed=base_seed, split=split,
                          split_frac=CFG["split_frac"], split_seed=CFG["split_seed"])


def simulate_n(split, base_seed, n):
    """Deterministic (X, F) of n mixtures; item i == MixtureDataset(...)[i]."""
    ds = make_dataset(split, base_seed, n)
    X = np.empty((n, ds.sim.n_bands), np.float32)
    F = np.empty((n, ds.sim.n_classes), np.float32)
    for i in range(n):
        X[i], F[i] = ds[i]
    return X, F, ds.sim.wavelengths


def load_sim_test():
    d = np.load(out_path("data", "sim_test.npz"))
    return d["X"], d["F"], d["wavelengths"]


def load_library():
    """The reduced (train-split) endmember library used for SMA."""
    df = pd.read_csv(out_path("data", "unmixing_library.csv"))
    wl_cols = [c for c in df.columns if c != "label"]
    return df[wl_cols].to_numpy(np.float32), df["label"].to_numpy(), np.array(wl_cols, float)


# --------------------------------------------------------------------- OOD --
def load_ood(path=None):
    """(X, labels 0/1/2 in CLASSES order, mmgis_id, wavelengths)."""
    path = path or CFG["ood_path"]
    with h5py.File(path, "r") as f:
        X = f["spectra"][:].astype(np.float32)
        raw = f["labels"][:]
        file_classes = [c.decode() if isinstance(c, bytes) else str(c)
                        for c in f.attrs["classes"]]
        ids = np.array([m.decode() if isinstance(m, bytes) else str(m)
                        for m in f["mmgis_id"][:]])
        wl = np.asarray(f.attrs["good_wavelengths"], float).ravel()
    Y = raw[:, [file_classes.index(c) for c in CLASSES]].astype(np.float32)
    return X, Y, ids, wl


def ood_truth_masked(Y):
    """OOD labels with uncertain (2) -> NaN."""
    Y = Y.astype(float).copy()
    Y[Y == 2] = np.nan
    return Y


# -------------------------------------------------------------- thresholds --
def f1_opt_thr(probs, truth):
    """Per-class F1-optimal probability threshold (EMIT-FC analysis.f1_opt_thr).

    truth: 0/1 with NaN to ignore. Classes with <2 truth values -> 1.0.
    """
    from sklearn.metrics import precision_recall_curve
    thr = []
    for c in range(truth.shape[1]):
        v = np.isfinite(truth[:, c]) & np.isfinite(probs[:, c])
        yt, yp = truth[v, c].astype(int), probs[v, c]
        if yt.size == 0 or np.unique(yt).size < 2 or np.unique(yp).size < 2:
            thr.append(1.0)
            continue
        p, r, t = precision_recall_curve(yt, yp)
        p, r = p[:-1], r[:-1]
        fb = 2 * p * r / (p + r + 1e-12)
        thr.append(float(t[int(np.nanargmax(fb))]) if t.size else 1.0)
    return np.array(thr)


def presence_metrics(truth, pred):
    """Per-class TPR/FPR/F1 dict; truth NaN entries ignored."""
    out = {}
    for c, name in enumerate(CLASSES):
        v = np.isfinite(truth[:, c])
        yt, yp = truth[v, c].astype(bool), pred[v, c].astype(bool)
        tp, fp = int((yt & yp).sum()), int((~yt & yp).sum())
        fn, tn = int((yt & ~yp).sum()), int((~yt & ~yp).sum())
        tpr = tp / (tp + fn) if tp + fn else np.nan
        fpr = fp / (fp + tn) if fp + tn else np.nan
        prec = tp / (tp + fp) if tp + fp else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
        out[name] = dict(tpr=tpr, fpr=fpr, precision=prec, f1=f1,
                         tp=tp, fp=fp, fn=fn, tn=tn)
    return out


# ------------------------------------------------------------------ SpecTf --
def torch_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_spectf(wavelengths, device=None):
    import torch
    from models.engine import build_model, load_weights
    device = device or torch_device()
    with open(CFG["spectf"]["model_config"]) as f:
        model_cfg = yaml.safe_load(f)
    model = build_model(model_cfg, wavelengths, len(CLASSES), device)
    load_weights(model, CFG["spectf"]["weights"], device)
    model.eval()
    return model, device


def predict_spectf(model, device, X, batch_size=128, logits=False):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device).unsqueeze(-1)
            z = model(xb)
            out.append((z if logits else torch.sigmoid(z)).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------- MLP --
def build_mlp(input_dim):
    from mlp_npv import SpectralMLPClassifierNPV
    m = CFG["mlp"]
    return SpectralMLPClassifierNPV(
        input_dim=input_dim, num_classes=len(CLASSES), npv_idx=CLASSES.index("npv"),
        hidden_layers=m["hidden_layers"], start_width=m["start_width"],
        width_decay=m["width_decay"], min_width=m["min_width"], dropout=m["dropout"],
        npv_hidden_layers=m["npv_hidden_layers"], npv_start_width=m["npv_start_width"],
        npv_width_decay=m["npv_width_decay"], npv_min_width=m["npv_min_width"],
        npv_dropout=m["npv_dropout"])


def cv_seeds():
    m = CFG["mlp"]
    rng = np.random.default_rng(m["main_seed"])
    return [int(s) for s in rng.choice(10_000, m["num_cv_seeds"], replace=False)]


def load_mlp(seed=None):
    import torch
    if seed is None:
        seed = json.loads(out_path("mlp", "selection.json").read_text())["chosen_seed"]
    model = build_mlp(207)
    model.load_state_dict(torch.load(out_path("mlp", f"seed_{seed}", "model.pt"),
                                     map_location="cpu"))
    model.eval()
    return model, seed


def predict_mlp(model, X, batch_size=4096, logits=False):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            z = model(torch.tensor(X[i:i + batch_size]))
            out.append((z if logits else torch.sigmoid(z)).numpy())
    return np.concatenate(out)


# --------------------------------------------------------------- E(MC)^2 --
def write_sma_csv(path, X, wavelengths, truth):
    df = pd.DataFrame(X, columns=[str(w) for w in wavelengths])
    truth = np.asarray(truth, float).copy()
    truth[~np.isfinite(truth)] = 2.0
    df[CLASSES] = truth
    df.to_csv(path, index=False)


def run_sma(X, wavelengths, truth, out_dir, subsets=None, n_jobs=None, chunk=None):
    """Unmix X with unmix_pipeline_v2.jl, split across parallel Julia processes.

    subsets: per-pixel list of predicted class names (conditional unmixing), or
    None for global unmixing over all classes. Results do not depend on the
    chunking: the fork reseeds its RNG per pixel/MC draw. Returns fractions
    (N, n_classes) in CLASSES order and writes predicted_fraction.csv.
    """
    sma = CFG["sma"]
    out_dir = Path(out_dir)
    final = out_dir / "predicted_fraction.csv"
    if final.exists():
        return pd.read_csv(final)[CLASSES].to_numpy()
    n_jobs = n_jobs or sma["n_jobs"]
    n = len(X)
    chunk = chunk or int(np.ceil(n / n_jobs))
    lib = out_path("data", "unmixing_library.csv")
    work = out_dir / "chunks"
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, JULIA_DEPOT_PATH=str(sma["julia_depot"]))

    def job(k):
        lo, hi = k * chunk, min(n, (k + 1) * chunk)
        d = work / f"{k:04d}"
        if (d / "predicted_fraction.csv").exists():
            return d
        d.mkdir(exist_ok=True)
        write_sma_csv(d / "input.csv", X[lo:hi], wavelengths, truth[lo:hi])
        cmd = [str(sma["julia"]), f"--project={sma['julia_project']}", str(sma["script"]),
               "--sim_file", str(d / "input.csv"), "--out_dir", str(d),
               "--library_file", str(lib), "--class_valid_keys", ",".join(CLASSES)]
        for key, val in sma["kwargs"].items():
            cmd += [f"--{key}", str(val)]
        if subsets is None:
            cmd += ["--force_full_classes", "true"]
        else:
            (d / "class_subset.json").write_text(json.dumps(subsets[lo:hi]))
            cmd += ["--force_full_classes", "false", "--class_subset_file", str(d / "class_subset.json")]
        with open(d / "julia.log", "w") as log:
            subprocess.run(cmd, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
        return d

    n_chunks = int(np.ceil(n / chunk))
    with ThreadPoolExecutor(n_jobs) as ex:
        dirs = list(ex.map(job, range(n_chunks)))
    df = pd.concat([pd.read_csv(d / "predicted_fraction.csv") for d in dirs], ignore_index=True)
    assert len(df) == n, (len(df), n)
    df.to_csv(final, index=False)
    return df[CLASSES].to_numpy()


def subsets_from_presence(pred):
    return [[CLASSES[j] for j in np.flatnonzero(row)] for row in pred]


def load_sma(split, variant):
    """variant: 'global' | 'spectf' | 'mlp' | '<model>_logit'. Returns fractions (N, C)."""
    return pd.read_csv(out_path("sma", split, variant, "predicted_fraction.csv"))[CLASSES].to_numpy()


def load_preds(model, split):
    """Saved classifier probabilities / thresholds / presence for a split."""
    d = np.load(out_path(model, f"pred_{split}.npz"))
    return {k: d[k] for k in d.files}


def mpl_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
