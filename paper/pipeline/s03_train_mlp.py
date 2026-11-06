"""Stage 3: train MLP-NPV on the v2 simulator over the CV seeds, pick the final
seed, and save its presence probabilities.

Recipe follows EMIT-FC train_mlp.py: AdamW (lr 3e-4, wd 1e-4), 5000 steps of
batch 64, BCE on the combined logits + 0.3 x BCE on the full auxiliary NPV
branch, labels = class present (fraction > 0). Each seed draws a fresh
mixture stream from the *train* endmember split. The final seed is the one
whose mean sim-test F1 is closest to the CV median (EMIT-FC final_model.py).

Writes outputs/mlp/seed_<s>/{model.pt,metrics.json,loss.csv}, selection.json,
and pred_{sim_test,ood,forced}.npz for the chosen seed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C
from s02_spectf import run as save_predictions


def train_seed(seed, workers):
    m = C.CFG["mlp"]
    d = C.out_path("mlp", f"seed_{seed}", "model.pt").parent
    if (d / "model.pt").exists():
        print(f"seed {seed}: exists, skipping training")
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = C.make_dataset("train", seed, m["max_steps"] * m["batch_size"])
    from specmix import numpy_collate
    dl = DataLoader(ds, batch_size=m["batch_size"], shuffle=False, num_workers=workers,
                    collate_fn=numpy_collate, drop_last=True)
    Xt, Ft, _ = C.load_sim_test()
    Xt, Yt = torch.tensor(Xt), torch.tensor((Ft > 0).astype(np.float32))

    model = C.build_mlp(ds.sim.n_bands)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=m["learning_rate"], weight_decay=m["weight_decay"])
    log, ave = [], None
    model.train()
    for step, (xb, fb) in enumerate(dl):
        yb = (fb > 0).float()
        opt.zero_grad(set_to_none=True)
        logits, aux = model(xb, return_aux=True)
        loss = crit(logits, yb) + m["aux_weight"] * crit(aux, yb)
        loss.backward()
        opt.step()
        ave = loss.item() if ave is None else 0.95 * ave + 0.05 * loss.item()
        if step % 500 == 0 or step == m["max_steps"] - 1:
            model.eval()
            with torch.no_grad():
                tl = float(crit(model(Xt), Yt))
            model.train()
            log.append(dict(step=step, train=ave, test=tl))
            print(f"seed {seed} step {step:5d} train {ave:.4f} test {tl:.4f}", flush=True)
    model.eval()
    torch.save(model.state_dict(), d / "model.pt")
    pd.DataFrame(log).to_csv(d / "loss.csv", index=False)


def eval_seed(seed):
    model, _ = C.load_mlp(seed)
    X, F, _ = C.load_sim_test()
    Xo, Yo, _, _ = C.load_ood()
    res = {}
    for split, x, truth in [("sim_test", X, (F > 0).astype(float)), ("ood", Xo, C.ood_truth_masked(Yo))]:
        p = C.predict_mlp(model, x)
        thr = C.f1_opt_thr(p, truth)
        res[split] = C.presence_metrics(truth, (p >= thr[None]).astype(int))
        res[split]["mean_f1"] = float(np.mean([res[split][c]["f1"] for c in C.CLASSES]))
    C.out_path("mlp", f"seed_{seed}", "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(4)
    seeds = C.cv_seeds()
    scores = {}
    for s in seeds:
        train_seed(s, args.workers)
        r = eval_seed(s)
        scores[s] = r["sim_test"]["mean_f1"]
        print(f"seed {s}: sim mean F1 {scores[s]:.4f}  ood mean F1 {r['ood']['mean_f1']:.4f}")
    med = float(np.median(list(scores.values())))
    chosen = min(seeds, key=lambda s: abs(scores[s] - med))
    C.out_path("mlp", "selection.json").write_text(json.dumps(
        dict(chosen_seed=chosen, median_sim_mean_f1=med, sim_mean_f1=scores), indent=2))
    print("chosen seed", chosen)
    model, _ = C.load_mlp(chosen)
    save_predictions(lambda x: C.predict_mlp(model, x), "mlp")
