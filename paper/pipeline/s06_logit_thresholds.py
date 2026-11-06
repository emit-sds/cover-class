"""Stage 6 (optional): F1-optimal thresholds picked on pre-sigmoid logits.

SpecTf's sigmoid saturates in float32 (e.g. ~28% of OOD soil probabilities are
exactly 1.0), so the probability-space F1 search ties those pixels and can
land on a threshold of 1.0, leaving pixels with no predicted class (which
then fall back to global unmixing). Thresholding logits breaks the ties. For a
non-saturating model this is a monotone no-op.

Writes outputs/<model>_logit/pred_{sim_test,ood}.npz (probs = logits) and
prints how many pixels' predicted subsets change vs. probability thresholds.
With --run-sma, also runs conditional E(MC)^2 for those subsets ->
outputs/sma/<split>/<model>_logit/.

Result (2026-10-18): only 6 OOD pixels (SpecTf, soil) and 0 MLP pixels change,
so this does not move Table 4 -- the F1 optimum genuinely sits in the saturated
tail (SpecTf soil logit threshold ~16.7), not at an artificial tie.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C
from s04_sma import data


def save(tag, predict):
    X, F, _ = C.load_sim_test()
    Xo, Yo, _, _ = C.load_ood()
    for split, x, truth in [("sim_test", X, (F > 0).astype(float)), ("ood", Xo, C.ood_truth_masked(Yo))]:
        z = predict(x)
        thr = C.f1_opt_thr(z, truth)
        pred = (z >= thr[None]).astype(np.int8)
        np.savez_compressed(C.out_path(tag, f"pred_{split}.npz"), probs=z, thresholds=thr, presence=pred)
        print(tag, split, "logit thr", thr.round(2), "empty subsets", round(float((pred.sum(1) == 0).mean()), 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["spectf", "mlp"])
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--run-sma", action="store_true")
    args = ap.parse_args()
    _, _, wl = C.load_sim_test()
    for m in args.models:
        if m == "spectf":
            model, device = C.load_spectf(wl)
            save("spectf_logit", lambda x: C.predict_spectf(model, device, x, logits=True))
        else:
            model, _ = C.load_mlp()
            save("mlp_logit", lambda x: C.predict_mlp(model, x, logits=True))
        a = C.load_preds(m, "ood")["presence"]
        b = C.load_preds(f"{m}_logit", "ood")["presence"]
        print(f"{m}: {(a != b).any(1).sum()} OOD pixels change subset; per class {dict(zip(C.CLASSES, (a != b).sum(0)))}")
    if not args.run_sma:
        sys.exit(0)
    for split in ("sim_test", "ood"):
        X, wl_s, truth = data(split)
        for m in args.models:
            v = f"{m}_logit"
            C.run_sma(X, wl_s, truth, C.out_path("sma", split, v, "x").parent,
                      subsets=C.subsets_from_presence(C.load_preds(v, split)["presence"]), n_jobs=args.n_jobs)
