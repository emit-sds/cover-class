"""Stage 4: E(MC)^2 unmixing (nmc30_nend25_bvls) on the sim test set and OOD.

Variants per split:
  global  all five classes for every pixel ("Global E(MC)^2", *_full)
  spectf  restricted to SpecTf-predicted classes (conditional)
  mlp     restricted to MLP-NPV-predicted classes (conditional)

Conditional pixels with one predicted class get fraction 1.0 without unmixing;
pixels with none fall back to all classes (unmix_pipeline_v2.jl).
Writes outputs/sma/<split>/<variant>/predicted_fraction.csv.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C


def data(split):
    if split == "sim_test":
        X, F, wl = C.load_sim_test()
        return X, wl, F
    X, Y, _, wl = C.load_ood()
    return X, wl, C.ood_truth_masked(Y)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["global", "spectf", "mlp"])
    ap.add_argument("--splits", nargs="+", default=["sim_test", "ood"])
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()
    for split in args.splits:
        X, wl, truth = data(split)
        for v in args.variants:
            subsets = None if v == "global" else C.subsets_from_presence(C.load_preds(v, split)["presence"])
            print(f"SMA {split}/{v}: {len(X)} pixels", flush=True)
            C.run_sma(X, wl, truth, C.out_path("sma", split, v, "x").parent, subsets=subsets, n_jobs=args.n_jobs)
