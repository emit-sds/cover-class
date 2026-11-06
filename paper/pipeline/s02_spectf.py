"""Stage 2: SpecTf presence probabilities (existing spectf_decoder checkpoint).

Thresholds follow the EMIT-FC convention: per-class F1-optimal thresholds fit
on each split itself (sim test on sim truth, OOD on OOD labels). The forced-
fraction sweep uses 0.5, as plot_tpr_by_fraction.py did (no threshold file).

Writes outputs/spectf/pred_{sim_test,ood,forced}.npz and metrics.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C


def run(predict, tag):
    X, F, wl = C.load_sim_test()
    Xo, Yo, _, wlo = C.load_ood()
    # EMIT scene grid sits ~0.15 nm off the library grid; same 207 bands.
    assert np.allclose(wl, wlo, atol=0.5), "OOD and simulator band grids differ"
    ff = np.load(C.out_path("data", "forced_fraction.npz"))

    metrics = {}
    for split, x, truth in [("sim_test", X, (F > 0).astype(float)),
                            ("ood", Xo, C.ood_truth_masked(Yo))]:
        probs = predict(x)
        thr = C.f1_opt_thr(probs, truth)
        pred = (probs >= thr[None]).astype(np.int8)
        np.savez_compressed(C.out_path(tag, f"pred_{split}.npz"),
                            probs=probs, thresholds=thr, presence=pred)
        metrics[split] = dict(thresholds=dict(zip(C.CLASSES, thr.tolist())),
                              **C.presence_metrics(truth, pred))
        print(tag, split, {c: round(m["f1"], 3) for c, m in metrics[split].items() if c in C.CLASSES})

    probs = predict(ff["X"])
    thr = np.full(len(C.CLASSES), 0.5)
    np.savez_compressed(C.out_path(tag, "pred_forced.npz"), probs=probs, thresholds=thr,
                        presence=(probs >= thr[None]).astype(np.int8))
    C.out_path(tag, "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    _, _, wl = C.load_sim_test()
    model, device = C.load_spectf(wl)
    run(lambda x: C.predict_spectf(model, device, x), "spectf")
