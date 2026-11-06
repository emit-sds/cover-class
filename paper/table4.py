"""Table 4: OOD presence classification (TPR / FPR per class).

Presence = E(MC)^2 fraction >= 0.05 (EMIT-FC threshold_curve_by_ood_truth),
for global unmixing and for conditional unmixing gated by SpecTf / MLP-NPV.
Uncertain labels are excluded. Also reports the raw classifier presence
(before unmixing) in the CSV for reference.

Writes tables/table4_presence.{csv,tex}.
"""

import numpy as np
import pandas as pd

import common as C

VARIANTS = [("global", "Global E(MC)$^2$"), ("spectf", "SpecTF"), ("mlp", "MLP-NPV")]


def main():
    _, Y, _, _ = C.load_ood()
    truth = C.ood_truth_masked(Y)
    thr = C.CFG["presence_threshold"]
    rows = []
    for v, _ in VARIANTS:
        m = C.presence_metrics(truth, C.load_sma("ood", v) >= thr)
        raw = C.presence_metrics(truth, C.load_preds(v, "ood")["presence"]) if v != "global" else None
        for cls in C.CLASSES:
            rows.append(dict(variant=v, cls=cls, tpr=m[cls]["tpr"], fpr=m[cls]["fpr"], f1=m[cls]["f1"],
                             clf_tpr=raw[cls]["tpr"] if raw else np.nan,
                             clf_fpr=raw[cls]["fpr"] if raw else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(C.tab_path("table4_presence.csv"), index=False)

    piv = df.pivot(index="cls", columns="variant", values=["tpr", "fpr"])
    lines = [r"\begin{tabular}{l cc cc cc}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{lab}}}" for _, lab in VARIANTS) + r" \\",
             r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
             r"Class & TPR & FPR & TPR & FPR & TPR & FPR \\", r"\midrule"]
    for cls in C.TABLE_ORDER_PRESENCE:
        tprs = {v: round(piv.loc[cls, ("tpr", v)], 2) for v, _ in VARIANTS}
        fprs = {v: round(piv.loc[cls, ("fpr", v)], 2) for v, _ in VARIANTS}
        best_t, best_f = max(tprs.values()), min(fprs.values())
        cells = []
        for v, _ in VARIANTS:
            t = f"{tprs[v]:.2f}"
            f = f"{fprs[v]:.2f}"
            cells += [rf"\textbf{{{t}}}" if tprs[v] == best_t else t,
                      rf"\textbf{{{f}}}" if fprs[v] == best_f else f]
        lines.append(f"{C.CLASS_LABELS[cls]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.tab_path("table4_presence.tex").write_text("\n".join(lines) + "\n")
    print(piv.loc[C.TABLE_ORDER_PRESENCE].round(2).to_string())


if __name__ == "__main__":
    main()
