"""Side-by-side of the current manuscript numbers (old simulator / old SpecTf)
against the regenerated v2 numbers, for review before dropping tables in.

Reads tables/*.csv (run the tableN.py / figure10.py scripts first) and writes
tables/RESULTS.md.
"""

import json

import numpy as np
import pandas as pd

import common as C

# Values transcribed from ref/latex/main.tex.
OLD_T4 = {  # class: (global tpr, fpr, spectf tpr, fpr, mlp tpr, fpr)
    "npv": (0.64, 0.39, 0.68, 0.08, 0.70, 0.30),
    "pv": (0.80, 0.16, 0.82, 0.06, 0.79, 0.01),
    "snow+ice": (1.00, 0.45, 0.96, 0.05, 0.97, 0.00),
    "soil": (0.99, 0.14, 0.90, 0.05, 0.93, 0.00),
    "water": (1.00, 0.67, 0.99, 0.12, 1.00, 0.02),
}
OLD_T5 = {  # class: mixed (global, spectf, mlp), pure (global, spectf, mlp)
    "soil": (0.23, 0.17, 0.20, 0.45, 0.07, 0.07),
    "pv": (0.18, 0.12, 0.13, 0.19, 0.00, 0.01),
    "npv": (0.18, 0.16, 0.17, 0.45, 0.00, 0.04),
    "snow+ice": (0.18, 0.15, 0.18, 0.51, 0.02, 0.11),
    "water": (0.22, 0.25, 0.25, 0.32, 0.00, 0.02),
}
OLD_F10 = {"Global": dict(frac_1=0.12, frac_ge3=0.57, frac_ge4=0.24, **{"mean@0.05": 2.7, "mean@0.1": 2.2, "mean@0.2": 1.8}),
           "Conditional": dict(frac_1=0.53, frac_le2=0.90, frac_ge4=0.03, **{"mean@0.05": 1.6, "mean@0.1": 1.5, "mean@0.2": 1.35})}
V = ["global", "spectf", "mlp"]


def cell(old, new):
    return f"{old:.2f} → **{new:.2f}**"


def main():
    out = ["# v2 simulator + SpecTf (spectf_decoder_v3) results vs. current manuscript", "",
           "Format: `manuscript → **regenerated**`. OOD = validation_20260908_subsampled "
           "(94 polygons, 18,353 spectra) for all models.", ""]

    t4 = pd.read_csv(C.tab_path("table4_presence.csv")).set_index(["variant", "cls"])
    out += ["## Table 4 — OOD presence (E(MC)² fraction ≥ 0.05)", "",
            "| Class | Global TPR | Global FPR | SpecTF TPR | SpecTF FPR | MLP-NPV TPR | MLP-NPV FPR |",
            "|---|---|---|---|---|---|---|"]
    for cls in C.TABLE_ORDER_PRESENCE:
        new = [t4.loc[(v, cls), k] for v in V for k in ("tpr", "fpr")]
        out.append(f"| {C.CLASS_LABELS[cls]} | " + " | ".join(cell(o, n) for o, n in zip(OLD_T4[cls], new)) + " |")
    out += ["", "Raw classifier presence (before unmixing), for reference:", "",
            "| Class | SpecTF TPR | SpecTF FPR | MLP-NPV TPR | MLP-NPV FPR |", "|---|---|---|---|---|"]
    for cls in C.TABLE_ORDER_PRESENCE:
        out.append(f"| {C.CLASS_LABELS[cls]} | " + " | ".join(
            f"{t4.loc[(v, cls), k]:.2f}" for v in ("spectf", "mlp") for k in ("clf_tpr", "clf_fpr")) + " |")

    t5 = pd.read_csv(C.tab_path("table5_mae.csv")).set_index(["variant", "cls", "bucket"])
    out += ["", "## Table 5 — sim-test MAE", "",
            "| Class | Mixed Global | Mixed SpecTF | Mixed MLP | Pure Global | Pure SpecTF | Pure MLP |",
            "|---|---|---|---|---|---|---|"]
    for cls in C.TABLE_ORDER_MAE:
        new = [t5.loc[(v, cls, b), "mae"] for b in ("mixed", "pure") for v in V]
        out.append(f"| {C.CLASS_LABELS[cls]} | " + " | ".join(cell(o, n) for o, n in zip(OLD_T5[cls], new)) + " |")

    f10 = pd.read_csv(C.tab_path("figure10_class_counts.csv")).set_index("variant")
    out += ["", "## Section 4.3 / Figure 10 — active classes per OOD pixel (SpecTF conditional)", "",
            "| Quantity | Global | Conditional |", "|---|---|---|"]
    for k in ["frac_1", "frac_le2", "frac_ge3", "frac_ge4", "mean@0.05", "mean@0.1", "mean@0.2"]:
        row = []
        for v in ("Global", "Conditional"):
            o = OLD_F10[v].get(k)
            row.append(cell(o, f10.loc[v, k]) if o is not None else f"**{f10.loc[v, k]:.2f}**")
        out.append(f"| {k} | " + " | ".join(row) + " |")

    cov = pd.read_csv(C.tab_path("figure8b_knn_coverage.csv"), index_col=0)
    sn = ["soil", "npv"]
    others = cov.where(~(cov.index.isin(sn)[:, None] & cov.columns.isin(sn)[None]))
    out += ["", "## Figure 8b — OOD k-NN coverage (k=10)", "",
            "| Quantity | Value |", "|---|---|",
            f"| soil→npv | {cell(0.19, cov.loc['soil', 'npv'])} |",
            f"| npv→soil | {cell(0.14, cov.loc['npv', 'soil'])} |",
            f"| max other pair | {cell(0.09, np.nanmax(others.to_numpy()))} |"]

    sel = json.loads(C.out_path("mlp", "selection.json").read_text())
    rows = []
    for s, f1 in sel["sim_mean_f1"].items():
        m = json.loads(C.out_path("mlp", f"seed_{s}", "metrics.json").read_text())
        rows.append((int(s), f1, m["ood"]["mean_f1"]))
    sim, ood = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    out += ["", "## MLP-NPV CV seeds (v2 simulator)", "",
            f"Chosen seed **{sel['chosen_seed']}** (sim mean F1 closest to CV median {sel['median_sim_mean_f1']:.4f}).", "",
            f"Mean F1 across {len(rows)} seeds: sim {sim.mean():.3f} ± {sim.std():.3f}, OOD {ood.mean():.3f} ± {ood.std():.3f}.", "",
            "| Seed | Sim mean F1 | OOD mean F1 |", "|---|---|---|"]
    out += [f"| {s}{' (chosen)' if s == sel['chosen_seed'] else ''} | {a:.4f} | {b:.4f} |" for s, a, b in rows]

    thr = {m: C.load_preds(m, "ood")["thresholds"] for m in ("spectf", "mlp")}
    out += ["", "## OOD presence thresholds (F1-optimal on OOD, as in EMIT-FC)", "",
            "| Class | SpecTF | MLP-NPV |", "|---|---|---|"]
    out += [f"| {C.CLASS_LABELS[c]} | {thr['spectf'][j]:.3f} | {thr['mlp'][j]:.3f} |" for j, c in enumerate(C.CLASSES)]
    C.tab_path("RESULTS.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
