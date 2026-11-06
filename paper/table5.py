"""Table 5: fractional cover MAE on the simulated test set.

Per class, over pixels where the class is truly present: mixed (0 < x < 1)
and pure (x = 1), for global vs SpecTf- / MLP-NPV-conditioned E(MC)^2
(EMIT-FC sma.compute_mae buckets "0-1" and "1").

Writes tables/table5_mae.{csv,tex}.
"""

import numpy as np
import pandas as pd

import common as C

VARIANTS = [("global", "Global E(MC)$^2$"), ("spectf", "SpecTF"), ("mlp", "MLP-NPV")]


def main():
    _, F, _ = C.load_sim_test()
    rows = []
    for v, _ in VARIANTS:
        P = C.load_sma("sim_test", v)
        for j, cls in enumerate(C.CLASSES):
            t, p = F[:, j], P[:, j]
            for bucket, m in [("mixed", (t > 0) & (t < 1)), ("pure", t == 1)]:
                rows.append(dict(variant=v, cls=cls, bucket=bucket, n=int(m.sum()),
                                 mae=float(np.abs(t[m] - p[m]).mean()) if m.any() else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(C.tab_path("table5_mae.csv"), index=False)

    piv = df.pivot_table(index="cls", columns=["bucket", "variant"], values="mae")
    lines = [r"\begin{tabular}{l ccc ccc}", r"\toprule",
             r" & \multicolumn{3}{c}{Mixed-pixel MAE ($0<x<1$)} & \multicolumn{3}{c}{Pure-pixel MAE ($x=1$)} \\",
             r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
             "Class & " + " & ".join(lab for _, lab in VARIANTS) * 1 + " & " + " & ".join(lab for _, lab in VARIANTS) + r" \\",
             r"\midrule"]
    for cls in C.TABLE_ORDER_MAE:
        cells = []
        for bucket in ("mixed", "pure"):
            vals = {v: round(piv.loc[cls, (bucket, v)], 2) for v, _ in VARIANTS}
            best = min(vals.values())
            cells += [rf"\textbf{{{vals[v]:.2f}}}" if vals[v] == best else f"{vals[v]:.2f}" for v, _ in VARIANTS]
        lines.append(f"{C.CLASS_LABELS[cls]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.tab_path("table5_mae.tex").write_text("\n".join(lines) + "\n")
    print(piv.loc[C.TABLE_ORDER_MAE].round(2).to_string())


if __name__ == "__main__":
    main()
