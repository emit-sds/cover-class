"""Table 3: validation polygons and spectra per class, by label.

A polygon is one mmgis_id; its label is that of its spectra (constant within a
polygon; checked). Independent of the simulator/model.
Writes tables/table3_validation_summary.{csv,tex}.
"""

import numpy as np
import pandas as pd

import common as C


def main():
    _, Y, ids, _ = C.load_ood()
    uniq, first = np.unique(ids, return_index=True)
    for u in uniq:
        assert (Y[ids == u] == Y[ids == u][0]).all(), f"labels vary within polygon {u}"
    P = Y[first]
    rows = []
    for cls in ["soil", "pv", "npv", "snow+ice", "water"]:
        j = C.CLASSES.index(cls)
        rows.append(dict(cls=C.CLASS_LABELS[cls],
                         **{f"poly_{n}": int((P[:, j] == v).sum()) for n, v in [("present", 1), ("absent", 0), ("uncertain", 2)]},
                         **{f"spec_{n}": int((Y[:, j] == v).sum()) for n, v in [("present", 1), ("absent", 0), ("uncertain", 2)]}))
    df = pd.DataFrame(rows)
    df.to_csv(C.tab_path("table3_validation_summary.csv"), index=False)

    fmt = lambda v: f"{v:,}"
    lines = [f"% {len(uniq)} polygons, {len(Y):,} spectra",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{Polygons} & \multicolumn{3}{c}{Spectra} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Class & Present & Absent & Uncert. & Present & Absent & Uncert. \\", r"\midrule"]
    for r in df.itertuples():
        lines.append(f"{r.cls} & {r.poly_present} & {r.poly_absent} & {r.poly_uncertain} & "
                     f"{fmt(r.spec_present)} & {fmt(r.spec_absent)} & {fmt(r.spec_uncertain)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.tab_path("table3_validation_summary.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(uniq)} polygons, {len(Y)} spectra")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
