"""Table 1 (and its duplicate, Table 2): reference spectral libraries by class.

Counts come straight from the cleaned endmember HDF5s the v2 simulator reads
(soil = soil + rock). Adds the size of the reduced (v2 subsample, train-split)
library actually used for simulation and E(MC)^2.

Writes tables/table1_libraries.{csv,tex}.
"""

import collections
import os

import h5py
import pandas as pd
import yaml

import common as C


def main():
    with open(C.CFG["sim_config"]) as f:
        sim_cfg = yaml.safe_load(f)
    base = os.path.dirname(C.CFG["sim_config"])
    _, lib_labels, _ = C.load_library()
    reduced = collections.Counter(lib_labels)
    rows = []
    for cls in C.TABLE_ORDER_MAE:
        n, sources = 0, collections.Counter()
        for p in sim_cfg["endmembers"][cls]:
            with h5py.File(os.path.normpath(os.path.join(base, p))) as h:
                n += len(h["spectra"])
                sources.update(s.decode() if isinstance(s, bytes) else s for s in h["source"][:])
        rows.append(dict(cls=C.CLASS_LABELS[cls], n_spectra=n, n_reduced=reduced[cls],
                         sources="; ".join(f"{s} ({k})" for s, k in sources.most_common())))
    df = pd.DataFrame(rows)
    df.to_csv(C.tab_path("table1_libraries.csv"), index=False)

    lines = [r"\begin{tabular}{l r r}", r"\toprule",
             r"Class & $n$ spectra & $n$ reduced (v2) \\", r"\midrule"]
    lines += [f"{r.cls} & {r.n_spectra:,} & {r.n_reduced} \\\\".replace(",", "{,}") for r in df.itertuples()]
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.tab_path("table1_libraries.tex").write_text("\n".join(lines) + "\n")
    print(df[["cls", "n_spectra", "n_reduced"]].to_string(index=False))


if __name__ == "__main__":
    main()
