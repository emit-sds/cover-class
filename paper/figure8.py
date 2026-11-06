"""Figure 8: soil/NPV overlap in real EMIT data.

(a) t-SNE of library + OOD soil and NPV spectra only (no simulated), same
    recipe as Figure 7.
(b) Pairwise k-NN coverage (k=10, Naeem et al. 2020) between OOD classes on
    full-resolution reflectance: Coverage(i -> j) is the fraction of class-i
    spectra whose k-NN ball within class i holds at least one class-j
    spectrum. As EMIT-FC tsne_comparison.class_pairwise_coverage
    (source_name="OOD", k=10): StandardScaler fit on all OOD spectra, then
    <=3000 spectra per class (rng seed 0, classes in sorted order). Panel (b)
    depends only on the validation set, not on the simulator or models.

Writes figures/figure8a_tsne_soil_npv_real_only.png and
figure8b_class_coverage_heatmap_ood.png.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import common as C
import figure7 as F7


def knn_coverage(ref, cand, k):
    k = min(k, len(ref) - 1)
    radius = NearestNeighbors(n_neighbors=k + 1).fit(ref).kneighbors(ref)[0][:, -1]
    nearest = NearestNeighbors(n_neighbors=1).fit(cand).kneighbors(ref)[0][:, 0]
    return float((nearest <= radius).mean())


def coverage_matrix(ood_mode, k=10, max_points=3000, seed=0):
    X, Y, _, _ = C.load_ood()
    lab = np.array(C.CLASSES)[F7.ood_classes(Y, ood_mode)]
    Z = StandardScaler().fit_transform(X.astype(float))
    order = sorted(set(lab))
    rng = np.random.default_rng(seed)
    per_class = {}
    for c in order:
        z = Z[lab == c]
        if len(z) > max_points:
            z = z[rng.choice(len(z), max_points, replace=False)]
        per_class[c] = z
    M = pd.DataFrame(np.nan, index=order, columns=order)
    for a in order:
        for b in order:
            if a != b:
                M.loc[a, b] = knn_coverage(per_class[a], per_class[b], k)
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood-class", choices=["argmax", "present"], default="argmax")
    args = ap.parse_args()
    F7.paper_style()
    import seaborn as sns

    src = F7.sources(args.ood_class, classes=["soil", "npv"])
    del src["Simulated"]
    z, cls, s = F7.embed(src)
    fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)
    F7.plot(z, cls, s, ax, legend="inside")
    fig.savefig(C.fig_path("figure8a_tsne_soil_npv_real_only.png"), dpi=300, bbox_inches="tight")

    M = coverage_matrix(args.ood_class)
    M.to_csv(C.tab_path("figure8b_knn_coverage.csv"))
    n = len(M)
    fig, ax = plt.subplots(figsize=(1.1 * n + 2.2, 0.9 * n + 1.8), constrained_layout=True)
    sns.heatmap(M, annot=True, fmt=".2f", cmap="rocket_r", vmin=0, vmax=1, square=True,
                linewidths=0.5, linecolor="white", cbar_kws={"label": "k-NN coverage (k=10)"}, ax=ax)
    ax.set_xlabel("candidate class (nearby neighbor)"), ax.set_ylabel("reference class")
    fig.savefig(C.fig_path("figure8b_class_coverage_heatmap_ood.png"), dpi=300, bbox_inches="tight")
    print(M.round(2))


if __name__ == "__main__":
    main()
