"""Stage 1: simulated test sets and the SMA endmember library (v2 simulator).

Writes outputs/data/:
  sim_test.npz           held-out mixtures from the *val* endmember split
                         (X, F fractions, wavelengths); Tables 5, Figure 7
  forced_fraction.npz    stratified forced-fraction set (Figure 9)
  unmixing_library.csv   reduced *train*-split endmembers (what the classifiers
                         were trained from); the E(MC)^2 library
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C


def write_library():
    sim = C.make_simulator("train")
    rows, labels = [], []
    for cls in C.CLASSES:
        rows.append(sim.library[cls])
        labels += [cls] * len(sim.library[cls])
    df = pd.DataFrame(np.concatenate(rows), columns=[str(w) for w in sim.wavelengths])
    df["label"] = labels
    df.to_csv(C.out_path("data", "unmixing_library.csv"), index=False)
    print("library:", df["label"].value_counts().to_dict())


def write_sim_test():
    X, F, wl = C.simulate_n("val", C.CFG["test_base_seed"], C.CFG["sim_test_size"])
    np.savez_compressed(C.out_path("data", "sim_test.npz"), X=X, F=F, wavelengths=wl)
    print("sim test:", X.shape, "pure frac per class:", (F == 1).mean(0).round(3))


def _forced_mixture(sim, rng, ci, f):
    """One v2 mixture with class ci forced to fraction f (f == 1 -> pure pixel).

    Mirrors MixtureSimulator.simulate, except the presence pattern is drawn
    from the prior restricted to patterns containing ci (and feasible given f
    with every other present class >= interclass_min_frac), and ci's fraction
    is fixed instead of Dirichlet-drawn.
    """
    floor = sim.interclass_min_frac
    pats, k = sim._patterns, sim._patterns.sum(1)
    ok = pats[:, ci] & ((k == 1) if f >= 1 else ((k >= 2) & ((k - 1) * floor <= 1 - f + 1e-9)))
    p = sim._probs * ok
    present = np.flatnonzero(pats[rng.choice(len(p), p=p / p.sum())])

    class_spectra = np.stack([sim._augment_class(sim.classes[j], sim._intra_class_spectrum(sim.classes[j], rng), rng)
                              for j in present])
    fracs = np.zeros(len(present))
    me = present == ci
    fracs[me] = f
    m = len(present) - 1
    if m:
        lo, hi = sim.interclass_alpha
        alpha = np.full(m, lo if lo == hi else rng.uniform(lo, hi))
        for _ in range(100):
            d = (1 - f) * rng.dirichlet(alpha)
            if (d >= floor).all():
                break
        else:
            d = floor + (1 - f - m * floor) * rng.dirichlet(alpha)
        fracs[~me] = d
    pixel = fracs @ class_spectra
    g_lo, g_hi = sim.global_illumination
    if not (g_lo == g_hi == 1.0):
        from specmix.augment import albedo_augment
        pixel = albedo_augment(pixel, g_lo, g_hi, sim.magnitude_max, rng=rng)
    pixel = pixel + sim.noise.sample(pixel.astype(np.float32), rng)
    full = np.zeros(sim.n_classes, np.float32)
    full[present] = fracs
    return pixel.astype(np.float32), full


def write_forced():
    ff = C.CFG["forced_fraction"]
    sim = C.make_simulator("val")
    w, n = ff["bin_width"], ff["n_per_bin"]
    n_bins = int(round(1 / w))
    X, F, cls, frac = [], [], [], []
    for ci in range(sim.n_classes):
        for b in range(n_bins):
            rng = np.random.default_rng(np.random.SeedSequence([ff["base_seed"], ci, b]))
            lo, hi = b * w, (b + 1) * w
            # Mixed pixels need every other class >= min frac, so the top bin
            # (1 - min_frac, 1] can only be filled by pure pixels.
            pure = lo >= 1 - sim.interclass_min_frac - 1e-9
            for _ in range(n):
                f = 1.0 if pure else rng.uniform(max(lo, ff["min_fraction"]), min(hi, 1 - sim.interclass_min_frac))
                x, fr = _forced_mixture(sim, rng, ci, f)
                X.append(x), F.append(fr), cls.append(ci), frac.append(f)
    np.savez_compressed(C.out_path("data", "forced_fraction.npz"), X=np.stack(X), F=np.stack(F),
                        forced_class=np.array(cls), forced_fraction=np.array(frac),
                        wavelengths=sim.wavelengths)
    print("forced fraction set:", len(X))


if __name__ == "__main__":
    write_library()
    write_sim_test()
    write_forced()
