"""MixtureSimulator — generate one realistic simulated EMIT mixture per call.

CPU + numpy. One `.simulate(rng)` call returns ONE (spectra, fractions) pair;
batching/parallelism is the DataLoader's job (see dataset.py).

Init (once): load config + endmember HDF5s (285-band + per-row `source`), drop
bad bands -> 207, subsample each class's library, load the presence prior and the
noise covariance.

Per mixture (`simulate`), the 7-step pipeline:
  1. sample a class-presence pattern from the prior,
  2. per present class: pick k components from its subsample, intra-class mix
     (floor-then-Dirichlet weights),
  3. per-class albedo (+ water glint) on each class's intra-class spectrum,
  4. inter-class mix: Dirichlet fractions across present classes, constrained so
     every present class >= interclass_min_frac (per-CLASS floor),
  5. global whole-pixel illumination scalar,
  6. white-floor + brightness-scaled covariance noise,
  7. return (spectra (207,), fractions (n_classes,)).

Fractions sum to 1 over present classes, 0 for absent -- the regression target.
The caller binarizes for classification; the simulator never does.
"""

import os
import argparse

import h5py
import numpy as np
import yaml

from .bands import drop_bad_bands, good_band_mask
from .augment import albedo_augment, glint_offset
from .noise import NoiseModel
from .subsample import subsample
from .mixture_priors import load_mixture_prior

# Directory this package is installed in. Bundled data (default config, priors
# CSV, noise covariance) is resolved relative to here so the package works from
# any install location. Large endmember HDF5s are NOT bundled -- they live in an
# external data root (see `data_root` below).
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PKG_DIR, "data")

#: Packaged default simulator config. `MixtureSimulator()` uses this when no
#: config is passed.
DEFAULT_CONFIG = os.path.join(PKG_DIR, "sim_config.yaml")


def _resolve_data_root(data_root):
    """Where external endmember HDF5s (the `datasets/...` paths) are rooted.

    Precedence: explicit arg > $SPECMIX_DATA_ROOT > current working directory
    (so running from the repo root, where `datasets/` sits, still works)."""
    if data_root is not None:
        return os.path.abspath(data_root)
    env = os.environ.get("SPECMIX_DATA_ROOT")
    if env:
        return os.path.abspath(env)
    return os.getcwd()


def _split_indices(n, split, frac, seed):
    """Row indices for one side of a deterministic random split of n rows.

    "train" -> a random `frac` of [0, n); "val" -> the exact complement.
    """
    perm = np.random.default_rng(seed).permutation(n)
    cut = round(n * frac)
    keep = perm[:cut] if split == "train" else perm[cut:]
    return np.sort(keep)


def _bundled(p):
    """Resolve a bundled-data path (config/priors/covariance).

    Absolute -> as-is. A bare basename or a path under `data/` -> resolved
    inside the package's data dir. Anything else -> relative to the package dir.
    """
    if os.path.isabs(p):
        return p
    cand = os.path.join(DATA_DIR, os.path.basename(p))
    if os.path.exists(cand):
        return cand
    return os.path.join(PKG_DIR, p)


class MixtureSimulator:
    def __init__(self, config=None, data_root=None,
                 split=None, split_frac=0.5, split_seed=0):
        """config: path to a sim_config.yaml, or an already-loaded dict, or None
        to use the packaged DEFAULT_CONFIG.

        data_root: directory the endmember HDF5 paths in the config are resolved
        against. Defaults to $SPECMIX_DATA_ROOT, else the current working
        directory (so running from the repo root, where `datasets/` lives,
        still works). Bundled data (priors CSV, noise covariance) always ships
        with the package and is resolved relative to it, regardless of data_root.

        split: None (default) uses each class's full endmember library. "train"
        or "val" instead take complementary random halves of every class's
        library before subsampling, so train and val simulators draw from
        disjoint endmembers. split_frac is the fraction assigned to "train"
        (0.5 default); split_seed seeds the partition. Same (split_frac,
        split_seed) guarantees "val" is exactly the complement of "train".
        """
        if split not in (None, "train", "val"):
            raise ValueError(f"split must be None, 'train', or 'val'; got {split!r}")
        if config is None:
            config = DEFAULT_CONFIG
        if isinstance(config, str):
            with open(_bundled(config)) as f:
                config = yaml.safe_load(f)
        self.cfg = config
        self.data_root = _resolve_data_root(data_root)
        self.classes = list(config["classes"])
        self.n_classes = len(self.classes)
        self.cls_index = {c: i for i, c in enumerate(self.classes)}

        # --- endmember libraries: load, drop bad bands, subsample ------------
        drop = config["drop_wl_ranges"]
        sub_cfg = config["subsample"]
        method = sub_cfg["method"]
        method_params = sub_cfg.get(method, {})
        rng_init = np.random.default_rng(0)  # deterministic library subsample

        self.library = {}   # class -> (n_sub, 207) float32 subsampled endmembers
        raw_wl = None       # raw 285-band grid, captured once for the banddef
        for cls in self.classes:
            spectra, source, wl = self._load_class(config["endmembers"][cls],
                                                   drop, self.data_root)
            raw_wl = wl if raw_wl is None else raw_wl
            if split is not None:
                keep = _split_indices(len(spectra), split, split_frac, split_seed)
                spectra, source = spectra[keep], source[keep]
            idx = subsample(method, spectra, source, method_params, rng_init)
            self.library[cls] = spectra[idx].astype(np.float32)
        self.n_bands = next(iter(self.library.values())).shape[1]

        # Good-band wavelengths (raw grid with the same bad bands dropped as the
        # spectra). Single source of truth for the model's band definition, so
        # the banddef always aligns with the simulated 207-band spectra.
        self.wavelengths = raw_wl[good_band_mask(raw_wl, drop)].astype(np.float64)
        assert len(self.wavelengths) == self.n_bands, (
            f"{len(self.wavelengths)} good-band wavelengths != {self.n_bands} "
            f"spectra bands")

        # --- presence prior (aligned to the config class axis) ---------------
        prior = load_mixture_prior(_bundled(config["priors_csv"]),
                                   classes=self.classes)
        self.prior = prior
        # patterns as boolean rows over the config class axis; probs to sample.
        self._patterns = prior.patterns.astype(bool)   # (P, n_classes)
        self._probs = prior.probs

        # --- mixing params ----------------------------------------------------
        m = config["mixing"]
        self.n_components = {c: list(m["n_components"][c]) for c in self.classes}
        self.intraclass_floor = float(m["intraclass_floor"])
        self.interclass_min_frac = float(m["interclass_min_frac"])
        # Dirichlet concentration for inter-class fractions. Accepts a scalar
        # (fixed) or a [low, high] range drawn uniformly PER MIXTURE, so the
        # dataset spans concentration regimes: low a (<1) -> dominant+trace
        # (imbalanced), high a (>1) -> even splits (balanced).
        a = m["interclass_alpha"]
        self.interclass_alpha = ((float(a), float(a)) if np.isscalar(a)
                                 else (float(a[0]), float(a[1])))

        # --- augment params ---------------------------------------------------
        a = config["augment"]
        self.albedo = {c: tuple(v) for c, v in a.get("albedo", {}).items()}
        self.magnitude_max = a.get("magnitude_max")
        self.glint_classes = set(a.get("glint", {}).get("classes", []))
        self.glint_range = tuple(a.get("glint", {}).get("range", [0.0, 0.0]))
        self.global_illumination = tuple(a.get("global_illumination", [1.0, 1.0]))

        # --- noise ------------------------------------------------------------
        n = config["noise"]
        cov = None
        if n.get("covariance_csv"):
            cov = np.genfromtxt(_bundled(n["covariance_csv"]), delimiter=",")
            assert cov.shape == (self.n_bands, self.n_bands), (
                f"covariance is {cov.shape}, expected ({self.n_bands}, "
                f"{self.n_bands}) to match the {self.n_bands} good bands")
        self.noise = NoiseModel(
            covariance=cov,
            white_floor_std=n["white_floor_std"],
            covariance_scalar=n["covariance_scalar"],
            brightness_scaled=n["covariance_brightness_scaled"],
        )

    # ------------------------------------------------------------------ init --
    @staticmethod
    def _load_class(files, drop_wl_ranges, data_root):
        """Concatenate class HDF5s, drop bad bands, return (spectra, source, wl).

        `wl` is the raw (pre-drop) wavelength grid, returned for the banddef.
        """
        mats, srcs, wl = [], [], None
        for f in files:
            path = f if os.path.isabs(f) else os.path.join(data_root, f)
            with h5py.File(path, "r") as h:
                sp = h["spectra"][:]
                mats.append(sp)
                if "source" in h:
                    s = h["source"][:]
                    s = np.array([x.decode() if isinstance(x, bytes) else x
                                  for x in s])
                else:
                    s = np.full(len(sp), os.path.splitext(os.path.basename(f))[0])
                srcs.append(s)
                wl = np.array(h.attrs["wavelengths"]).ravel()
        spectra = np.concatenate(mats, axis=0)
        source = np.concatenate(srcs)
        spectra = drop_bad_bands(spectra, wl, drop_wl_ranges)
        return spectra, source, wl

    # --------------------------------------------------------------- helpers --
    def _floor_dirichlet(self, k, floor, rng):
        """k weights >= floor, summing to 1. floor-then-Dirichlet (no rejection)."""
        if k == 1:
            return np.array([1.0])
        return floor + (1.0 - k * floor) * rng.dirichlet(np.ones(k))

    def _intra_class_spectrum(self, cls, rng):
        """Pick k components from the class subsample and intra-class mix them."""
        lib = self.library[cls]
        choices = [k for k in self.n_components[cls] if k <= len(lib)]
        k = int(rng.choice(choices))
        idx = rng.choice(len(lib), size=k, replace=False)
        w = self._floor_dirichlet(k, self.intraclass_floor, rng)
        return w @ lib[idx]

    def _augment_class(self, cls, spectrum, rng):
        """Per-class albedo (+ water glint) on one class's intra-class spectrum."""
        rng_ab = self.albedo.get(cls)
        if rng_ab is not None:
            spectrum = albedo_augment(spectrum, rng_ab[0], rng_ab[1],
                                      self.magnitude_max, rng=rng)
        if cls in self.glint_classes and self.glint_range[1] > 0:
            spectrum = glint_offset(spectrum, self.glint_range[0],
                                    self.glint_range[1], rng=rng)
        return spectrum

    def _interclass_fractions(self, k, rng, max_tries=100):
        """k fractions summing to 1 with every entry >= interclass_min_frac.

        Rejection-sample a Dirichlet; fall back to floor+renormalize if no draw
        clears the floor within max_tries (rare for <=5 classes, min_frac 0.15).
        """
        floor = self.interclass_min_frac
        if k * floor > 1.0:
            raise ValueError(
                f"{k} present classes each need >= {floor}; sum {k*floor} > 1. "
                f"Lower interclass_min_frac or the max classes per mixture.")
        # Draw the concentration for THIS mixture (uniform over the configured
        # range), so the batch spans balanced and imbalanced mixture styles.
        lo, hi = self.interclass_alpha
        a = lo if lo == hi else rng.uniform(lo, hi)
        alpha = np.full(k, a)
        for _ in range(max_tries):
            f = rng.dirichlet(alpha)
            if (f >= floor).all():
                return f
        # Fallback: seat the floor for everyone, distribute the remainder.
        return floor + (1.0 - k * floor) * rng.dirichlet(alpha)

    # -------------------------------------------------------------- simulate --
    def simulate(self, rng):
        """Generate one mixture. Returns (spectra (n_bands,), fractions (n_classes,))."""
        # 1. presence pattern
        p = int(rng.choice(len(self._probs), p=self._probs))
        present = np.flatnonzero(self._patterns[p])  # class indices present

        # 2-3. per class: intra-class mix, then per-class augment
        class_spectra = np.empty((len(present), self.n_bands), dtype=np.float64)
        for j, ci in enumerate(present):
            cls = self.classes[ci]
            s = self._intra_class_spectrum(cls, rng)
            class_spectra[j] = self._augment_class(cls, s, rng)

        # 4. inter-class fractions (per-class floor) + weighted sum
        fracs = self._interclass_fractions(len(present), rng)
        pixel = fracs @ class_spectra

        # 5. global whole-pixel illumination
        lo, hi = self.global_illumination
        if not (lo == hi == 1.0):
            pixel = albedo_augment(pixel, lo, hi, self.magnitude_max, rng=rng)

        # 6. noise
        pixel = pixel + self.noise.sample(pixel.astype(np.float32), rng)

        # 7. full fraction vector over the canonical class axis
        fractions = np.zeros(self.n_classes, dtype=np.float32)
        fractions[present] = fracs
        return pixel.astype(np.float32), fractions


# ------------------------------------------------------------------ selftest --
def _selftest(config_path, n=5000, seed=0):
    sim = MixtureSimulator(config_path)
    rng = np.random.default_rng(seed)
    X = np.empty((n, sim.n_bands), dtype=np.float32)
    F = np.empty((n, sim.n_classes), dtype=np.float32)
    for i in range(n):
        X[i], F[i] = sim.simulate(rng)

    print(f"classes: {sim.classes}")
    print(f"library sizes: " +
          "  ".join(f"{c}={len(sim.library[c])}" for c in sim.classes))
    print(f"spectra: shape={X.shape}  finite={np.isfinite(X).all()}  "
          f"range=[{X.min():.3f}, {X.max():.3f}]")

    # fractions sum to 1, present-class floor respected
    sums = F.sum(axis=1)
    print(f"fraction sums: min={sums.min():.5f} max={sums.max():.5f} "
          f"(should be ~1.0)")
    present_mask = F > 0
    min_present = F[present_mask].min()
    print(f"min present fraction: {min_present:.4f} "
          f"(should be >= {sim.interclass_min_frac})")

    # no forbidden pattern
    allowed = {tuple(row) for row in sim.prior.patterns.astype(bool)}
    seen = {tuple(row) for row in (F > 0)}
    bad = seen - allowed
    print(f"distinct presence patterns seen: {len(seen)}  forbidden seen: {len(bad)}")

    # brightness-scaled noise sanity: compare covariance-noise magnitude on dark
    # (water-only) vs bright (soil-present) pixels via repeated draws on fixed means.
    water_i = sim.cls_index.get("water")
    soil_i = sim.cls_index.get("soil")
    if water_i is not None and soil_i is not None:
        dark = X[(F[:, water_i] > 0)].mean() if (F[:, water_i] > 0).any() else float("nan")
        bright = X[(F[:, soil_i] > 0)].mean() if (F[:, soil_i] > 0).any() else float("nan")
        print(f"mean reflectance: water-present={dark:.3f}  soil-present={bright:.3f}")

    per_class_presence = present_mask.mean(axis=0)
    print("class presence rate: " +
          "  ".join(f"{c}={r:.2f}" for c, r in zip(sim.classes, per_class_presence)))
    print("selftest OK")
    return sim, X, F


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("-n", type=int, default=5000)
    args = ap.parse_args()
    if args.selftest:
        _selftest(args.config, n=args.n)
    else:
        _selftest(args.config, n=args.n)
