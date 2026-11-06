"""Figure 12: per-class fractional cover maps for the FC-8 scene, conditional
(top row) vs global (bottom row) E(MC)^2, one shared color scale.

Needs pipeline/s05_scene.py FC-8 (granule download required, see README).
Writes figures/figure12_fraction_cover_comparison.png.
"""

import argparse

import common as C
import scene_plots as S

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="FC-8")
    a = ap.parse_args()
    s = S.load_scene(a.scene)
    name = "figure12_fraction_cover_comparison.png" if a.scene == "FC-8" else f"figure12_fraction_cover_comparison_{a.scene}.png"
    S.fraction_cover_comparison([s["cond"], s["full"]], ["Conditional unmixing", "Global unmixing"], C.fig_path(name))
    print("saved", C.fig_path(name))
