"""Figure 11: RGB + classes-per-pixel (fraction > 0.05) under global vs
SpecTf-conditional E(MC)^2 for the FC-8 scene (Northern Patagonian Ice Field).

Needs pipeline/s05_scene.py FC-8 (granule download required, see README).
Writes figures/figure11_num_classes_comparison_fc-8.png.
"""

import argparse

import common as C
import scene_plots as S

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="FC-8")
    a = ap.parse_args()
    s = S.load_scene(a.scene)
    out = C.fig_path(f"figure11_num_classes_comparison_{a.scene.lower()}.png")
    S.num_classes_comparison(s["rgb"], [s["full"], s["cond"]], ["Global unmixing", "Conditional unmixing"], out)
    print("saved", out)
