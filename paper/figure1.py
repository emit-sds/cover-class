"""Figure 1: global E(MC)^2 of a bare-soil desert scene (FC-4, Anti-Atlas).

Needs pipeline/s05_scene.py FC-4 (granule download required, see README).
Writes figures/figure1_fraction_cover_full_FC-4.png.
"""

import argparse

import common as C
import scene_plots as S

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="FC-4")
    a = ap.parse_args()
    s = S.load_scene(a.scene)
    S.fraction_cover(s["full"], s["rgb"], C.fig_path(f"figure1_fraction_cover_full_{a.scene}.png"))
    print("saved", C.fig_path(f"figure1_fraction_cover_full_{a.scene}.png"))
