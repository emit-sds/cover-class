"""Figure 5: MLP-NPV architecture diagram (static; unchanged by v2).

Copied from EMIT-FC/paper/mlp_npv_diagram.py; only the output path changed.
"""
"""Architecture diagram for SpectralMLPClassifierNPV, styled to match
conditional_unmixing_pipeline_v4.png."""
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

import glob
import os

# Look for TeX Gyre Pagella (Palatino clone) in common Linux / macOS / TeX Live
# locations; fall back to Palatino or any serif font if it isn't installed.
_search = [
    "/usr/share/texmf/fonts/opentype/public/tex-gyre",
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre",
    "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/tex-gyre",
    "/Library/TeX/Root/texmf-dist/fonts/opentype/public/tex-gyre",
    "/opt/homebrew/share/fonts",
    os.path.expanduser("~/Library/Fonts"),
    "/Library/Fonts",
    os.path.expanduser("~/.local/share/fonts"),
]
_found = []
for d in _search:
    _found += glob.glob(os.path.join(d, "texgyrepagella-*.otf"))
for p in _found:
    font_manager.fontManager.addfont(p)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["TeX Gyre Pagella", "Palatino", "Palatino Linotype",
                              "Book Antiqua", "STIXGeneral", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

# palette taken from the pipeline figure
PINK = ("#f8dfe1", "#b85450")
YELLOW = ("#f1f2c6", "#9a9a3a")
PURPLE = ("#e1dff0", "#6c66a8")
BLUE = ("#cfe6f3", "#4a8ab8")
ORANGE = ("#fbe3c4", "#c8853a")
GREEN = ("#d9ecd6", "#5c9a52")
WHITE = ("#ffffff", "#555555")
PANEL = ("#f2f2f2", "#444444")
INK = "#1a1a1a"

fig, ax = plt.subplots(figsize=(26, 10.8))
ax.set_xlim(16, 260)
ax.set_ylim(-16, 94)
ax.set_aspect("equal")
ax.axis("off")


def box(x, y, w, h, colors, lw=2.2, ls="-", r=1.2, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=colors[0], ec=colors[1], lw=lw, ls=ls, zorder=z))


def text(x, y, s, size=21, style="normal", weight="normal", color=INK, ha="center", z=5):
    ax.text(x, y, s, fontsize=size, style=style, weight=weight, color=color,
            ha=ha, va="center", zorder=z)


def arrow(pts, ls="-", lw=2.2):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys, color=INK, lw=0, zorder=1)
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=INK, lw=lw, ls=ls, zorder=1,
                solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=1,
                arrowprops=dict(arrowstyle="-|>,head_length=0.9,head_width=0.45",
                                color=INK, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


def layer_chips(x0, y, widths, chip_w=8.5, gap=1.6, h=5.4):
    for i, wd in enumerate(widths):
        cx = x0 + i * (chip_w + gap)
        box(cx, y, chip_w, h, ("#ffffff", PURPLE[1] if wd else "#999"), lw=1.8, r=0.7, z=3)
        text(cx + chip_w / 2, y + h / 2, str(wd), size=19)


# ---------------- panel ----------------
box(58, 8, 199, 78, PANEL, lw=2.4, r=1.8, z=0)
text(61, 81, "Spectral MLP-NPV Classifier", size=25, weight="bold", ha="left")

# ---------------- input ----------------
text(35, 56, "Input", size=21, style="italic")
box(20, 36, 30, 16, PINK)
text(35, 47.5, "Reflectance")
text(35, 43.3, "spectrum")
text(35, 39.2, r"$x \in \mathbb{R}^{D}$", style="italic", size=19)


# ---------------- shared branch (top) ----------------
SY = 56  # shared row bottom
box(84, SY, 42, 22, PURPLE)
text(105, SY + 18, "Shared backbone")
layer_chips(84 + (42 - (2 * 8.5 + 1.6)) / 2, SY + 9.3, [64, 32])
text(105, SY + 5.6, "Linear → BatchNorm → ReLU", style="italic", size=17)
text(105, SY + 2.4, "dropout 0.0", style="italic", size=18)

box(130.5, SY + 4, 29, 14, BLUE)
text(145, SY + 13.8, "Shared head")
text(145, SY + 8.4, r"Linear 32 → $C$", style="italic")

# ---------------- NPV branch (bottom) ----------------
NY = 12
box(84, NY, 42, 22, ORANGE)
text(105, NY + 18, "NPV backbone")
cw, cg = 8.5, 1.4
layer_chips(84 + (42 - (4 * cw + 3 * cg)) / 2, NY + 9.3, [128, 64, 64, 64], chip_w=cw, gap=cg)
text(105, NY + 5.6, "Linear → BatchNorm → ReLU", style="italic", size=17)
text(105, NY + 2.4, "dropout 0.1", style="italic", size=18)

box(130.5, NY + 4, 29, 14, ORANGE)
text(145, NY + 13.8, "NPV aux head")
text(145, NY + 8.4, r"Linear 64 → $C$", style="italic")

# fork from input norm
arrow([(50, 44), (74, 44), (74, SY + 11), (84, SY + 11)])
arrow([(74, 44), (74, NY + 11), (84, NY + 11)])
arrow([(126, SY + 11), (130.5, SY + 11)])
arrow([(126, NY + 11), (130.5, NY + 11)])

# ---------------- logit merge ----------------
MX, MY, MW, MH = 182, 31, 48, 26
box(MX, MY, MW, MH, WHITE)
text(MX + MW / 2, MY + 21.5, "Final logits")
classes = ["Soil", "PV", "NPV", "Water", "Snow"]
chip_w, chip_g = 8.4, 0.8
x0 = MX + (MW - (5 * chip_w + 4 * chip_g)) / 2
for i, c in enumerate(classes):
    col = ORANGE if c == "NPV" else BLUE
    cx = x0 + i * (chip_w + chip_g)
    box(cx, MY + 11, chip_w, 6, col, lw=1.8, r=0.6, z=3)
    text(cx + chip_w / 2, MY + 14, c, size=15)
text(MX + MW / 2, MY + 6.6, "NPV logit from aux branch,", style="italic", size=18)
text(MX + MW / 2, MY + 3.0, "all others from shared head", style="italic", size=18)

npv_cx = x0 + 2 * (chip_w + chip_g) + chip_w / 2
# shared logits -> merge (top)
arrow([(159.5, SY + 11), (176, SY + 11), (176, MY + 20), (MX, MY + 20)])
text(167.5, SY + 17.5, "shared", style="italic", size=18)
text(167.5, SY + 14.0, "logits", style="italic", size=18)
# aux NPV logit -> merge (bottom, into NPV chip)
arrow([(159.5, NY + 11), (npv_cx, NY + 11), (npv_cx, MY)])
text(171.5, NY + 14.0, "aux logit [NPV]", style="italic", size=18)

# ---------------- output ----------------
box(236, 34, 19, 20, GREEN)
text(245.5, 48.5, "Class", size=20)
text(245.5, 44.4, "presence", size=20)
text(245.5, 39.8, "sigmoid", style="italic", size=18)
arrow([(MX + MW, 44), (236, 44)])

# ---------------- optional aux output (dashed) ----------------
# ---------------- training loss (dashed = training only) ----------------
DASH = (0, (5, 3))
LX, LY, LW, LH = 162, -13, 70, 15
box(LX, LY, LW, LH, ("#ffffff", "#555555"))
text(LX + LW / 2, LY + 11.2, "Training loss")
text(LX + LW / 2, LY + 5.0,
     r"$\mathcal{L} = \mathrm{BCE}(z_{\mathrm{final}}, y) + 0.3\,\mathrm{BCE}(z_{\mathrm{aux}}, y)$",
     size=20)
# all aux logits -> aux term
arrow([(145, NY + 4), (145, LY + LH / 2), (LX, LY + LH / 2)], ls=DASH)
text(143, LY + LH / 2, r"all $C$ aux logits, $z_{\mathrm{aux}}$", style="italic", size=18, ha="right")
# final logits -> main term
fx = MX + MW - 6
arrow([(fx, MY), (fx, LY + LH)], ls=DASH)
text(fx + 1.5, 20, r"$z_{\mathrm{final}}$", style="italic", size=19, ha="left")
text(LX + LW + 2, LY + LH / 2, "training only", style="italic", size=18, ha="left",
     color="#555555")

for ext in ["png", "pdf", "svg"]:
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", f"figure5_mlp_npv_architecture.{ext}"), dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
