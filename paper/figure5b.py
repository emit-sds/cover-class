"""Figure 5b: SpecTf label-query classifier architecture diagram.

Styled to match Figure 5 (MLP-NPV) and the pipeline figure: same palette,
fonts and box/arrow helpers. Layer sizes, head count and layer counts are read
from configs/spectf_model_config.yaml, and the parameter count from the
spectf_decoder checkpoint, so the diagram follows the model that was evaluated
(spectf.model.SpecTfLabelQueryEncoder).

Writes figures/figure5b_spectf_architecture.{png,pdf,svg}.
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

import common as C

# Same font lookup as figure5.py (TeX Gyre Pagella, falling back to Palatino/serif).
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
for d in _search:
    for p in glob.glob(os.path.join(d, "texgyrepagella-*.otf")):
        font_manager.fontManager.addfont(p)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["TeX Gyre Pagella", "Palatino", "Palatino Linotype",
                              "Book Antiqua", "STIXGeneral", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

# palette taken from the pipeline figure (same as figure5.py)
PINK = ("#f8dfe1", "#b85450")
YELLOW = ("#f1f2c6", "#9a9a3a")
PURPLE = ("#e1dff0", "#6c66a8")
BLUE = ("#cfe6f3", "#4a8ab8")
ORANGE = ("#fbe3c4", "#c8853a")
GREEN = ("#d9ecd6", "#5c9a52")
WHITE = ("#ffffff", "#555555")
PANEL = ("#f2f2f2", "#444444")
INK = "#1a1a1a"
DASH = (0, (5, 3))

with open(C.CFG["spectf"]["model_config"]) as f:
    M = yaml.safe_load(f)["model"]
D_MODEL, HEADS, D_FF = M["dim_proj"], M["num_heads"], M["dim_ff"]
N_ENC, N_POOL = M["num_layers"], M["num_pool_layers"]
N_BANDS = len(C.load_sim_test()[2])


def n_params():
    import torch
    sd = torch.load(C.CFG["spectf"]["weights"], map_location="cpu")
    return sum(v.numel() for v in sd.values())


fig, ax = plt.subplots(figsize=(26, 10.8))
ax.set_xlim(2, 290)
ax.set_ylim(-16, 94)
ax.set_aspect("equal")
ax.axis("off")


def box(x, y, w, h, colors, lw=2.2, ls="-", r=1.2, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=colors[0], ec=colors[1], lw=lw, ls=ls, zorder=z))


def text(x, y, s, size=21, style="normal", weight="normal", color=INK, ha="center", z=5):
    ax.text(x, y, s, fontsize=size, style=style, weight=weight, color=color,
            ha=ha, va="center", zorder=z)


def arrow(pts, ls="-", lw=2.2, z=1):
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=INK, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=z,
                arrowprops=dict(arrowstyle="-|>,head_length=0.9,head_width=0.45",
                                color=INK, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


def chip(x, y, w, h, label, ec, fc="#ffffff", size=17, lw=1.8):
    box(x, y, w, h, (fc, ec), lw=lw, r=0.7, z=3)
    text(x + w / 2, y + h / 2, label, size=size)


def block(x, y, w, h, title, sub, ec):
    """White sub-block with a title line and an italic detail line."""
    box(x, y, w, h, ("#ffffff", ec), lw=1.8, r=0.7, z=3)
    text(x + w / 2, y + h * 0.66, title, size=18)
    text(x + w / 2, y + h * 0.28, sub, style="italic", size=15)


# ---------------- panel ----------------
box(40, 8, 218, 78, PANEL, lw=2.4, r=1.8, z=0)
text(43, 81, "SpecTf Label-Query Classifier", size=25, weight="bold", ha="left")
text(255, 81, f"{n_params() / 1e3:.0f}k parameters", size=18, style="italic", ha="right",
     color="#555555")

# ---------------- input ----------------
text(20, 56, "Input", size=21, style="italic")
box(5, 36, 30, 16, PINK)
text(20, 47.5, "Reflectance")
text(20, 43.3, "spectrum")
text(20, 39.2, r"$x \in \mathbb{R}^{D}$", style="italic", size=19)
text(20, 32, f"$D$ = {N_BANDS} bands", style="italic", size=17)
text(20, 28.4, r"centers $\lambda_1 \ldots \lambda_D$", style="italic", size=17)

# ---------------- band tokens ----------------
TX, TY, TW, TH = 46, 26, 40, 36
box(TX, TY, TW, TH, YELLOW)
text(TX + TW / 2, TY + 31.5, "Band tokens")
text(TX + TW / 2, TY + 25.5, r"$[\,x_i,\ \tilde\lambda_i\,]$, one per band", style="italic", size=18)
text(TX + TW / 2, TY + 21.0, r"$\tilde\lambda_i = (\lambda_i - 1440)/600$", style="italic", size=17)
text(TX + TW / 2, TY + 16.0, rf"Linear 2 → {D_MODEL}, tanh", style="italic", size=18)
cw, cg = 6.6, 0.9
x0 = TX + (TW - (5 * cw + 4 * cg)) / 2
for i, lab in enumerate([r"$t_1$", r"$t_2$", r"$t_3$", r"$\cdots$", r"$t_D$"]):
    chip(x0 + i * (cw + cg), TY + 4, cw, 6, lab, YELLOW[1], size=17)
arrow([(35, 44), (TX, 44)])

# ---------------- encoder ----------------
EX, EY, EW, EH = 97, 26, 44, 36
box(EX, EY, EW, EH, PURPLE)
text(EX + EW / 2, EY + 31.5, rf"Encoder layer $\times${N_ENC}")
block(EX + 3, EY + 16.5, EW - 6, 9, "Self-attention", f"across bands, {HEADS} heads", PURPLE[1])
chip(EX + 3, EY + 7, EW - 6, 7, f"FFN {D_MODEL}→{D_FF}→{D_MODEL}", PURPLE[1], size=18)
text(EX + EW / 2, EY + 3.2, "pre-LayerNorm" + ("" if M["use_residual"] else ", no residual"),
     style="italic", size=16)
arrow([(TX + TW, 44), (EX, 44)])
text((TX + TW + EX) / 2, 40.6, rf"$D\!\times\!{D_MODEL}$", style="italic", size=15)

# ---------------- label-query pooling ----------------
PX, PY, PW, PH = 157, 13, 56, 65
box(PX, PY, PW, PH, ORANGE)
text(PX + PW / 2, PY + PH - 4.5, rf"Label-query pooling $\times${N_POOL}")
# learned queries, one per class, in the class colors used throughout the paper
QY = PY + PH - 15
qw, qg = 9.4, 0.9
qx0 = PX + (PW - (5 * qw + 4 * qg)) / 2
for i, c in enumerate(C.CLASSES):
    chip(qx0 + i * (qw + qg), QY, qw, 6, C.CLASS_LABELS[c].replace("Snow/ice", "Snow"),
         C.CLASS_COLORS[c], size=15, lw=2.4)
text(PX + PW / 2, QY - 2.6, r"learned class queries $q_c \in \mathbb{R}^{%d}$" % D_MODEL,
     style="italic", size=16)

SX, SW, SH = PX + 4, PW - 8, 8.2
s_cross, s_self, s_ff = PY + 30.5, PY + 19.5, PY + 8.5
block(SX, s_cross, SW, SH, "Cross-attention", "queries attend to band tokens", ORANGE[1])
if M["use_self_attn"]:
    block(SX, s_self, SW, SH, "Self-attention", "among the class queries", ORANGE[1])
    chip(SX, s_ff, SW, SH, f"FFN {D_MODEL}→{D_FF}→{D_MODEL}", ORANGE[1], size=18)
else:
    chip(SX, s_self, SW, SH, f"FFN {D_MODEL}→{D_FF}→{D_MODEL}", ORANGE[1], size=18)
    s_ff = s_self
text(PX + PW / 2, PY + 4.2, "pre-LayerNorm" + (", residual" if M["use_pool_residual"] else ""),
     style="italic", size=16)
mid = SX + SW / 2
arrow([(mid, QY - 4.8), (mid, s_cross + SH)], z=4)
if M["use_self_attn"]:
    arrow([(mid, s_cross), (mid, s_self + SH)], z=4)
    arrow([(mid, s_self), (mid, s_ff + SH)], z=4)
else:
    arrow([(mid, s_cross), (mid, s_ff + SH)], z=4)
# encoder tokens -> cross-attention keys/values
kx = (EX + EW + PX) / 2
arrow([(EX + EW, 44), (kx, 44), (kx, s_cross + SH / 2), (SX, s_cross + SH / 2)], z=4)
text(kx, 40.6, "keys,", style="italic", size=16)
text(kx, 37.4, "values", style="italic", size=16)

# ---------------- per-class head ----------------
HX, HY, HW, HH = 220, 30, 33, 28
box(HX, HY, HW, HH, BLUE)
text(HX + HW / 2, HY + 23, "Per-class head")
text(HX + HW / 2, HY + 15.5, r"$z_c = w_c^{\top} h_c + b_c$", size=21)
text(HX + HW / 2, HY + 8.6, r"independent $w_c$ per class,", style="italic", size=16)
text(HX + HW / 2, HY + 5.2, "no shared output layer", style="italic", size=16)
hx = (PX + PW + HX) / 2
arrow([(SX + SW, s_ff + SH / 2), (hx, s_ff + SH / 2), (hx, 44), (HX, 44)], z=4)
text(hx + 0.8, s_ff + SH / 2 - 3.6, r"$h_1 \ldots h_C$", style="italic", size=17, ha="left")

# ---------------- output ----------------
box(266, 34, 21, 20, GREEN)
text(276.5, 48.5, "Class", size=20)
text(276.5, 44.4, "presence", size=20)
text(276.5, 39.8, "sigmoid", style="italic", size=18)
arrow([(HX + HW, 44), (266, 44)])

# ---------------- training loss (dashed = training only) ----------------
LX, LY, LW, LH = 201, -13, 52, 15
box(LX, LY, LW, LH, WHITE)
text(LX + LW / 2, LY + 11.2, "Training loss")
text(LX + LW / 2, LY + 5.0, r"$\mathcal{L} = \mathrm{BCE}(z, y)$", size=20)
arrow([(HX + HW - 5, HY), (HX + HW - 5, LY + LH)], ls=DASH)
text(HX + HW - 3.5, 20, r"$z$", style="italic", size=19, ha="left")
text(LX + LW + 2, LY + LH / 2, "training only", style="italic", size=18, ha="left", color="#555555")

# ---------------- band-invariance note ----------------
text(43, 3.2, "Each token carries its own wavelength, and attention pooling does not fix "
     "the number or order of tokens.", style="italic", size=17, ha="left", color="#555555")

for ext in ["png", "pdf", "svg"]:
    fig.savefig(C.fig_path(f"figure5b_spectf_architecture.{ext}"), dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
print("saved", C.fig_path("figure5b_spectf_architecture.png"))
