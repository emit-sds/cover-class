"""Figure 2: conditional unmixing pipeline diagram (static; unchanged by v2).

Copied from EMIT-FC/paper/make_pipeline_figure.py; only OUTPUT changed.
"""
#!/usr/bin/env python3
"""
Conditional unmixing pipeline figure (Training / Inference).

Edit the TEXT, CLASSES and OUTPUT sections below, then run:
    python make_pipeline_figure.py

Needs only matplotlib. Writes PDF (for LaTeX), PNG and SVG.
Math works in any string with $...$, e.g. 'E(MC)$^2$'.
Boxes and spacing resize automatically to fit the text, so change text or
font sizes freely. Empty strings "" are skipped.
"""
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arc

# =============================================================================
# 1. TEXT  -- change any string here. For boxes, the first item is the bold-ish
#    title and the rest are italic sub-lines (add or remove lines freely).
# =============================================================================
TEXT = {
    # panel headings
    "train_title": "Training",
    "train_note":  "",
    "infer_title": "Inference",
    "infer_note":  "",
    "input_label": "Input",

    # input column
    "libraries":   ["Spectral libraries", "field + lab"],
    "pixel":       ["Observed pixel", "reflectance spectrum"],

    # training row
    "reduction":   ["Library reduction", "representative", "endmembers"],
    "simulation":  ["Mixture simulation", "single-class, binary,", "complex mixtures"],
    "clf_title":   "Class-presence classifier",
    "clf_models":  ["MLP", "SpecTF"],            # small boxes inside classifier
    "clf_note":    ["multi-label, trained on", "simulated train split"],
    "evaluation":  ["Evaluation", "held-out simulated", "test split"],
    "ood":         ["OOD data", "real EMIT spectra"],   # input above Evaluation

    # inference row
    "clf_infer":   ["Class-presence classifier", ""],
    "pred_title":  "Predicted classes present",
    "pred_note":   "example: probabilities thresholded",
    "sma":         ["SMA / E(MC)$^2$", "predicted classes’", "endmembers only"],
    "fractions":   ["Fractional cover", "of predicted classes"],

    # arrow labels
    "lbl_sim_to_clf":   ["labeled", "spectra"],   # above / below the arrow
    "lbl_weights":      "trained classifier weights",
    "lbl_endmembers":   "reduced endmembers",
    "lbl_observed":     "observed spectrum",
}

# =============================================================================
# 2. CLASSES shown in the "predicted classes" box: (name, predicted present?)
# =============================================================================
CLASSES = [("Soil", True), ("PV", True), ("NPV", True),
           ("Water", False), ("Snow", False)]

# =============================================================================
# 3. OUTPUT / STYLE
# =============================================================================
import os as _os
OUTPUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "figures", "figure2_conditional_unmixing_pipeline_v4")   
PNG_DPI = 300
FONTS = ["Latin Modern Roman", "CMU Serif", "Times New Roman",
         "Times", "Nimbus Roman", "DejaVu Serif"]   # first one installed wins

COLORS = {  # (fill, edge)
    "pink":   ("#f8dfe0", "#a8656b"),
    "yellow": ("#f2f3c8", "#8f8f45"),
    "purple": ("#e0e0f3", "#6d6da3"),
    "blue":   ("#cfe8f5", "#4a87a8"),
    "orange": ("#fbe3c4", "#b98545"),
    "green":  ("#d7ecd4", "#5a8c57"),
    "gray":   ("#f1f1f1", "#555555"),
    "panel":  ("#f3f3f3", "#444444"),
    "white":  ("#ffffff", "#555555"),
}

add=10
# Font sizes (boxes resize automatically to fit the text)
FS_TITLE      = 15+add    # first line in each box
FS_SUB        = 13+add    # italic lines in boxes
FS_LABEL      = 12.5+add  # text next to arrows
FS_PANEL      = 18+add    # "Training" / "Inference"
FS_PANEL_NOTE = 13+add
FS_MODEL      = 14+add    # "MLP" / "SpecTF"
FS_CHIP       = 12.5+add  # class chips
FS_INPUT      = 14+add    # "Input"

PX = 0.67      # overall text scale; raise for bigger text everywhere

# Spacing (layout units; 1 unit = 1/100 inch in the saved figure)
BOX_PAD_X = 20        # left/right padding inside boxes
BOX_PAD_Y = 16        # top/bottom padding inside boxes
MIN_ARROW = 60        # shortest horizontal arrow between boxes
MARGIN    = 20        # blank border around the figure
LINE_LW   = 1.6       # arrow line width

# =============================================================================
# Drawing code -- no need to edit below.
# =============================================================================
# font sizes -> points
(FS_TITLE, FS_SUB, FS_LABEL, FS_PANEL, FS_PANEL_NOTE, FS_MODEL, FS_CHIP,
 FS_INPUT) = [v * PX for v in (FS_TITLE, FS_SUB, FS_LABEL, FS_PANEL,
                               FS_PANEL_NOTE, FS_MODEL, FS_CHIP, FS_INPUT)]
# drop empty lines
for _k, _v in TEXT.items():
    if isinstance(_v, list):
        TEXT[_k] = [s for s in _v if s]
INK, SUBINK = "#222222", "#333333"
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams.update({"font.family": "serif", "font.serif": FONTS,
                     "mathtext.fontset": "cm",
                     "pdf.fonttype": 42, "svg.fonttype": "none"})

from matplotlib.font_manager import findfont, FontProperties
print("Font used:", findfont(FontProperties(family="serif"), fallback_to_default=True))

fig = plt.figure(figsize=(10, 5), dpi=100)
_r = fig.canvas.get_renderer()


def px(pt):                       # font size in points -> layout units
    return pt * 100 / 72


def tw(s, pt, italic=False, bold=False):
    """Rendered width of a string, in layout units."""
    t = fig.text(0, 0, s, fontsize=pt, style="italic" if italic else "normal",
                 weight="bold" if bold else "normal")
    w = t.get_window_extent(_r).width
    t.remove()
    return w


def block_metrics(lines):
    """Width, height and line offsets for title + italic sub-lines."""
    tp, sp = px(FS_TITLE), px(FS_SUB)
    offs = [0.0]
    for i in range(1, len(lines)):
        offs.append(offs[-1] + (1.25 * (tp + sp) / 2 if i == 1 else 1.25 * sp))
    last = tp if len(lines) == 1 else sp
    h = offs[-1] + 0.5 * tp + 0.5 * last + 2 * BOX_PAD_Y + 6
    w = max([tw(lines[0], FS_TITLE)] +
            [tw(s, FS_SUB, italic=True) for s in lines[1:]]) + 2 * BOX_PAD_X
    return w, h, offs


# ---------- measure every element ----------------------------------------------
M = {}
for k in ["libraries", "pixel", "reduction", "simulation", "evaluation", "ood",
          "clf_infer", "sma", "fractions"]:
    M[k] = block_metrics(TEXT[k])

# classifier-training box: title / model boxes / note lines
mod_w = max(tw(m, FS_MODEL) for m in TEXT["clf_models"]) + 28
mod_h = px(FS_MODEL) + 16
mod_gap = 16
mods_w = len(TEXT["clf_models"]) * mod_w + (len(TEXT["clf_models"]) - 1) * mod_gap
clf_w = max(tw(TEXT["clf_title"], FS_TITLE), mods_w,
            *[tw(s, FS_SUB, italic=True) for s in TEXT["clf_note"]]) + 2 * BOX_PAD_X
clf_h = (BOX_PAD_Y + px(FS_TITLE) + 10 + mod_h + 10 +
         len(TEXT["clf_note"]) * 1.25 * px(FS_SUB) + BOX_PAD_Y)
M["clf_train"] = (clf_w, clf_h, None)

# predicted-classes box: title / chips / note
chip_w = max(tw(c, FS_CHIP) for c, _ in CLASSES) + 20
chip_h = px(FS_CHIP) + 12
chip_gap = 8
chips_w = len(CLASSES) * chip_w + (len(CLASSES) - 1) * chip_gap
pred_w = max(tw(TEXT["pred_title"], FS_TITLE), chips_w,
             tw(TEXT["pred_note"], FS_SUB, italic=True)) + 2 * BOX_PAD_X
pred_h = (BOX_PAD_Y + px(FS_TITLE) + 10 + chip_h + 10 +
          1.25 * px(FS_SUB) + BOX_PAD_Y)
M["pred"] = (pred_w, pred_h, None)

lbl_h = px(FS_LABEL) * 1.2
sim_lbl_w = max(tw(s, FS_LABEL, italic=True) for s in TEXT["lbl_sim_to_clf"])

# ---------- horizontal layout --------------------------------------------------
in_w = max(M["libraries"][0], M["pixel"][0], tw(TEXT["input_label"], FS_INPUT))
in_x = MARGIN
panel_x = in_x + in_w + MIN_ARROW / 2
start = panel_x + MIN_ARROW / 2 + 10

train_keys = ["reduction", "simulation", "clf_train", "evaluation"]
infer_keys = ["clf_infer", "pred", "sma", "fractions"]
train_gaps = [MIN_ARROW, max(MIN_ARROW, sim_lbl_w + 24), MIN_ARROW]
infer_gaps = [MIN_ARROW] * 3


def row_len(keys, gaps):
    return sum(M[k][0] for k in keys) + sum(gaps)


L = max(row_len(train_keys, train_gaps), row_len(infer_keys, infer_gaps))
for keys, gaps in [(train_keys, train_gaps), (infer_keys, infer_gaps)]:
    extra = (L - row_len(keys, gaps)) / len(gaps)       # spread rows evenly
    for i in range(len(gaps)):
        gaps[i] += extra

X = {}
for keys, gaps in [(train_keys, train_gaps), (infer_keys, infer_gaps)]:
    x = start
    for i, k in enumerate(keys):
        X[k] = x
        x += M[k][0] + (gaps[i] if i < len(gaps) else 0)
panel_w = start + L + 20 - panel_x
W = panel_x + panel_w + MARGIN

# ---------- vertical layout ----------------------------------------------------
has_note = bool(TEXT['train_note'] or TEXT['infer_note'])
head_h = px(FS_PANEL) + (px(FS_PANEL_NOTE) * 1.3 if has_note else 0) + 22
row_pad = 20
t_top = MARGIN + M["ood"][1] + 40      # room for the OOD input box above
t_rowh = max(M[k][1] for k in train_keys + ["libraries"])
t_cy = t_top + head_h + row_pad + t_rowh / 2
t_bot = t_cy + t_rowh / 2 + row_pad

line1 = t_bot + lbl_h + 8           # trained-weights link
line2 = line1 + 20                  # endmember link
i_top = line2 + lbl_h + 16
i_rowh = max(M[k][1] for k in infer_keys + ["pixel"])
i_cy = i_top + head_h + row_pad + i_rowh / 2
obs_y = i_cy + i_rowh / 2 + lbl_h + 16  # observed-spectrum link
i_bot = obs_y + 18
H = i_bot + MARGIN

# ---------- drawing helpers ----------------------------------------------------
fig.set_size_inches(W / 100, H / 100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")


def rect(x, y, w, h, col, r=7, lw=1.3, dashed=False, fill=None, edge=None):
    f, e = COLORS[col]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        fc=fill or f, ec=edge or e, lw=lw, ls=(0, (3, 2)) if dashed else "-",
        zorder=2))


def txt(x, y, s, pt, italic=False, color=INK, ha="center", weight="normal"):
    ax.text(x, y, s, fontsize=pt, style="italic" if italic else "normal",
            color=color, ha=ha, va="center_baseline", weight=weight, zorder=5)


def box(key, x, cy, col):
    w, h, offs = M[key]
    lines = TEXT[key]
    rect(x, cy - h / 2, w, h, col)
    y0 = cy - offs[-1] / 2
    for i, (s, o) in enumerate(zip(lines, offs)):
        if i == 0:
            txt(x + w / 2, y0 + o, s, FS_TITLE)
        else:
            txt(x + w / 2, y0 + o, s, FS_SUB, italic=True, color=SUBINK)
    return x, cy - h / 2, w, h


def arrow(pts, dashed=False, hop=None):
    ls = (0, (6, 4)) if dashed else "-"
    pts = list(pts)
    if hop:
        hx, hy, r = hop[0], hop[1], 8
        for i in range(len(pts) - 1):
            (x1, y1), (x2, y2) = pts[i], pts[i + 1]
            if y1 == y2 == hy and min(x1, x2) < hx < max(x1, x2):
                s = 1 if x2 > x1 else -1
                ax.plot([x1, hx - s * r], [y1, y1], color=INK, lw=LINE_LW, ls=ls, zorder=3)
                t = np.linspace(np.pi, 0, 30) if s > 0 else np.linspace(0, np.pi, 30)
                ax.plot(hx + r * np.cos(t), hy - r * np.sin(t), color=INK, lw=LINE_LW, zorder=3)
                pts = [(hx + s * r, hy)] + pts[i + 1:]
                break
            ax.plot([x1, x2], [y1, y2], color=INK, lw=LINE_LW, ls=ls, zorder=3)
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=INK, lw=LINE_LW, ls=ls, zorder=3)
    (xa, ya), (xb, yb) = pts[-2], pts[-1]
    d = np.hypot(xb - xa, yb - ya)
    back = (xb - (xb - xa) / d * 12, yb - (yb - ya) / d * 12)
    ax.annotate("", xy=(xb, yb), xytext=back, zorder=4,
                arrowprops=dict(arrowstyle="-|>,head_length=0.7,head_width=0.3",
                                color=INK, lw=LINE_LW, shrinkA=0, shrinkB=0))


def label(x, y, s, ha="center"):
    txt(x, y, s, FS_LABEL, italic=True, color=SUBINK, ha=ha)


def panel(y, h, title, note):
    rect(panel_x, y, panel_w, h, "panel", r=12)
    txt(panel_x + 18, y + 12 + px(FS_PANEL) * 0.55, title, FS_PANEL, ha="left", weight="bold")
    if note:
        txt(panel_x + 18, y + 16 + px(FS_PANEL) + px(FS_PANEL_NOTE) * 0.6, note,
            FS_PANEL_NOTE, italic=True, color=SUBINK, ha="left")


def right(b): return b[0] + b[2]
def cx(b): return b[0] + b[2] / 2
def top(b): return b[1]
def bottom(b): return b[1] + b[3]


# ---------- panels -------------------------------------------------------------
panel(t_top, t_bot - t_top, TEXT["train_title"], TEXT["train_note"])
panel(i_top, i_bot - i_top, TEXT["infer_title"], TEXT["infer_note"])

# ---------- input column -------------------------------------------------------
lib = box("libraries", in_x + (in_w - M["libraries"][0]) / 2, t_cy, "pink")
pix = box("pixel", in_x + (in_w - M["pixel"][0]) / 2, i_cy, "pink")
txt(in_x + in_w / 2, top(lib) - px(FS_INPUT), TEXT["input_label"], FS_INPUT,
    italic=True, color=SUBINK)

# ---------- training row -------------------------------------------------------
red = box("reduction", X["reduction"], t_cy, "yellow")
sim = box("simulation", X["simulation"], t_cy, "purple")

cw_, ch_ = M["clf_train"][:2]
clf = (X["clf_train"], t_cy - ch_ / 2, cw_, ch_)
rect(*clf, "blue")
y = top(clf) + BOX_PAD_Y + px(FS_TITLE) / 2
txt(cx(clf), y, TEXT["clf_title"], FS_TITLE)
y += px(FS_TITLE) / 2 + 10
mx = cx(clf) - mods_w / 2
for m in TEXT["clf_models"]:
    rect(mx, y, mod_w, mod_h, "blue", r=5, lw=1.1, fill="#ffffff")
    txt(mx + mod_w / 2, y + mod_h / 2, m, FS_MODEL)
    mx += mod_w + mod_gap
y += mod_h + 10 + 1.25 * px(FS_SUB) / 2
for s in TEXT["clf_note"]:
    txt(cx(clf), y, s, FS_SUB, italic=True, color=SUBINK)
    y += 1.25 * px(FS_SUB)

ev = box("evaluation", X["evaluation"], t_cy, "gray")
ood_x = min(cx(ev) - M["ood"][0] / 2, W - MARGIN - M["ood"][0])
ood = box("ood", ood_x, MARGIN + M["ood"][1] / 2, "pink")
ood_ax = min(max(cx(ood), ev[0] + 15), right(ev) - 15)  # stay within Evaluation box
arrow([(ood_ax, bottom(ood)), (ood_ax, top(ev))])

arrow([(right(lib), t_cy), (red[0], t_cy)])
arrow([(right(red), t_cy), (sim[0], t_cy)])
arrow([(right(sim), t_cy), (clf[0], t_cy)])
mid = (right(sim) + clf[0]) / 2
label(mid, t_cy - lbl_h * 0.75, TEXT["lbl_sim_to_clf"][0])
label(mid, t_cy + lbl_h * 0.95, TEXT["lbl_sim_to_clf"][1])
arrow([(right(clf), t_cy), (ev[0], t_cy)])

# ---------- inference row ------------------------------------------------------
ci = box("clf_infer", X["clf_infer"], i_cy, "blue")

pw_, ph_ = M["pred"][:2]
pred = (X["pred"], i_cy - ph_ / 2, pw_, ph_)
rect(*pred, "white")
y = top(pred) + BOX_PAD_Y + px(FS_TITLE) / 2
txt(cx(pred), y, TEXT["pred_title"], FS_TITLE)
y += px(FS_TITLE) / 2 + 10
kx = cx(pred) - chips_w / 2
for name, on in CLASSES:
    if on:
        rect(kx, y, chip_w, chip_h, "blue", r=5, lw=1.2)
        txt(kx + chip_w / 2, y + chip_h / 2, name, FS_CHIP)
    else:
        rect(kx, y, chip_w, chip_h, "white", r=5, lw=1.1, dashed=True, edge="#b5b5b5")
        txt(kx + chip_w / 2, y + chip_h / 2, name, FS_CHIP, color="#a8a8a8")
    kx += chip_w + chip_gap
txt(cx(pred), y + chip_h + 10 + 1.25 * px(FS_SUB) / 2, TEXT["pred_note"],
    FS_SUB, italic=True, color=SUBINK)

sma = box("sma", X["sma"], i_cy, "orange")
frac = box("fractions", X["fractions"], i_cy, "green")

arrow([(right(pix), i_cy), (ci[0], i_cy)])
arrow([(right(ci), i_cy), (pred[0], i_cy)])
arrow([(right(pred), i_cy), (sma[0], i_cy)])
arrow([(right(sma), i_cy), (frac[0], i_cy)])
arrow([(cx(pix), bottom(pix)), (cx(pix), obs_y), (cx(sma), obs_y), (cx(sma), bottom(sma))])
label((cx(pix) + cx(sma)) / 2, obs_y - lbl_h * 0.7, TEXT["lbl_observed"])

# ---------- links between phases (dashed) --------------------------------------
note_end = panel_x + 18 + (tw(TEXT["infer_note"], FS_PANEL_NOTE, italic=True) + 15
                           if TEXT["infer_note"] else 0)
x_land = min(max(cx(ci), note_end), right(ci) - 15)   # avoid panel note
arrow([(cx(clf), bottom(clf)), (cx(clf), line1), (x_land, line1), (x_land, top(ci))],
      dashed=True)
label((cx(clf) + x_land) / 2, line1 - lbl_h * 0.65, TEXT["lbl_weights"])
arrow([(cx(red), bottom(red)), (cx(red), line2), (cx(sma), line2), (cx(sma), top(sma))],
      dashed=True, hop=(x_land, line2) if cx(red) < x_land else None)
label(cx(sma) + 10, line2 + lbl_h * 0.8, TEXT["lbl_endmembers"], ha="left")

# ---------- save ---------------------------------------------------------------
for ext in ("pdf", "png", "svg"):
    fig.savefig(f"{OUTPUT}.{ext}", dpi=PNG_DPI if ext == "png" else None,
                facecolor="white")
print(f"Saved {OUTPUT}.pdf / .png / .svg  ({W/100:.1f} x {H/100:.1f} in)")
