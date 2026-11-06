"""End-of-training reporting: markdown metric tables, JSON, + figure export.

Replaces the old ``cover_class.reporting`` PDF pipeline with lightweight
artifacts written to the run's ``outdir``:

- ``<model>_<run>.md`` — human-readable per-class metric tables, one section
  per evaluation regime (see ``write_report``).
- ``<model>_<run>.json`` — the same sections in a machine-readable form (see
  ``write_json``).
- PNG figures (ROC, confusion, regression summary) written alongside.

The metric dicts are exactly what ``models.metrics.binary_metrics_at`` /
``regression_error`` return: ``{class_name: {metric: value}}``.
"""

import json
import math
import os
from typing import Dict, List, Optional

# Columns rendered for the binary presence tables, in order.
_BINARY_COLS = ["threshold", "f1", "tpr", "fpr", "roc_auc", "tp", "fp", "fn", "tn"]
_BINARY_HDR = ["Threshold", "F1", "TPR", "FPR", "ROC-AUC", "TP", "FP", "FN", "TN"]
_REG_COLS = ["mae", "rmse", "r2"]
_REG_HDR = ["MAE", "RMSE", "R2"]


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _md_table(metrics: Dict[str, dict], cols: List[str], hdr: List[str]) -> str:
    lines = ["| Class | " + " | ".join(hdr) + " |",
             "|" + "---|" * (len(hdr) + 1)]
    for cls, d in metrics.items():
        row = [cls] + [_fmt(d.get(c, "")) for c in cols]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_report(sections: List[tuple], title: str) -> str:
    """Assemble a markdown document from (key, heading, metrics, kind) sections.

    kind is 'binary' or 'regression', selecting the column set.
    """
    parts = [f"# {title}", ""]
    for _key, heading, metrics, kind in sections:
        parts.append(f"## {heading}")
        parts.append("")
        if kind == "regression":
            parts.append(_md_table(metrics, _REG_COLS, _REG_HDR))
        else:
            parts.append(_md_table(metrics, _BINARY_COLS, _BINARY_HDR))
        parts.append("")
    return "\n".join(parts)


def write_report(sections: List[tuple], outdir: str, basename: str, title: str) -> str:
    """Write the report as ``.md``. Returns the path.

    ``sections`` is a list of ``(key, heading, metrics_dict, kind)`` tuples.
    """
    os.makedirs(outdir, exist_ok=True)
    doc = build_report(sections, title)
    md_path = os.path.join(outdir, f"{basename}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return md_path


def _json_safe(v):
    """NaN/Inf -> None so the output is standard (strict) JSON."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def write_json(sections: List[tuple], outdir: str, basename: str) -> str:
    """Write ``sections`` as ``{key: {class: {metric: value}}}``. Returns the path.

    Same data as ``write_report``, keyed by each section's short ``key``
    instead of rendered as a table, for programmatic comparison across runs.
    """
    os.makedirs(outdir, exist_ok=True)
    doc = {
        key: {cls: {k: _json_safe(v) for k, v in d.items()} for cls, d in metrics.items()}
        for key, _heading, metrics, _kind in sections
    }
    json_path = os.path.join(outdir, f"{basename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return json_path


def save_figures(figs: Dict[str, "object"], outdir: str, basename: str) -> List[str]:
    """Save a ``{name: matplotlib Figure}`` dict as ``<basename>_<name>.png``.

    Returns the list of written paths. Figures are closed after saving.
    """
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    paths = []
    for name, fig in figs.items():
        if fig is None:
            continue
        p = os.path.join(outdir, f"{basename}_{name}.png")
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths
