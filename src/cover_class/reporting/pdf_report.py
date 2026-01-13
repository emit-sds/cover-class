from typing import Any, List, Dict, TYPE_CHECKING
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
import numpy as np
from numpy.typing import NDArray
import math
import os
import warnings

from reportlab.lib.pagesizes import LETTER # type: ignore[import]
from reportlab.lib import colors # type: ignore[import]
from reportlab.platypus import ( # type: ignore[import]
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle # type: ignore[import]
from reportlab.lib.units import inch # type: ignore[import]

from cover_class.reporting.utils import inference_over_scene, rgb_from_scene
if TYPE_CHECKING:
    from cover_class.reporting import Report, GenLinePlot

rwb = LinearSegmentedColormap.from_list("rwb", [(0.0, "blue"), (0.5, "white"), (1.0, "red")])

def _metrics_table(contents: List, styles: Dict, metrics: Dict, title: str) -> None:
    if not metrics:
        contents.append(Paragraph(f"No {title.lower()} metrics provided.", styles["Italic"]))
        return
    
    def make_table(data):
        t = Table(data, hAlign="CENTER")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        return t

    contents.append(Paragraph(title, styles["Heading2"]))

    metric_tables = []
    data = [["Metric", "Value"]]
    for k, v in metrics.items():
        if isinstance(v, dict):
            metric_tables.append(make_table([[str(k), "Value"]] + [[str(i), str(j)] for i, j in v.items()]))
        else:
            data.append([str(k), str(v)])
    if len(data) >1:
        metric_tables.append(make_table(data))

    # chunk into rows of 4
    rows = [metric_tables[i:i+4] for i in range(0, len(metric_tables), 4)]

    for row in rows:
        while len(row) < 4:
            row.append(Spacer(1, 0.2 * inch))

    outer = Table(rows, colWidths=[1.75*inch]*4)
    outer.setStyle(TableStyle([
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))

    contents.append(outer)
    

def _figure_to_image_contents(fig: Figure, styles:Dict, caption: str, width=6.5, height=3.5) -> List[Any]:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img = Image(buf, width=width * inch, height=height * inch, mask="auto")
    elements: List[Any] = [img]
    elements.append(Paragraph("<b>"+caption+"</b>", styles["NormalCenter"]))
    return elements

def _genlineplot_to_figure(glp: "GenLinePlot") -> Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(glp.x, glp.y, **glp.kwargs)
    ax.set_title(glp.title)
    ax.set_xlabel(glp.x_label)
    ax.set_ylabel(glp.y_label)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig

def _scene_class_heatmaps(posterior: NDArray, rgb: NDArray, class_names: List[str]) -> Figure:
    num_classes = len(class_names)
    ncols = 2
    total_panels = 1 + num_classes # +1 for the RGB
    nrows = math.ceil(total_panels / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 8*nrows))
    axes_flat = axes.ravel() if isinstance(axes, np.ndarray) else np.array([axes])

    # Top-left: RGB image
    # rgb_img = plt.imread("/Users/makiper/Desktop/emit/cover-class/src/cover_class/reporting/qualitative/emit20250404t194611_o09413_s001_l2a_rfl_b0106_v01.png")
    ax_rgb = axes_flat[0]
    # ax_rgb.imshow(rgb_img)
    ax_rgb.imshow(rgb)
    ax_rgb.set_title("RGB Image", fontsize=20)
    ax_rgb.axis("off")

    # Heatmaps for each class
    for class_idx in range(num_classes):
        ax_idx = class_idx + 1  # +1 because RGB
        ax = axes_flat[ax_idx]
        prob_map = posterior[:, :, class_idx]

        # Heatmap normalized between 0 and 1
        im = ax.imshow(prob_map, vmin=0.0, vmax=1.0, cmap=rwb)
        ax.set_title(f"{class_names[class_idx]} probabilities", fontsize=20)
        ax.axis("off")

        # Individual colorbar on the right of each heatmap
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Probability", rotation=90)

    # Hide any unused subplots
    for idx in range(total_panels, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.tight_layout()
    return fig

def _add_styles():
    styles = getSampleStyleSheet()
    if "HeadingCenter" not in styles.byName:
        styles.add(ParagraphStyle(name="HeadingCenter", parent=styles["Heading1"], alignment=1, spaceAfter=12))
    if "Small" not in styles.byName:
        styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8, leading=10))
    if "NormalCenter" not in styles.byName:
        styles.add(ParagraphStyle(name="NormalCenter", parent=styles["Normal"], alignment=1))
    if "Code" not in styles.byName:
        styles.add(ParagraphStyle(name="Code", parent=styles["Normal"], fontName="Courier", fontSize=8, leading=10))
    return styles


def generate_pdf_report(
    report_config: "Report",
    pdf_path: str
) -> None:

    m = 0.75*inch
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=LETTER,
        rightMargin=m,
        leftMargin=m,
        topMargin=m,
        bottomMargin=m,
        title=f"Model Report: {report_config.model_config.model_name} - {report_config.timestamp}",
        author=report_config.author,
    )
    report_styles = _add_styles()

    contents: List[Any] = []

    ######## Model Report - general intro: ########
    contents.append(Paragraph(f"Model Report", report_styles["Title"]))
    contents.append(Spacer(1, 0.3 * inch))
    contents.append(Paragraph(f"<b>Author:</b> {str(report_config.author)}", report_styles["NormalCenter"]))
    contents.append(Paragraph(f"<b>Model:</b>  {str(report_config.model_config.model_name)}", report_styles["NormalCenter"]))
    contents.append(Spacer(1, 0.1 * inch))
    contents.append(Paragraph(f"<b>Timestamp:</b> {str(report_config.timestamp)}", report_styles["Normal"]))
    contents.append(Paragraph(f"<b>Weights and Biases Run:</b> {str(report_config.wandb_link)}", report_styles["Normal"]))
    contents.append(Spacer(1, 0.3 * inch))
    ###############################################


    #################### Notes: ###################
    if report_config.notes:
        contents.append(Paragraph("<b>Notes:</b>", report_styles["Heading2"]))
        contents.append(Paragraph(report_config.notes.replace("\n", "<br/>"), report_styles["Normal"]))
        contents.append(Spacer(1, 0.5 * inch))
    ###############################################


    ################# Model info: #################
    contents.append(Paragraph(f"Model: {report_config.model_config.model_name}", report_styles["Heading1"]))
    if report_config.model_config.version:
        contents.append(Paragraph(f"<b>Version:</b> {report_config.model_config.version}", report_styles["Normal"]))
    if report_config.model_config.tags:
        contents.append(Paragraph(f"<b>Model Tags:</b> {', '.join(map(str, report_config.model_config.tags))}", report_styles["Normal"]))
    contents.append(Spacer(1, 0.25 * inch))

    data = [["Hyperparameter", "Value"]]
    for k, v in report_config.model_config.hyperparams.items():
        data.append([str(k), str(v)])
    table = Table(data, hAlign="CENTER", colWidths=[2.5 * inch, 3 * inch])
    table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ]))
    contents.append(table)
    contents.append(PageBreak())
    ###############################################


    ############### Training report: ##############
    fn = 0
    contents.append(Paragraph("Training Report", report_styles["Heading1"]))
    contents.append(Spacer(1, 0.1 * inch))
    if report_config.train_metric_table:
        _metrics_table(contents, report_styles, report_config.train_metric_table, "Training Metrics")
        contents.append(Spacer(1, 0.1 * inch))
    contents.append(Paragraph("Training Plots", report_styles["Heading2"]))
    contents.append(Spacer(1, 0.1 * inch))

    for i, glp in enumerate(report_config.train_plots, start=1):
        fn += 1
        fig = _genlineplot_to_figure(glp)
        contents.extend(_figure_to_image_contents(fig, report_styles, caption=f"[Train] Figure {fn}: {glp.title}"))
        contents.append(Spacer(1, 0.1 * inch))
        plt.close(fig)

    for i, fig in enumerate(report_config.train_figures, start=1):
        fn += 1
        contents.extend(_figure_to_image_contents(fig, report_styles, caption=f"[Train] Figure {fn}"))
        contents.append(Spacer(1, 0.5 * inch))
    contents.append(PageBreak())
    ###############################################


    ############### Testing report: ###############
    contents.append(Paragraph("Testing Report", report_styles["Heading1"]))
    contents.append(Spacer(1, 0.1 * inch))
    if report_config.test_metric_table:
        _metrics_table(contents, report_styles, report_config.test_metric_table, "Test Metrics")
        contents.append(Spacer(1, 0.1 * inch))
    if report_config.fractional_simulation_test_results:
        _metrics_table(contents, report_styles, report_config._fractional_simulation_test_dict['TPR'], "Fractional Simulation TPR")
        contents.append(Spacer(1, 0.1 * inch))
        _metrics_table(contents, report_styles, report_config._fractional_simulation_test_dict['FPR'], "Fractional Simulation FPR")
        contents.append(Spacer(1, 0.1 * inch))
    contents.append(Paragraph("Testing Plots", report_styles["Heading2"]))
    contents.append(Spacer(1, 0.1 * inch))

    for i, glp in enumerate(report_config.test_plots, start=1):
        fn += 1
        fig = _genlineplot_to_figure(glp)
        contents.extend(_figure_to_image_contents(fig, report_styles, caption=f"[Test] Figure {fn}: {glp.title}"))
        contents.append(Spacer(1, 0.1 * inch))
        plt.close(fig)

    for i, fig in enumerate(report_config.test_figures, start=1):
        fn += 1
        contents.extend(_figure_to_image_contents(fig, report_styles, caption=f"[Test] Figure {fn}"))
        contents.append(Spacer(1, 0.5 * inch))
    contents.append(PageBreak())
    ###############################################


    ############# Qualitative Testing: ############
    contents.append(Paragraph("Testing Qualitative Scene Report", report_styles["Heading1"]))
    drop_wls: List[Any] = report_config.config['drop-bands-wavelengths'] #type: ignore
    ds: Dict = report_config.config['datasets'] #type: ignore
    class_names = [str(c) for c in ds.keys() if ds[c] is not None and len(ds[c])]
    for scene in report_config.qualitative_testing_scenes_paths:
        try:
            scene_posterior = inference_over_scene(scene, report_config.model_config.model, drop_wls)
            rgb = rgb_from_scene(scene)
            fig = _scene_class_heatmaps(scene_posterior, rgb, class_names)
            scene_name = os.path.basename(scene)
            contents.extend(_figure_to_image_contents(fig, report_styles, caption=scene_name, width=7.5, height=9))
        except Exception as e:
            warnings.warn(f"Failed to do inference over scene: {scene}\nException: {e}")
            continue
    contents.append(PageBreak())
    ###############################################


    ############## Build the Report ###############
    def _add_page_number(canvas, document):
        canvas.saveState()
        footer_text = f"{report_config.model_config.model_name} – Page {document.page}"
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(
            document.pagesize[0] - 0.75 * inch, 0.5 * inch, footer_text
        )
        canvas.restoreState()

    doc.build(contents, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    ###############################################
