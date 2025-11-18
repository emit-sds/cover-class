from typing import Any, List, Optional, Dict
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from glob import glob
import os
import io

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

@dataclass
class GenLinePlot:
    title: str
    x: List[Any]
    y: List[Any]
    x_label: str
    y_label: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelConfig:
    model: Any
    model_name: str
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    version: Optional[str] = None
    tags: Optional[List[str]] = None

@dataclass
class Report:
    author: str
    model_config: ModelConfig
    train_plots: List[GenLinePlot] = field(default_factory=list)
    test_plots: List[GenLinePlot] = field(default_factory=list)
    train_figures: List[Figure] = field(default_factory=list)
    test_figures: List[Figure] = field(default_factory=list)
    train_metric_table: Optional[dict] = None
    test_metric_table: Optional[dict] = None
    wandb_link: Optional[str] = None
    random_seed: Optional[int] = None
    git_commit: Optional[str] = None
    notes: Optional[str] = None
    timestamp: Optional[str] = None
    qualitative_testing_scenes_paths: List[str] = field(default_factory=list)


    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        if not self.qualitative_testing_scenes_paths:
            self.qualitative_testing_scenes_paths = glob(str(resources.files(__package__) / "qualitative/*.img"))

        styles = getSampleStyleSheet()
        if "HeadingCenter" not in styles.byName:
            styles.add(ParagraphStyle(name="HeadingCenter", parent=styles["Heading1"], alignment=1, spaceAfter=12))
        if "Small" not in styles.byName:
            styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8, leading=10))
        if "NormalCenter" not in styles.byName:
            styles.add(ParagraphStyle(name="NormalCenter", parent=styles["Normal"], alignment=1))
        if "Code" not in styles.byName:
            styles.add(ParagraphStyle(name="Code", parent=styles["Normal"], fontName="Courier", fontSize=8, leading=10))
        self.styles = styles


    @staticmethod
    def _metrics_table(contents: List, styles: Dict, metrics: Dict, title: str) -> None:
        if not metrics:
            contents.append(Paragraph(f"No {title.lower()} metrics provided.", styles["Italic"]))
            return

        contents.append(Paragraph(title, styles["Heading2"]))
        data = [["Metric", "Value"]]
        for k, v in metrics.items():
            data.append([str(k), str(v)])
        t = Table(data, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        contents.append(t)


    @staticmethod
    def _figure_to_image_contents(fig: Figure, styles:Dict, caption: str) -> List[Any]:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        img = Image(buf, width=6.5 * inch, height=3.5 * inch,mask="auto")
        elements: List[Any] = [img]
        elements.append(Paragraph("<b>"+caption+"</b>", styles["NormalCenter"]))
        return elements


    @staticmethod
    def _genlineplot_to_figure(glp: GenLinePlot) -> Figure:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(glp.x, glp.y, **glp.kwargs)
        ax.set_title(glp.title)
        ax.set_xlabel(glp.x_label)
        ax.set_ylabel(glp.y_label)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        return fig


    def generate_report(
        self,
        outdir: str
    ) -> None:
        os.makedirs(outdir, exist_ok=True)
        filename = f"{self.model_config.model_name}_{str(self.timestamp).replace(":", "-")}.pdf"
        pdf_path = os.path.join(outdir, filename)

        m = 0.75*inch
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=LETTER,
            rightMargin=m,
            leftMargin=m,
            topMargin=m,
            bottomMargin=m,
            title=f"Model Report: {self.model_config.model_name} - {self.timestamp}",
            author=self.author,
        )

        contents: List[Any] = []

        ######## Model Report - general intro: ########
        contents.append(Paragraph(f"Model Report", self.styles["Title"]))
        contents.append(Spacer(1, 0.3 * inch))
        contents.append(Paragraph(f"<b>Author:</b> {str(self.author)}", self.styles["NormalCenter"]))
        contents.append(Paragraph(f"<b>Model:</b>  {str(self.model_config.model_name)}", self.styles["NormalCenter"]))
        contents.append(Spacer(1, 0.1 * inch))
        contents.append(Paragraph(f"<b>Timestamp:</b> {str(self.timestamp)}", self.styles["Normal"]))
        contents.append(Paragraph(f"<b>Git Commit:</b> {str(self.git_commit)}", self.styles["Normal"]))
        contents.append(Paragraph(f"<b>Weights and Biases Run:</b> {str(self.wandb_link)}", self.styles["Normal"]))
        contents.append(Spacer(1, 0.3 * inch))
        ###############################################


        #################### Notes: ###################
        if self.notes:
            contents.append(Paragraph("<b>Notes:</b>", self.styles["Heading2"]))
            contents.append(Paragraph(self.notes.replace("\n", "<br/>"), self.styles["Normal"]))
            contents.append(Spacer(1, 0.5 * inch))
        ###############################################


        ################# Model info: #################
        contents.append(Paragraph(f"Model: {self.model_config.model_name}", self.styles["Heading1"]))
        if self.model_config.version:
            contents.append(Paragraph(f"<b>Version:</b> {self.model_config.version}", self.styles["Normal"]))
        if self.model_config.tags:
            contents.append(Paragraph(f"<b>Model Tags:</b> {', '.join(map(str, self.model_config.tags))}", self.styles["Normal"]))
        contents.append(Spacer(1, 0.25 * inch))

        data = [["Hyperparameter", "Value"]]
        for k, v in self.model_config.hyperparams.items():
            data.append([str(k), str(v)])
        table = Table(data, hAlign="LEFT")
        table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ]))
        contents.append(table)
        contents.append(PageBreak())
        ###############################################


        ############### Training report: ##############
        fn = 0
        contents.append(Paragraph("Training Report", self.styles["Heading1"]))
        contents.append(Spacer(1, 0.1 * inch))
        if self.train_metric_table:
            self._metrics_table(contents, self.styles, self.train_metric_table, "Training Metrics")
            contents.append(Spacer(1, 0.1 * inch))
        contents.append(Paragraph("Training Plots", self.styles["Heading2"]))
        contents.append(Spacer(1, 0.1 * inch))

        for i, glp in enumerate(self.train_plots, start=1):
            fn += 1
            fig = self._genlineplot_to_figure(glp)
            contents.extend(self._figure_to_image_contents(fig, self.styles, caption=f"[Train] Figure {fn}: {glp.title}"))
            contents.append(Spacer(1, 0.1 * inch))
            plt.close(fig)

        for i, fig in enumerate(self.train_figures, start=1):
            fn += 1
            contents.extend(self._figure_to_image_contents(fig, self.styles, caption=f"[Train] Figure {fn}"))
            contents.append(Spacer(1, 0.1 * inch))
        contents.append(PageBreak())
        ###############################################


        ############### Testing report: ###############
        contents.append(Paragraph("Testing Report", self.styles["Heading1"]))
        contents.append(Spacer(1, 0.1 * inch))
        if self.test_metric_table:
            self._metrics_table(contents, self.styles, self.test_metric_table, "Test Metrics")
            contents.append(Spacer(1, 0.1 * inch))
        contents.append(Paragraph("Testing Plots", self.styles["Heading2"]))
        contents.append(Spacer(1, 0.1 * inch))

        for i, glp in enumerate(self.test_plots, start=1):
            fn += 1
            fig = self._genlineplot_to_figure(glp)
            contents.extend(self._figure_to_image_contents(fig, self.styles, caption=f"[Test] Figure {fn}: {glp.title}"))
            contents.append(Spacer(1, 0.1 * inch))
            plt.close(fig)

        for i, fig in enumerate(self.test_figures, start=1):
            fn += 1
            contents.extend(self._figure_to_image_contents(fig, self.styles, caption=f"[Test] Figure {fn}"))
            contents.append(Spacer(1, 0.1 * inch))
        ###############################################

        ############# Qualitative Testing: ############
        contents.append(PageBreak())
        ###############################################

        ############## Build the Report ###############
        def _add_page_number(canvas, document):
            canvas.saveState()
            footer_text = f"{self.model_config.model_name} – Page {document.page}"
            canvas.setFont("Helvetica", 8)
            canvas.drawRightString(
                document.pagesize[0] - 0.75 * inch, 0.5 * inch, footer_text
            )
            canvas.restoreState()

        doc.build(contents, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
        ###############################################

