from typing import Any, List, Optional, Dict, Union
from matplotlib.figure import Figure
from dataclasses import dataclass, field
from datetime import datetime
from numpy.typing import NDArray
from torch import Tensor
import torch
import getpass
import os

from cover_class.utils import read_config
from cover_class.reporting.metrics import (
    confusion_matrix, 
    roc_auc, 
    missed_class_confusion, 
    tpr_fpr, 
    f_beta_scores,
)
from cover_class.reporting.json_report import generate_json_report
from cover_class.reporting.pdf_report import generate_pdf_report
from cover_class.reporting.download_scenes import download_scenes
from cover_class.reporting.utils import make_numpy

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
    outdir: str
    config: str|Dict
    model_config: ModelConfig
    X_test: Union[Tensor, NDArray]
    Y_test: Union[Tensor, NDArray]

    classification_threshold: float = 0.5
    train_plots: List[GenLinePlot] = field(default_factory=list)
    test_plots: List[GenLinePlot] = field(default_factory=list)
    train_figures: List[Figure] = field(default_factory=list)
    test_figures: List[Figure] = field(default_factory=list)
    train_metric_table: Optional[dict] = None
    test_metric_table: Optional[dict] = None
    author: Optional[str] = None
    wandb_link: Optional[str] = None
    random_seed: Optional[int] = None
    notes: Optional[str] = None
    timestamp: Optional[str] = None
    qualitative_testing_scenes_paths: List[str] = field(default_factory=list)
    _download_missing_qualitative_testing_scenes_from_config: bool = True

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        if isinstance(self.config, str):
            self.config = read_config(self.config)

        if self.author is None:
            self.author = getpass.getuser()

        # finally, make sure that all of the qualitative scenes are installed
        if len(self.qualitative_testing_scenes_paths):
            for i, f in enumerate(self.qualitative_testing_scenes_paths):
                assert f.endswith(".nc") or f.endswith('.h5') or f.endswith('.hdf5'), f"File {f} is not supported - needs to be one of [.nc, .hdf5, .h5]"
                self.qualitative_testing_scenes_paths[i] = os.path.abspath(f)

        if self._download_missing_qualitative_testing_scenes_from_config:
            uris = read_config(self.config).get("test-scene-urls", [])
            self.qualitative_testing_scenes_paths.extend(download_scenes(uris))

    def __enter__(self):
        return self
    
    def __exit__(self, _, __, ___):
        self.make_report()

    def make_report(self):
        if isinstance(self.model_config.model, torch.nn.Module):
            self.model_config.model.eval()
            with torch.no_grad():
                y_hat = torch.nn.functional.softmax(self.model_config.model(self.X_test),dim=1)
        else:
            y_hat = torch.from_numpy(self.model_config.model(self.X_test))
            y_hat = torch.nn.functional.softmax(y_hat,dim=1)
        y_hat = (y_hat >= self.classification_threshold).to(torch.long)
        y_hat = make_numpy(y_hat)

        # 1. Get Metrics
        class_names = [str(c) for c in self.config['datasets'].keys()]
        cm_plot            = confusion_matrix(y_hat, self.Y_test, class_names)
        mcc_plot           = missed_class_confusion(y_hat, self.Y_test, class_names)
        rates              = tpr_fpr(y_hat, self.Y_test, class_names)
        f1_scores          = f_beta_scores(y_hat, self.Y_test, class_names)
        roc_plot, roc_dict = roc_auc(y_hat, self.Y_test, class_names)
        # zip together the metric dicts
        metrics = {c:{} for c in class_names}
        for m in [rates, f1_scores, roc_dict]:
            for k, v in m.items():
                metrics[k].update(v)

        self.test_figures.extend([cm_plot, mcc_plot, roc_plot])
        self.test_metric_table.update(metrics)

        # 2. Generate Report
        os.makedirs(self.outdir, exist_ok=True)
        pdf_path = os.path.join(self.outdir, f"{self.model_config.model_name}_{str(self.timestamp).replace(":", "-")}.pdf")
        json_path = pdf_path.replace('.pdf', '.json')
        generate_pdf_report(self, pdf_path)
        generate_json_report(self, json_path)

