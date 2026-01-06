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
    Y_test: Union[Tensor, NDArray]

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
    run_name: Optional[str] = None
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

    def make_report(self, y_hat: Tensor, class_thresholds: List[float]):
        assert len(class_thresholds) == y_hat.shape[-1], f"Got {len(class_thresholds)} thresholds for {y_hat.shape[-1]} classes"

        y_hat_binary = (y_hat >= torch.tensor(class_thresholds, device=y_hat.device, dtype=y_hat.dtype)).to(torch.long)
        y_hat_binary = make_numpy(y_hat_binary) # type: ignore
        y_hat = make_numpy(y_hat) # type: ignore

        # 1. Get Metrics
        ds: Dict = self.config['datasets'] # type: ignore
        class_names = [str(c) for c in ds.keys() if ds[c] is not None and len(ds[c])]
        
        assert y_hat.shape[-1] == len(class_names), f"Got {y_hat.shape[-1]} classes in y_hat, but {len(class_names)} classes from the config"
        assert len(class_thresholds) == len(class_names), f"Got {len(class_thresholds)} class thresholds, but {len(class_names)} classes from the config"

        cm_plot            = confusion_matrix(y_hat_binary, self.Y_test, class_names)
        mcc_plot           = missed_class_confusion(y_hat_binary, self.Y_test, class_names)
        rates              = tpr_fpr(y_hat_binary, self.Y_test, class_names)
        f1_scores          = f_beta_scores(y_hat_binary, self.Y_test, class_names)
        roc_plot, roc_dict = roc_auc(y_hat, self.Y_test, class_names)
        # zip together the metric dicts
        metrics: Dict = {class_names[i]:{'Threshold': class_thresholds[i]} for i in range(len(class_names))}
        for m in [rates, f1_scores, roc_dict]:
            for k, v in m.items():
                metrics[k].update(v)

        self.test_figures.extend([cm_plot, mcc_plot, roc_plot])
        if self.test_metric_table is None:
            self.test_metric_table = {}
        self.test_metric_table.update(metrics)

        # 2. Generate Report
        os.makedirs(self.outdir, exist_ok=True)
        rn = self.run_name if self.run_name else str(self.timestamp).replace(":", "-")
        pdf_path = os.path.join(self.outdir, f"{self.model_config.model_name}_{rn}.pdf")
        json_path = pdf_path.replace('.pdf', '.json')
        generate_pdf_report(self, pdf_path)
        generate_json_report(self, json_path)

