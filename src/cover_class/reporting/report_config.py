from typing import Any, List, Optional, Dict, Union
from matplotlib.figure import Figure
from dataclasses import dataclass, field
from datetime import datetime
from numpy.typing import NDArray
import torch
from torch import Tensor
import getpass
import os
import numpy as np

from cover_class.utils import read_config
from cover_class.reporting.metrics import (
    confusion_matrix, 
    roc_auc, 
    missed_class_confusion, 
    tpr_fpr, 
    f_beta_scores,
    f1_opt_thr
)
from cover_class.reporting.json_report import generate_json_report
from cover_class.reporting.pdf_report import generate_pdf_report
from cover_class.reporting.download_scenes import download_scenes
from cover_class.reporting.utils import make_numpy
from cover_class.simulation.force_fractions import ForcedFractionSimulation

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
class FractionalSimulationResult:
    range_low: float
    range_high: float
    class_id: int
    y_hat: Tensor
    y: Tensor

@dataclass
class Report:
    outdir: str
    config: str|Dict
    model_config: ModelConfig
    Y_test: Union[Tensor, NDArray]
    Y_ood_test: Optional[Union[Tensor, NDArray]] = None
    train_plots: List[GenLinePlot] = field(default_factory=list)
    test_plots: List[GenLinePlot] = field(default_factory=list)
    ood_test_plots: List[GenLinePlot] = field(default_factory=list)
    train_figures: List[Figure] = field(default_factory=list)
    test_figures: List[Figure] = field(default_factory=list)
    ood_test_figures: List[Figure] = field(default_factory=list)
    train_metric_table: Optional[dict] = None
    test_metric_table: Optional[dict] = None
    ood_test_metric_table: Optional[dict] = None
    fractional_simulation_test_results: List[FractionalSimulationResult] = field(default_factory=list)
    author: Optional[str] = None
    wandb_link: Optional[str] = None
    random_seed: Optional[int] = None
    notes: Optional[str] = None
    timestamp: Optional[str] = None
    run_name: Optional[str] = None
    qualitative_testing_scenes_paths: List[str] = field(default_factory=list)
    _download_missing_qualitative_testing_scenes_from_config: bool = True
    _fractional_simulation_test_dict: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.Y_ood_test is not None:
            a,b = self.Y_test.shape[1], self.Y_ood_test.shape[1]
            assert a == b, f"Got mismatched number of classes for Y_test ({a}) and Y_ood_test ({b})"

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
        
        self.y_hat = np.zeros_like(self.Y_test)

    def make_report(self, y_hat: Tensor, class_thresholds: List[float], y_hat_ood_test: Optional[Tensor] = None):
        # 1. Get Metrics
        ds: Dict = self.config['datasets'] # type: ignore
        class_names = [str(c) for c in ds.keys() if ds[c] is not None and len(ds[c])]

        # Calculate the optimal F1 thresholds if not provided
        if class_thresholds is None:
            test_thresholds = f1_opt_thr(y_hat, self.Y_test, class_names)
            class_thresholds = test_thresholds
            ood_thresholds = f1_opt_thr(y_hat_ood_test, self.Y_ood_test, class_names) if y_hat_ood_test is not None else None
        else:
            test_thresholds = class_thresholds
            ood_thresholds = class_thresholds

        if self.test_metric_table is None:
            self.test_metric_table = {}
        if self.ood_test_metric_table is None:
            self.ood_test_metric_table = {}
        self.test_metric_table.update(self.generate_metrics(self.Y_test, y_hat, test_thresholds, self.test_figures, class_names))
        if y_hat_ood_test is not None:
            assert self.Y_ood_test is not None, "Got probabilities for OOD Test set, but no labels were provided"
            self.ood_test_metric_table.update(self.generate_metrics(self.Y_ood_test, y_hat_ood_test, ood_thresholds, self.ood_test_figures, class_names))

        # Get the metrics for the fractional simulation results
        # TODO: what is this? -Jake
        self._fractional_simulation_test_dict['TPR'] = {}
        self._fractional_simulation_test_dict['FPR'] = {}
        for i, thresh in enumerate(class_thresholds):
            class_name = class_names[i]
            self._fractional_simulation_test_dict['TPR'][class_name] = {}
            self._fractional_simulation_test_dict['FPR'][class_name] = {}
            for f in self.fractional_simulation_test_results:
                if f.class_id != i:
                    continue
                f_y_hat_binary = (f.y_hat >= torch.tensor(thresh, device=f.y_hat.device, dtype=f.y_hat.dtype)).to(torch.long)
                rates = tpr_fpr(f_y_hat_binary, f.y, class_names)[class_name]
                rname = f'{f.range_low*100} - {f.range_high*100} %'
                self._fractional_simulation_test_dict['TPR'][class_name][rname] = rates['TPR']
                self._fractional_simulation_test_dict['FPR'][class_name][rname] = rates['FPR']

        # 2. Generate Report
        os.makedirs(self.outdir, exist_ok=True)
        rn = self.run_name if self.run_name else str(self.timestamp).replace(":", "-")
        pdf_path = os.path.join(self.outdir, f"{self.model_config.model_name}_{rn}.pdf")
        json_path = pdf_path.replace('.pdf', '.json')
        generate_pdf_report(self, pdf_path)
        generate_json_report(self, json_path)

    def generate_metrics(self, y: Union[Tensor, NDArray], y_hat: Union[Tensor, NDArray], class_thresholds: List[float], figure_list: List[Figure], class_names: List[str]) -> dict:
        if class_thresholds is None:
            class_thresholds = f1_opt_thr(y_hat, y, class_names)

        assert len(class_thresholds) == y_hat.shape[-1], f"Got {len(class_thresholds)} thresholds for {y_hat.shape[-1]} classes"

        y = make_numpy(y)
        y_hat = make_numpy(y_hat)
        y_hat_binary = (y_hat >= np.array(class_thresholds)).astype(int)

        lcn = len(class_names)
        
        assert y_hat.shape[-1] == lcn, f"Got {y_hat.shape[-1]} classes in y_hat, but {lcn} classes from the config"
        assert len(class_thresholds) == lcn, f"Got {len(class_thresholds)} class thresholds, but {lcn} classes from the config"

        cm_plot            = confusion_matrix(y_hat_binary, y, class_names)
        mcc_plot           = missed_class_confusion(y_hat_binary, y, class_names)
        rates              = tpr_fpr(y_hat_binary, y, class_names)
        f1_scores          = f_beta_scores(y_hat_binary, y, class_names)
        roc_plot, roc_dict = roc_auc(y_hat, y, class_names)
        # zip together the metric dicts
        metrics: Dict = {class_names[i]:{'Threshold': class_thresholds[i]} for i in range(lcn)}
        for m in [rates, f1_scores, roc_dict]:
            for k, v in m.items():
                metrics[k].update(v)

        figure_list.extend([cm_plot, mcc_plot, roc_plot])
        return metrics

    def append_fractional_simulation_result(self, f: ForcedFractionSimulation, y_hat: Tensor):
        if f.latest_simulation_labels is None:
            raise RuntimeWarning("Cannot associate proper simulation labels to append results data")
        fr = FractionalSimulationResult(f.ranges[f.range_idx][0], f.ranges[f.range_idx][1], f.class_idx, y_hat, f.latest_simulation_labels)
        self.fractional_simulation_test_results.append(fr)

