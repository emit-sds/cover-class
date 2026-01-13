from typing import Dict, Any, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from cover_class.reporting import Report

def generate_json_report(
    report_config: "Report",
    json_path: str
) -> None:
    out: Dict[str, Any] = {}
    out.update({'Author':report_config.author})
    out.update({'Timestamp':report_config.timestamp})
    out.update({'W&B Link':report_config.wandb_link})
    out.update({'Seed':report_config.random_seed})
    out.update({'Notes':report_config.notes})
    out.update({'Train':report_config.train_metric_table})
    out.update({'Test':{'metrics':report_config.test_metric_table, 'fractional simulation':report_config._fractional_simulation_test_dict}})
    out.update({'Model':
        {
            'Name':report_config.model_config.model_name,
            'Version':report_config.model_config.version,
            'Tags':report_config.model_config.tags,
            'Hyperparameters':report_config.model_config.hyperparams,
        }
    })
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=4)
