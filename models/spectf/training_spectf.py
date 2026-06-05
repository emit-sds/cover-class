import os

# Set CUBLAS_WORKSPACE_CONFIG for deterministic CUDA behavior
# pylint: disable=wrong-import-position
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import getpass
from datetime import datetime
import rich_click as click
import yaml
import wandb
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import schedulefree

from cover_class.train import setup_training_from_config, make_simulation_test_set, banddef_from_config #type: ignore
from cover_class.utils import seed as sseed, ood_test_set_from_config #type: ignore
from cover_class.reporting import ModelConfig, Report #type: ignore

from spectf.model import SpecTfEncoder
from spectf.utils import get_device

ENV_VAR_PREFIX = 'COVER_CLASS_TRAIN_'

class TestDataset(Dataset):
    def __init__(self, test_X, test_Y):
        super().__init__()

        self.test_X = test_X
        self.test_Y = test_Y
    
    def __len__(self):
        return len(self.test_Y)

    def __getitem__(self, idx):
        return self.test_X[idx], self.test_Y[idx]


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()

        return focal_loss

@click.command()
@click.option(
    "--outdir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Output file directory.",
    envvar=f'{ENV_VAR_PREFIX}OUTDIR'
)
@click.option(
    "--data-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the YAML config for the dataloader.",
    envvar=f'{ENV_VAR_PREFIX}_DATA_CONFIG'
)
@click.option(
    "--model-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the YAML config for the model architecture.",
    envvar=f'{ENV_VAR_PREFIX}_MODEL_CONFIG'
)
@click.option(
    "--simulated-test-set-size",
    required=False,
    type=int,
    default=100_000,
    help="Number of rows to generate for the simulated test set.",
    envvar=f'{ENV_VAR_PREFIX}_SIMULATED_TEST_SET_SIZE'
)
@click.option(
    "--focal-alpha",
    required=False,
    default="0.25",
    help="Focal loss alpha parameter. Set to 'None' to disable.",
    envvar=f'{ENV_VAR_PREFIX}_FOCAL_ALPHA'
)
@click.option(
    "--focal-gamma",
    required=False,
    type=float,
    default=2.0,
    help="Focal loss gamma parameter.",
    envvar=f'{ENV_VAR_PREFIX}_FOCAL_GAMMA'
)
def run_pipeline_classifier(
        outdir: str,
        data_config: str,
        model_config: str,
        simulated_test_set_size: int = 100_000,
        focal_alpha: str = "0.25",
        focal_gamma: float = 2.0
    ):

    with open(model_config, 'r', encoding='utf-8') as f:
        m_config = yaml.safe_load(f)

    dataloader, test_X, test_Y = setup_training_from_config(
        data_config,
        m_config['batch_size'],
        shuffle=True,
        seed=m_config['random_seed'],
        subsampled_files_outdir=outdir,
        misc_dataloader_params={'num_workers': m_config['training']['num_workers']})

    banddef = banddef_from_config(data_config)

    # create simulation eval set
    sseed(m_config['random_seed'])
    simulation_x_test, simulation_y_test, _ = make_simulation_test_set(dataloader, test_X, test_Y, simulated_test_set_size)

    # Test set dataloader
    test_dataset = TestDataset(simulation_x_test, simulation_y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=m_config['batch_size'], shuffle=False)

    # Validation set dataloader for OOD evaluation
    ood_test_set_x, ood_test_set_y = ood_test_set_from_config(data_config)
    ood_dataset = TestDataset(ood_test_set_x, ood_test_set_y)
    ood_dataloader = DataLoader(ood_dataset, batch_size=m_config['batch_size'], shuffle=False)

    # hardcoded GPU 0
    device = get_device(0)

    # model definition
    model = SpecTfEncoder(banddef.to(dtype=torch.float32, device=device),
                         dim_output=m_config['model']['dim_output'],
                         num_heads=m_config['model']['num_heads'],
                         dim_proj=m_config['model']['dim_proj'],
                         dim_ff=m_config['model']['dim_ff'],
                         dropout=m_config['model']['dropout'],
                         agg=m_config['model']['agg'],
                         use_residual=m_config['model']['use_residual'],
                         num_layers=m_config['model']['num_layers']).to(device)

    # criterion = nn.BCEWithLogitsLoss()
    alpha_val = None if focal_alpha == "None" else float(focal_alpha)
    if alpha_val is None and focal_gamma == 0.0:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = FocalLoss(alpha=alpha_val, gamma=focal_gamma)

    optimizer = schedulefree.AdamWScheduleFree(
        (p for p in model.parameters() if p.requires_grad),
        lr=m_config['training']['learning_rate'],
        warmup_steps=m_config['training']['warmup_steps']
    )

    # W&B
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = wandb.init(
        entity=m_config['wandb']['entity'],
        project=m_config['wandb']['project'],
        name=timestamp,
        dir='./',
        config={
            "outdir": outdir,
            "data_config": data_config,
            "model_config": model_config,
            "simulated_test_set_size": simulated_test_set_size,
            "focal_alpha": focal_alpha,
            "focal_gamma": focal_gamma,
        },
        settings=wandb.Settings(_service_wait=300)
    )

    report = Report(
        outdir=outdir,
        config=data_config,
        author=getpass.getuser(),
        model_config=ModelConfig(
            model=model,
            model_name=SpecTfEncoder.__name__,
            hyperparams={
                "learning_rate": m_config['training']['learning_rate'],
                "batch_size": m_config['batch_size'],
                "optimizer": optimizer.__class__.__name__,
                "focal_alpha": alpha_val,
                "focal_gamma": focal_gamma,
                "params": m_config['model']
            },
        ),
        Y_test=simulation_y_test,
        Y_ood_test=ood_test_set_y,
        random_seed=m_config['random_seed'],
        run_name=timestamp
    )

    # Epoch-based training loop
    total_epochs = m_config['training']['total_epochs']
    steps_per_epoch = m_config['training']['steps_per_epoch']
    bs = m_config['batch_size']

    ds = report.config['datasets'] # type: ignore
    class_names = [str(c) for c in ds.keys() if ds[c] is not None and len(ds[c])]

    for epoch in range(total_epochs):
        model.train()
        optimizer.train()
        epoch_loss = 0.0
        step = 0
        for batch_X, batch_Y in dataloader:
            batch_X = batch_X.to(device=device, dtype=torch.float32)
            batch_X = torch.unsqueeze(batch_X, -1)
            batch_Y = batch_Y.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_Y)
            loss.backward()
            optimizer.step()

            nats = loss.cpu().item()
            epoch_loss += nats
            run.log({"loss_train": nats, "step": step + epoch * steps_per_epoch})

            step += 1
            if step >= steps_per_epoch:
                break

        avg_epoch_loss = epoch_loss / steps_per_epoch
        run.log({"loss_train_epoch": avg_epoch_loss, "epoch": epoch})

        # Save model at each epoch
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(outdir, f'model_epoch{epoch+1}.pth'))

        # Test loop
        model.eval()
        optimizer.eval()
        y_hat = np.zeros_like(simulation_y_test, dtype=float)
        test_loss_sum = 0.0
        test_batches = 0
        for i, (batch_X, batch_Y) in enumerate(test_dataloader):
            batch_X = batch_X.to(device=device, dtype=torch.float32)
            batch_X = torch.unsqueeze(batch_X, -1)
            batch_Y = batch_Y.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                logits = model(batch_X)
                batch_loss = criterion(logits, batch_Y)
                test_loss_sum += batch_loss.cpu().item()
                test_batches += 1
                batch_y_hat = torch.sigmoid(logits)
                batch_y_hat = batch_y_hat.detach().cpu().numpy().astype(float)
                batch_len = len(batch_y_hat)
                y_hat[i*bs:i*bs+batch_len] = batch_y_hat
        avg_test_loss = test_loss_sum / test_batches if test_batches > 0 else float('nan')

        # OOD loop
        y_hat_ood = np.zeros_like(ood_test_set_y, dtype=float)
        ood_loss_sum = 0.0
        ood_batches = 0
        for i, (batch_X, batch_Y) in enumerate(ood_dataloader):
            batch_X = batch_X.to(device=device, dtype=torch.float32)
            batch_X = torch.unsqueeze(batch_X, -1)
            batch_Y = batch_Y.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                logits = model(batch_X)
                mask = ~torch.isnan(batch_Y)
                if mask.any():
                    batch_loss = criterion(logits[mask], batch_Y[mask])
                    ood_loss_sum += batch_loss.cpu().item()
                    ood_batches += 1
                batch_y_hat = torch.sigmoid(logits)
                batch_y_hat = batch_y_hat.detach().cpu().numpy().astype(float)
                batch_len = len(batch_y_hat)
                y_hat_ood[i*bs:i*bs+batch_len] = batch_y_hat

        avg_ood_loss = ood_loss_sum / ood_batches if ood_batches > 0 else float('nan')


        # Calculate test set metrics using the best thresholds for the test set
        _figs = []
        test_metrics = report.generate_metrics(simulation_y_test, y_hat, None, _figs, class_names)
        for f in _figs: plt.close(f)
        del _figs

        # Extract the thresholds used for the test set 
        test_thresholds = [test_metrics[class_name]['Threshold'] for class_name in class_names]

        # Calculate OOD validation set metrics using the test set thresholds
        _figs = []
        ood_metrics = report.generate_metrics(ood_test_set_y, y_hat_ood, test_thresholds, _figs, class_names)
        for f in _figs: plt.close(f)
        del _figs

        run.log({
            "test_metrics": test_metrics,
            "ood_metrics": ood_metrics,
            "epoch": epoch,
            "loss_test_epoch": avg_test_loss,
            "loss_ood_epoch": avg_ood_loss
        })

    # Generate report at end of training
    report.make_report(y_hat, y_hat_ood, None, ood_overfit=False)

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run_pipeline_classifier()