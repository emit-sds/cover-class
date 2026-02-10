import os
from typing import Optional
import getpass
from datetime import datetime
import matplotlib.pyplot as plt
import rich_click as click
import yaml
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
    help="Path to the YAML config for the dataloader.",
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
def run_pipeline_classifier(
        outdir: str,
        data_config: str,
        model_config: str,
        simulated_test_set_size: int = 100_000
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
    print("DEBUG", banddef.shape)

    # create simulation eval set
    sseed(m_config['random_seed'])
    simulation_x_test, simulation_y_test, _ = make_simulation_test_set(dataloader, test_X, test_Y, simulated_test_set_size)

    # Test set dataloader
    test_dataset = TestDataset(simulation_x_test, simulation_y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=m_config['batch_size'], shuffle=False)

    # Validation set dataloader for OOD evaluation
    ood_test_set_x, ood_test_set_y = ood_test_set_from_config(data_config)
    val_dataset = TestDataset(ood_test_set_x, ood_test_set_y)
    val_dataloader = DataLoader(val_dataset, batch_size=m_config['batch_size'], shuffle=False)

    # hardcoded GPU 0
    device = get_device(0)

    # model definition
    model = SpecTfEncoder(torch.tensor(banddef, dtype=torch.float32).to(device),
                         dim_output=m_config['model']['dim_output'],
                         num_heads=m_config['model']['num_heads'],
                         dim_proj=m_config['model']['dim_proj'],
                         dim_ff=m_config['model']['dim_ff'],
                         dropout=m_config['model']['dropout'],
                         agg=m_config['model']['agg'],
                         use_residual=m_config['model']['use_residual'],
                         num_layers=m_config['model']['num_layers']).to(device)

    criterion = nn.BCEWithLogitsLoss()
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
        settings=wandb.Settings(_service_wait=300)
    )

    threshold = 0.5
    accumulator = 0
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
                "optimizer": optim.AdamW.__name__,
                "params": m_config['model']
            },
        ),
        Y_test=simulation_y_test,
        Y_ood_test=ood_test_set_y,
        random_seed=m_config['random_seed'],
        run_name=timestamp
    )

    # Training loop
    model.train()
    optimizer.train()
    for batch_X, batch_Y in dataloader:
        batch_X = batch_X.to(device=device, dtype=torch.float32)
        batch_X = torch.unsqueeze(batch_X, -1)
        batch_Y = batch_Y.to(device)

        accumulator += 1
        if accumulator == m_config['training']['total_steps']:
            break

        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_Y)
        loss.backward()
        optimizer.step()

        nats = loss.cpu().item()

        run.log({"loss_train": nats})

    ## Save model
    torch.save(model.state_dict(), os.path.join(outdir, 'model.pth'))

    bs = m_config['batch_size']

    # Test loop
    model.eval()
    optimizer.eval()
    y_hat = np.zeros_like(simulation_y_test)
    for i, (batch_X, batch_Y) in enumerate(test_dataloader):
        batch_X = torch.tensor(batch_X, dtype=torch.float32).to(device)
        batch_X = torch.unsqueeze(batch_X, -1)
        with torch.no_grad():
            batch_y_hat = model(batch_X)
            batch_y_hat = torch.sigmoid(batch_y_hat) >= threshold
            batch_y_hat = batch_y_hat.detach().cpu().numpy().astype(int)
            y_hat[i*bs:(i+1)*bs] = batch_y_hat

    # Validation loop
    y_hat_ood = np.zeros_like(ood_test_set_y)
    for i, (batch_X, batch_Y) in enumerate(val_dataloader):
        batch_X = torch.tensor(batch_X, dtype=torch.float32).to(device)
        batch_X = torch.unsqueeze(batch_X, -1)
        with torch.no_grad():
            batch_y_hat = model(batch_X)
            batch_y_hat = torch.sigmoid(batch_y_hat) >= threshold
            batch_y_hat = batch_y_hat.detach().cpu().numpy().astype(int)
            y_hat_ood[i*bs:(i+1)*bs] = batch_y_hat

    thresholds = [0.5] * y_hat.shape[-1]
    report.make_report(y_hat, thresholds, y_hat_ood)

if __name__ == "__main__": 
    run_pipeline_classifier()