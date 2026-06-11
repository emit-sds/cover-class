# SpecTf v1

Jake Lee jake.h.lee@jpl.nasa.gov
2026-06-11

## Description

This is the v1 5-class multiclass multilabel classification model. This model is intended for use for baselining, research and development is ongoing.

This model was trained with:

```bash
python training_spectf.py \
    --outdir outdir/ \
    --data-config data_config.yaml \
    --model-config spectf_sweep1.yaml \
    --focal-alpha None \
    --focal-gamma 2.0
```

`data_config.yaml` is only included as a reference, its actual filepaths must be replaced with valid paths.

`spectf_sweep1.yaml` must be used as-is.

## Usage

`report_spectf.py` is a complete example of how to use this model for inference. The minimum amount of code required is as follows:

```python
# Get the spectra wavelengths, required for SpecTf
banddef = banddef_from_config(data_config)

# Load the OOD dataset
ood_test_set_x, ood_test_set_y = ood_test_set_from_config(data_config, include_unknown=False)
ood_dataset = TestDataset(ood_test_set_x, ood_test_set_y)
ood_dataloader = DataLoader(ood_dataset, batch_size=m_config['batch_size'], shuffle=False)

# Define device
device = get_device(0)

# Define model
model = SpecTfEncoder(banddef.to(dtype=torch.float32, device=device),
		 dim_output=m_config['model']['dim_output'],
		 num_heads=m_config['model']['num_heads'],
		 dim_proj=m_config['model']['dim_proj'],
		 dim_ff=m_config['model']['dim_ff'],
		 dropout=m_config['model']['dropout'],
		 agg=m_config['model']['agg'],
		 use_residual=m_config['model']['use_residual'],
		 num_layers=m_config['model']['num_layers']).to(device)

# Load model weights
model.load_state_dict(torch.load(model_weights, map_location=device))
model.eval()

# Run inference
y_hat_ood = np.zeros_like(ood_test_set_y, dtype=float)
for i, (batch_X, _) in enumerate(ood_dataloader):
batch_X = batch_X.to(device=device, dtype=torch.float32)
batch_X = torch.unsqueeze(batch_X, -1)
with torch.no_grad():
    logits = model(batch_X)
    batch_y_hat = torch.sigmoid(logits)
    batch_y_hat = batch_y_hat.detach().cpu().numpy().astype(float)
    batch_len = len(batch_y_hat)
    y_hat_ood[i*bs:i*bs+batch_len] = batch_y_hat

```



