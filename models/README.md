
## Example Usage
```
$ python3 models/training_example.py --outdir . --config /path/to/dataloader.yml --lr 0.0001 --batch-size 1024 --max-training-steps 100 --log-interval 20 --simulated-test-set-size 1000 --seed 42
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 8473.34it/s]
Step      20 | BCE loss: 0.6535 nats | Running average mean: 0.6474
Step      40 | BCE loss: 0.6188 nats | Running average mean: 0.6261
Step      60 | BCE loss: 0.5232 nats | Running average mean: 0.6097
Step      80 | BCE loss: 0.5816 nats | Running average mean: 0.5953
src/cover_class/reporting/metrics.py:114: RuntimeWarning: All-NaN slice encountered
  vmax = max(1.0, np.nanmax(conf))
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-0.031505488..1.2835808].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-9999.0..1.729243].
```

### Help Message
```
$ python3 models/training_example.py --help

 Usage: training_example.py [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --outdir                   DIRECTORY  Output file directory. [required]                                                                                                       │
│ *  --config                   FILE       Path to the YAML config for the dataloader. [required]                                                                                  │
│ *  --lr                       FLOAT      Learning rate. [required]                                                                                                               │
│ *  --batch-size               INTEGER    Batch size. [required]                                                                                                                  │
│    --max-training-steps       INTEGER    Maximum number of training steps to do. This is needed for simulated data as otherwise there would be no loop termination condition.    │
│    --log-interval             OPTIONAL   Number of steps between logging                                                                                                         │
│    --static-config            OPTIONAL   Path to the YAML config for generating an HDF5 from a set of CSVs. If provided, it will run a generator function for creating the HDF5s, │
│                                          but if omitted, it will not.                                                                                                            │
│    --simulated-test-set-size  INTEGER    Number of rows to generate for the simulated test set.                                                                                  │
│    --seed                     INTEGER    Seed.                                                                                                                                   │
│    --help                                Show this message and exit.                                                                                                             │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## SpecTf

`models/spectf/training_spectf.py` is a SpecTf implementation of the training script, and `models/spectf/report_spectf.py` is an example intended to demonstrate inference.

These scripts rely on the SpecTf package. Installation instructions can be found at https://github.com/emit-sds/SpecTf/blob/dev/spectf_cloud/README.md#-installation . Specifically, `make dev-install` should be called from the SpecTf repository.

`models/spectf/scene_spectf.py` can be used to run a trained model on an entire scene and to produce an n-channel GeoTIFF of classwise probabilities.
