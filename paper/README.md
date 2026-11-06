# Manuscript reproduction: v2 simulator + SpecTf

Regenerates the tables and figures of *Fractional Cover Estimation from Imaging
Spectroscopy with Machine Learning Informed Conditional Unmixing* (`ref/latex`)
with the **v2 specmix simulator** (`../src/specmix`), the **existing SpecTf
label-query checkpoint** (`spectf_decoder_v3/model_epoch800.pth`, not retrained),
and **MLP-NPV retrained** on v2. It ports the relevant parts of
`EMIT-FC/frac-cover` (Y. Marchetti), which depended on the old `cover_class`
simulator.

`tables/RESULTS.md` is a side-by-side of the manuscript numbers and the
regenerated ones.

## Setup

```bash
mamba activate spectf
pip install seaborn opencv-python-headless       # figures 7/8, scene RGB
```

E(MC)^2 runs `julia/unmix_pipeline_v2.jl` (Marchetti's script) on the
`add_spec_res` branch of her SpectralUnmixing.jl fork. The upstream emit-sds
release has a different API. Paths are set in `configs/paper.yaml`.

```bash
cd ~/Documents/EMIT/tools
curl -LO https://julialang-s3.julialang.org/bin/mac/aarch64/1.10/julia-1.10.5-macaarch64.tar.gz && tar xzf julia-1.10.5-macaarch64.tar.gz
git clone -b add_spec_res https://github.com/marchett2/SpectralUnmixing SpectralUnmixing-marchett
cd SpectralUnmixing-marchett
JULIA_DEPOT_PATH=../julia-depot ../julia-1.10.5/bin/julia --project=. \
  -e 'using Pkg; Pkg.instantiate(); Pkg.add(["ProgressMeter","JSON"])'
```

## Run

`./run_all.sh` runs everything that does not need an EMIT scene, in about
1.5 h on an M-series laptop. Each pipeline stage caches its results in
`outputs/`; delete a stage's outputs to recompute it.

| Stage | Produces |
|---|---|
| `pipeline/s01_make_data.py` | 10k held-out sim test set (val endmember split, seed 43), stratified forced-fraction set (5 classes x 20 bins x 1000), and the SMA library (v2-subsampled train-split endmembers: soil 80, pv 40, npv 40, snow+ice 50, water 50) |
| `pipeline/s02_spectf.py` | SpecTf probabilities and F1-optimal thresholds |
| `pipeline/s03_train_mlp.py` | MLP-NPV trained over 10 CV seeds; picks the seed whose sim mean F1 is closest to the median |
| `pipeline/s04_sma.py` | E(MC)^2 `nmc30_nend25_bvls`, global plus SpecTf- and MLP-conditional, on sim test and OOD |
| `pipeline/s05_scene.py <name>` | Scene crop: SpecTf, then global and conditional E(MC)^2 (`configs/scenes.yaml`) |
| `pipeline/s06_logit_thresholds.py` | Optional: F1 thresholds picked on logits instead of probabilities |

| Script | Manuscript item | Output (`figures/`) | Manuscript file (`ref/latex/Figures/`) | EMIT-FC source |
|---|---|---|---|---|
| `figure1.py` | Fig 1, FC-4 global unmixing (scene) | `figure1_fraction_cover_full_FC-4.png` | `fraction_cover_full_FC-4.png` | `scenes/plot_scene.py` |
| `figure2.py` | Fig 2, pipeline diagram (static) | `figure2_conditional_unmixing_pipeline_v4.pdf` | `conditional_unmixing_pipeline_v4.pdf` | `paper/make_pipeline_figure.py` |
| `figure3.py` | Fig 3, reduced library | `figure3_reduced_library.pdf` | `reduced_library.pdf` | `scripts/plot_reduced_library.py` |
| `figure4.py` | Fig 4, simulated spectra | `figure4_simulated_spectra.png` | `sim_spectra_seed6.png` | `scripts/plot_simulated_examples.py` |
| `figure5.py` | Fig 5, MLP architecture (static) | `figure5_mlp_npv_architecture.png` | `mlp_npv_architecture.png` | `paper/mlp_npv_diagram.py` |
| `figure5b.py` | New: SpecTf label-query architecture (not yet in the tex) | `figure5b_spectf_architecture.png` | | new (sizes from `configs/spectf_model_config.yaml` + checkpoint) |
| `figure6.py` | Fig 6, polygon FC-35 (scene, no model) | `figure6_polygon_FC-35.pdf` | `polygon_FC-35.pdf` | `paper/polygon_figure.py` |
| `figure7.py` | Fig 7, t-SNE | `figure7_tsne_class_colored.png` | `tsne_class_colored.png` | `scripts/tsne_comparison.py` |
| `figure8.py` | Fig 8a/b, soil/NPV t-SNE and k-NN coverage | `figure8a_tsne_soil_npv_real_only.png`, `figure8b_class_coverage_heatmap_ood.png` | `tsne_soil_npv_real_only.png`, `class_coverage_heatmap_ood.png` | `scripts/tsne_comparison.py` |
| `figure9.py` | Fig 9, TPR vs fraction | `figure9_tpr_by_fraction.pdf` | `tpr_by_fraction.pdf` | `scripts/plot_tpr_by_fraction.py` |
| `figure10.py` | Fig 10 and the Sec 4.3 numbers | `figure10_sma_ood_class_count.png` | `sma_ood_class_count_figure.png` | `scripts/compare_sma.py` (`plot_class_count_figure`) |
| `figure11.py` | Fig 11, FC-8 classes per pixel (scene) | `figure11_num_classes_comparison_fc-8.png` | `num_classes_comparison_fc-8.png` | `scenes/plot_scene.py` |
| `figure12.py` | Fig 12, FC-8 fraction maps (scene) | `figure12_fraction_cover_comparison.png` | `fraction_cover_comparison.png` | `scenes/plot_scene.py` |
| `scene_info.py <name>` | Fig 1/11 caption metadata (granule, date, center, RGB nm) | stdout | | `scenes/scene_info.py` |
| `table1.py` | Tables 1/2, libraries | `table1_libraries.{csv,tex}` | | |
| `table3.py` | Table 3, validation summary | `table3_validation_summary.{csv,tex}` | | `paper/summarize_polygons.py` |
| `table4.py` | Table 4, OOD TPR/FPR | `table4_presence.{csv,tex}` | | `scripts/figures.py` |
| `table5.py` | Table 5, sim MAE | `table5_mae.{csv,tex}` | | `scripts/figures.py` |
| `report.py` | Old vs new comparison | `tables/RESULTS.md` | | |

Every figure file starts with `figureN_` so it is easy to tell which manuscript
figure it is. Drop it in by renaming to the manuscript file name in the fourth
column. EMIT-FC sources are relative to `EMIT-FC/frac-cover/` (`paper/` is
`EMIT-FC/paper/`) and were last synced at EMIT-FC `db36a37` (2026-10-26).

`figure4.py` uses example-selection seed 6, as the manuscript did. Since the
v2 sim set is new, the examples it picks are different. `python figure4.py
--browse 0-9` writes quick looks to `outputs/figures/figure4_browse/` so you
can choose another seed.

### Scene figures (1, 6, 11, 12)

These need the EMIT L2A RFL granules (`pipeline/download_scenes.sh`, which
requires an Earthdata login in `~/.netrc`). After downloading:

```bash
python pipeline/s05_scene.py FC-4 && python figure1.py && python scene_info.py FC-4
python pipeline/s05_scene.py FC-8 && python figure11.py && python figure12.py && python scene_info.py FC-8
python figure6.py
```

A 600x600 crop is ~360k pixels of E(MC)^2. At ~27 ms/pixel/core on 8 jobs,
that is roughly 20-40 min per variant. The pipeline was validated end to end
on the local `test-scene` entry, a 40x40 crop of `data/test-scene`.

## Conventions (same as EMIT-FC unless noted)

- **OOD**: `validation_20260908_subsampled.h5` (94 polygons, 18,353 spectra)
  for every model. Uncertain labels (2) are excluded. The manuscript had used
  the 20260317 file for SpecTf and the class-count figure.
- **Thresholds**: per-class F1-optimal, fit on the split being evaluated (sim
  thresholds for sim test, OOD thresholds for OOD). This is in-sample on OOD,
  as before. The forced-fraction sweep (Fig 9) uses 0.5, as
  `plot_tpr_by_fraction.py` did. Scene runs use the OOD thresholds.
- **Table 4**: presence means E(MC)^2 fraction >= 0.05.
- **Table 5**: MAE over pixels where the class is truly present; mixed is
  0 < x < 1, pure is x = 1.
- **E(MC)^2**: `n_mc 30`, `num_endmembers 25`, BVLS, class-even, `max_combinations 10`,
  `refl_scale 0.3`. Conditional pixels with one class get 1.0; pixels with no
  predicted class fall back to global (`unmix_pipeline_v2.jl`). On scenes,
  as in `run_scene.py`, they are left blank.
- **One change to the Julia script**: `SpectralLibrary(..., Float64[])`, with no
  ignore regions. The fork's `get_good_bands_mask` otherwise marks *every*
  band bad when no band lies above 2440 nm, which turns the whole library into NaN.
- **Figs 7/8**: OOD spectra are colored by the argmax of the 0/1/2 label row,
  as in the original. An uncertain (2) label outranks a present (1) one,
  which affects 7,809 / 18,353 spectra. `--ood-class present` gives the
  corrected version. The t-SNE uses perplexity 40, KDE `bw_adjust` 1.5, and
  serif paper fonts, matching EMIT-FC `db36a37`. Fig 8b follows
  `class_pairwise_coverage`: the scaler is fit on all OOD spectra and each
  class is capped at 3,000. It reproduces the manuscript's 0.19 / 0.14 / ≤ 0.09.
- **Fig 3**: the library is the v2-subsampled train split (soil 70 + 10 rock,
  PV 40, NPV 40, snow/ice 50, water 50), not k-medoids-100. The caption's
  "k-medoids" and "100 endmembers per class + 30 rock" no longer apply.
