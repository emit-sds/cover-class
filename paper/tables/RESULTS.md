# v2 simulator + SpecTf (spectf_decoder_v3) results vs. current manuscript

Format: `manuscript → **regenerated**`. OOD = validation_20260908_subsampled (94 polygons, 18,353 spectra) for all models.

## Table 4 — OOD presence (E(MC)² fraction ≥ 0.05)

| Class | Global TPR | Global FPR | SpecTF TPR | SpecTF FPR | MLP-NPV TPR | MLP-NPV FPR |
|---|---|---|---|---|---|---|
| NPV | 0.64 → **0.66** | 0.39 → **0.32** | 0.68 → **0.76** | 0.08 → **0.03** | 0.70 → **0.77** | 0.30 → **0.06** |
| PV | 0.80 → **0.83** | 0.16 → **0.22** | 0.82 → **0.81** | 0.06 → **0.01** | 0.79 → **0.81** | 0.01 → **0.01** |
| Snow/ice | 1.00 → **1.00** | 0.45 → **0.45** | 0.96 → **1.00** | 0.05 → **0.01** | 0.97 → **1.00** | 0.00 → **0.00** |
| Soil | 0.99 → **1.00** | 0.14 → **0.08** | 0.90 → **0.96** | 0.05 → **0.01** | 0.93 → **0.93** | 0.00 → **0.00** |
| Water | 1.00 → **1.00** | 0.67 → **0.72** | 0.99 → **0.97** | 0.12 → **0.02** | 1.00 → **1.00** | 0.02 → **0.00** |

Raw classifier presence (before unmixing), for reference:

| Class | SpecTF TPR | SpecTF FPR | MLP-NPV TPR | MLP-NPV FPR |
|---|---|---|---|---|
| NPV | 0.96 | 0.02 | 0.93 | 0.06 |
| PV | 0.96 | 0.01 | 0.92 | 0.03 |
| Snow/ice | 0.98 | 0.00 | 1.00 | 0.00 |
| Soil | 0.96 | 0.00 | 0.93 | 0.00 |
| Water | 0.95 | 0.02 | 1.00 | 0.00 |

## Table 5 — sim-test MAE

| Class | Mixed Global | Mixed SpecTF | Mixed MLP | Pure Global | Pure SpecTF | Pure MLP |
|---|---|---|---|---|---|---|
| Soil | 0.23 → **0.17** | 0.17 → **0.17** | 0.20 → **0.18** | 0.45 → **0.51** | 0.07 → **0.11** | 0.07 → **0.05** |
| PV | 0.18 → **0.19** | 0.12 → **0.13** | 0.13 → **0.13** | 0.19 → **0.24** | 0.00 → **0.01** | 0.01 → **0.01** |
| NPV | 0.18 → **0.19** | 0.16 → **0.15** | 0.17 → **0.16** | 0.45 → **0.41** | 0.00 → **0.01** | 0.04 → **0.02** |
| Snow/ice | 0.18 → **0.15** | 0.15 → **0.14** | 0.18 → **0.17** | 0.51 → **0.41** | 0.02 → **0.04** | 0.11 → **0.07** |
| Water | 0.22 → **0.17** | 0.25 → **0.21** | 0.25 → **0.29** | 0.32 → **0.28** | 0.00 → **0.00** | 0.02 → **0.00** |

## Section 4.3 / Figure 10 — active classes per OOD pixel (SpecTF conditional)

| Quantity | Global | Conditional |
|---|---|---|
| frac_1 | 0.12 → **0.10** | 0.53 → **0.67** |
| frac_le2 | **0.46** | 0.90 → **0.91** |
| frac_ge3 | 0.57 → **0.54** | **0.09** |
| frac_ge4 | 0.24 → **0.28** | 0.03 → **0.00** |
| mean@0.05 | 2.70 → **2.73** | 1.60 → **1.42** |
| mean@0.1 | 2.20 → **2.27** | 1.50 → **1.35** |
| mean@0.2 | 1.80 → **1.76** | 1.35 → **1.25** |

## Figure 8b — OOD k-NN coverage (k=10)

| Quantity | Value |
|---|---|
| soil→npv | 0.19 → **0.19** |
| npv→soil | 0.14 → **0.14** |
| max other pair | 0.09 → **0.09** |

## MLP-NPV CV seeds (v2 simulator)

Chosen seed **1759** (sim mean F1 closest to CV median 0.9368).

Mean F1 across 10 seeds: sim 0.937 ± 0.003, OOD 0.952 ± 0.015.

| Seed | Sim mean F1 | OOD mean F1 |
|---|---|---|
| 154 | 0.9317 | 0.9246 |
| 9086 | 0.9352 | 0.9439 |
| 2202 | 0.9371 | 0.9441 |
| 3335 | 0.9424 | 0.9647 |
| 6818 | 0.9397 | 0.9314 |
| 5925 | 0.9352 | 0.9509 |
| 2550 | 0.9381 | 0.9650 |
| 1759 (chosen) | 0.9364 | 0.9604 |
| 1843 | 0.9319 | 0.9755 |
| 537 | 0.9392 | 0.9575 |

## OOD presence thresholds (F1-optimal on OOD, as in EMIT-FC)

| Class | SpecTF | MLP-NPV |
|---|---|---|
| Soil | 1.000 | 0.255 |
| PV | 0.987 | 0.036 |
| NPV | 0.087 | 0.151 |
| Snow/ice | 0.993 | 0.566 |
| Water | 0.201 | 0.437 |
