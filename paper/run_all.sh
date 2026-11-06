#!/usr/bin/env bash
# Regenerate every manuscript table/figure that does not need an EMIT scene.
# Stages cache their outputs under outputs/; delete a stage's outputs to redo it.
set -euo pipefail
cd "$(dirname "$0")"

python pipeline/s01_make_data.py        # sim test, forced-fraction set, SMA library   (~20 s)
python pipeline/s02_spectf.py           # SpecTf probabilities + thresholds             (~3 min)
python pipeline/s03_train_mlp.py        # MLP-NPV x 10 CV seeds, pick final             (~40 min)
python pipeline/s04_sma.py              # E(MC)^2 global / SpecTf / MLP, sim + OOD      (~30 min)

python table1.py; python table3.py; python table4.py; python table5.py
for f in 2 3 4 5 5b 7 8 9 10; do python figure$f.py; done
python report.py

# Scene figures (need Earthdata login in ~/.netrc; ~1-2 h of E(MC)^2 each):
#   pipeline/download_scenes.sh
#   python pipeline/s05_scene.py FC-4 && python figure1.py && python scene_info.py FC-4
#   python pipeline/s05_scene.py FC-8 && python figure11.py && python figure12.py && python scene_info.py FC-8
#   python figure6.py
