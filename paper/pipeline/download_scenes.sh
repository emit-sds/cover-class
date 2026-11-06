#!/usr/bin/env bash
# Download the three EMIT L2A RFL v001 granules used by the scene figures.
# Requires an Earthdata Login entry in ~/.netrc:
#   machine urs.earthdata.nasa.gov login <user> password <pass>
# Granules found via CMR (short_name=EMITL2ARFL), matching the fids in the
# validation h5 (FC-4, FC-8, FC-35).
set -euo pipefail
cd "$(dirname "$0")/../outputs"
BASE=https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001
for pair in \
    FC-4:EMIT_L2A_RFL_001_20250124T125641_2502409_027 \
    FC-8:EMIT_L2A_RFL_001_20230904T172442_2324711_001 \
    FC-35:EMIT_L2A_RFL_001_20240701T235941_2418315_004; do
  name=${pair%%:*}; g=${pair#*:}
  mkdir -p downloads/$name
  [ -s downloads/$name/$g.nc ] && { echo "have $g"; continue; }
  curl -fL --netrc -b /tmp/edl_cookies -c /tmp/edl_cookies -o downloads/$name/$g.nc "$BASE/$g/$g.nc"
done
