#!/bin/bash
# Stage npy_cache to local NVMe on cia017.
# Designed to be sourced from sbatch scripts. Skips if already staged.

LOCAL=/local/zzhu_npy_cache
mkdir -p "$LOCAL"/{run22,run3,run4}

stage_one() {
  local SRC="$1" DST="$2" NAME="$3"
  local COUNT=$(ls "$DST"/*.npy 2>/dev/null | wc -l)
  if [ "$COUNT" -gt 400 ]; then
    echo "[stage] $NAME: already on local NVMe ($COUNT files)"
  else
    echo "[stage] $NAME: copying to local NVMe ..."
    rsync -a "$SRC/" "$DST/"
    echo "[stage] $NAME: done ($(ls "$DST"/*.npy | wc -l) files)"
  fi
}

stage_one "/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/DATA/GMU_cell_1023/HBMVEC/Validation_Run2_3-11-26/Enhanced contour/npy_cache" "$LOCAL/run22" "Run2.2"
stage_one "/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/DATA/GMU_cell_1023/HBMVEC/Validation_Run3_3-13-26/npy_cache" "$LOCAL/run3" "Run3"
stage_one "/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/DATA/GMU_cell_1023/HBMVEC/Validation_Run_4_3-15-26/Enhanced contour/npy_cache" "$LOCAL/run4" "Run4"

echo "[stage] All data ready on local NVMe."
