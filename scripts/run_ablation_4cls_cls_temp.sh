#!/bin/bash
#SBATCH --job-name=abl_4cls_t
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=./slurm_LOG/abl_4cls_temp_out_%j.log
#SBATCH --error=./slurm_LOG/abl_4cls_temp_err_%j.log

mkdir -p ./slurm_LOG
cd /isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_tempo
python val_train.py --config configs/rowsplit_4cls_temporal_v2_cls_only.yaml
