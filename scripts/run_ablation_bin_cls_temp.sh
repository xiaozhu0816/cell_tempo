#!/bin/bash
#SBATCH --job-name=abl_bcls_t
#SBATCH --partition=ciaq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=./slurm_LOG/abl_bcls_temp_out_%j.log
#SBATCH --error=./slurm_LOG/abl_bcls_temp_err_%j.log

mkdir -p ./slurm_LOG
cd /isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_tempo
python val_train.py --config configs/rowsplit_binary_temporal_v2_cls_only.yaml
