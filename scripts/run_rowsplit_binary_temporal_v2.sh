#!/bin/bash
#SBATCH --job-name=rs_bt_v2
#SBATCH --partition=ciaq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=./slurm_LOG/rs_bin_temp_v2_out_%j.log
#SBATCH --error=./slurm_LOG/rs_bin_temp_v2_err_%j.log

mkdir -p ./slurm_LOG

cd /isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_tempo
python val_train.py --config configs/rowsplit_binary_temporal_v2.yaml
