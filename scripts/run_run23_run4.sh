#!/bin/bash
#SBATCH --job-name=run23_run4
#SBATCH --partition=ciaq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=7-00:00:00
#SBATCH --output=./slurm_LOG/run23_run4_out_%j.log
#SBATCH --error=./slurm_LOG/run23_run4_err_%j.log

mkdir -p ./slurm_LOG

cd /isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_tempo
python val_train.py --config configs/run23_train_run4_external.yaml
