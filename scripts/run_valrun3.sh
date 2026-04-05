#!/bin/bash
#SBATCH --job-name=valrun3
#SBATCH --partition=ciaq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=7-00:00:00
#SBATCH --output=./slurm_LOG/valrun3_out_%j.log
#SBATCH --error=./slurm_LOG/valrun3_err_%j.log

mkdir -p ./slurm_LOG

cd /isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_tempo
python val_train.py --config configs/valrun3_train.yaml
