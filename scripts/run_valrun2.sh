#!/bin/bash
#SBATCH --job-name=valrun2
#SBATCH --partition=ciaq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=7-00:00:00
#SBATCH --output=./slurm_LOG/valrun2_out_%j.log
#SBATCH --error=./slurm_LOG/valrun2_err_%j.log

mkdir -p ./slurm_LOG

cd /isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_tempo
python val_train.py --config configs/valrun2_train.yaml
