#!/bin/bash
#SBATCH --job-name=tif2npy
#SBATCH --partition=ciaq
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=./slurm_LOG/tif2npy_out_%j.log
#SBATCH --error=./slurm_LOG/tif2npy_err_%j.log

mkdir -p ./slurm_LOG

cd /isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_tempo
python convert_tiff_to_npy.py --workers 16
