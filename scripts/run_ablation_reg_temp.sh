#!/bin/bash
#SBATCH --job-name=abl_reg_t
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=./slurm_LOG/abl_reg_temp_out_%j.log
#SBATCH --error=./slurm_LOG/abl_reg_temp_err_%j.log

mkdir -p ./slurm_LOG
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
python train_rowsplit.py --config configs/rowsplit_binary_temporal_reg_only.yaml


