#!/bin/bash
#SBATCH --job-name=abl_bcls_sf
#SBATCH --partition=ciaq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=./slurm_LOG/abl_bcls_sf_out_%j.log
#SBATCH --error=./slurm_LOG/abl_bcls_sf_err_%j.log

mkdir -p ./slurm_LOG
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
python train_rowsplit.py --config configs/rowsplit_binary_cls_only.yaml


