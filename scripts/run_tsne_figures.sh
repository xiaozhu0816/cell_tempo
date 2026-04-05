#!/bin/bash
#SBATCH --job-name=tsne_figs
#SBATCH --partition=ciaq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/tsne_figs_%j.log

cd /isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_tempo
mkdir -p logs

echo "=== Starting t-SNE figure generation ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
date

# 1. Fast t-SNE from prediction logits (no GPU needed, ~5 min)
echo ""
echo "=== Step 1: Fast t-SNE from predictions ==="
python3 -u analysis/gen_tsne_from_predictions.py
echo "Step 1 done at $(date)"

# 2. Full backbone feature extraction + t-SNE (needs GPU, ~15 min)
echo ""
echo "=== Step 2: Backbone feature extraction + t-SNE ==="
python3 -u analysis/extract_features_tsne.py
echo "Step 2 done at $(date)"

echo ""
echo "=== ALL DONE ==="
ls -la analysis/fig_tsne_*.png 2>/dev/null
date
