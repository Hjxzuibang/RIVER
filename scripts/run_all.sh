#!/bin/bash
# Reproduce all RIVER results from the paper.
# Usage: bash scripts/run_all.sh [--device cuda] [--seeds "42"]
#
# Prerequisites:
#   1. Preprocessed data in processed_data/{Dataset}/{CellType}.h5ad
#   2. Prior graphs in processed_data/{Dataset}/prior_graph_{CellType}.pt
#   See README.md for data preparation instructions.

set -e

DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-42}"

echo "============================================================"
echo "RIVER: Reproducing Paper Results"
echo "Device: $DEVICE | Seeds: $SEEDS"
echo "============================================================"

CONFIGS=(
    "configs/replogle_k562.yaml"
    "configs/nadig_hepg2.yaml"
    "configs/nadig_jurkat.yaml"
    "configs/replogle_rpe1.yaml"
    "configs/norman.yaml"
)

for config in "${CONFIGS[@]}"; do
    dataset_name=$(basename "$config" .yaml)
    
    for seed in $SEEDS; do
        echo ""
        echo "============================================================"
        echo "Training: $dataset_name (seed=$seed)"
        echo "============================================================"
        
        python train.py \
            --config "$config" \
            --seed "$seed" \
            --device "$DEVICE"
        
        # Determine output dir from config
        exp_name=$(grep "exp_name:" "$config" | awk '{print $2}')
        model_dir="results/${exp_name}_seed${seed}/checkpoints"
        data_path=$(grep "processed_dir:" "$config" | awk '{print $2}')
        dataset=$(grep "  dataset:" "$config" | awk '{print $2}')
        celltype=$(grep "  celltype:" "$config" | awk '{print $2}')
        h5ad_path="${data_path}/${dataset}/${celltype}.h5ad"
        
        # Standard evaluation (no fine-tuning)
        echo ""
        echo "[Eval] Standard: $dataset_name (seed=$seed)"
        python evaluate.py \
            --model_dir "$model_dir" \
            --data "$h5ad_path" \
            --output_dir "results/${exp_name}_seed${seed}/eval" \
            --split test \
            --device "$DEVICE" \
            --method RIVER

        # Per-perturbation fine-tuning evaluation
        echo ""
        echo "[Eval] Fine-tuning: $dataset_name (seed=$seed)"
        python evaluate.py \
            --model_dir "$model_dir" \
            --data "$h5ad_path" \
            --output_dir "results/${exp_name}_seed${seed}/eval_ft" \
            --split test \
            --device "$DEVICE" \
            --method RIVER \
            --finetune \
            --ft_steps 200 \
            --ft_lr 1e-3 \
            --ft_grad_clip 1.0

    done
done

echo ""
echo "============================================================"
echo "All experiments complete. Results in results/"
echo "============================================================"
