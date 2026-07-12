#!/bin/bash
# Canonical MLX LoRA training invocation. Per TOOLKIT_SPEC.md §8.2.
#
# Usage: ./train.sh <base_model> <data_dir> <adapter_out_dir> [batch] [layers] [iters] [lr] [seq_len]
#
# Starting hyperparameters (tune from these, log actuals in eval/):
#   Generative distillation (Copilot, 3B):   batch=2  layers=16  iters=800-1200  lr=1e-4   seq=1536
#   Classifiers + grader (1.5B):              batch=4  layers=12  iters=600-1000  lr=1.5e-4 seq=768
set -euo pipefail

BASE_MODEL="${1:?base model id required, e.g. mlx-community/Qwen2.5-3B-Instruct-4bit}"
DATA_DIR="${2:?data dir required (must contain train.jsonl, valid.jsonl)}"
ADAPTER_DIR="${3:?adapter output dir required}"
BATCH="${4:-2}"
LAYERS="${5:-16}"
ITERS="${6:-800}"
LR="${7:-1e-4}"
SEQ_LEN="${8:-1536}"

mkdir -p "$ADAPTER_DIR"

echo "Training: model=$BASE_MODEL data=$DATA_DIR -> $ADAPTER_DIR"
echo "batch=$BATCH layers=$LAYERS iters=$ITERS lr=$LR seq_len=$SEQ_LEN"

mlx_lm.lora \
  --model "$BASE_MODEL" \
  --train \
  --data "$DATA_DIR" \
  --batch-size "$BATCH" \
  --num-layers "$LAYERS" \
  --iters "$ITERS" \
  --learning-rate "$LR" \
  --max-seq-length "$SEQ_LEN" \
  --val-batches 25 \
  --steps-per-eval 100 \
  --steps-per-report 20 \
  --save-every 200 \
  --adapter-path "$ADAPTER_DIR" \
  | tee "$ADAPTER_DIR/train.log"

# Extract loss curve from the log (mlx_lm prints "Iter N: Train loss X, Val loss Y" lines)
grep -E "Iter [0-9]+: (Train|Val) loss" "$ADAPTER_DIR/train.log" > "$ADAPTER_DIR/loss_lines.txt" || true
echo "Done. Loss lines saved to $ADAPTER_DIR/loss_lines.txt -- plot with lora/plot_loss.py"
