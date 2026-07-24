#!/bin/bash
# Canonical MLX LoRA training invocation. Per TOOLKIT_SPEC.md §8.2.
#
# Usage: ./train.sh <base_model> <data_dir> <adapter_out_dir> [batch] [layers] [iters] [lr] [seq_len] [grad_checkpoint]
#
# Starting hyperparameters (tune from these, log actuals in eval/):
#   Generative distillation (Copilot, 3B):   batch=2  layers=16  iters=800-1200  lr=1e-4   seq=1536
#   Classifiers + grader (1.5B):              batch=4  layers=12  iters=600-1000  lr=1.5e-4 seq=768
#
# grad_checkpoint (9th arg, default "0"): pass "1" to add --grad-checkpoint. Real bug
# found training Race Day Copilot (2026-07-22): its worst-case example is 4,197 real
# tokens (prompt + full ~43-row plan), well past the 1536 the spec assumed -- at
# max-seq-length=4608 without checkpointing, backprop activation memory exceeded this
# Mac's 16GB unified memory (mlx_lm reported "Peak mem" 46-49GB), and rather than a
# clean OOM crash, training silently produced Train loss 0.000 / Trained Tokens 0 every
# report -- degenerate, not-actually-training numbers that looked like a live process.
# --grad-checkpoint (recompute activations in the backward pass instead of storing
# them) dropped peak mem to 12.57GB and produced real loss/token numbers immediately.
# Not needed for the other 3 (much shorter seq_len) projects -- default stays off.
set -euo pipefail

BASE_MODEL="${1:?base model id required, e.g. mlx-community/Qwen2.5-3B-Instruct-4bit}"
DATA_DIR="${2:?data dir required (must contain train.jsonl, valid.jsonl)}"
ADAPTER_DIR="${3:?adapter output dir required}"
BATCH="${4:-2}"
LAYERS="${5:-16}"
ITERS="${6:-800}"
LR="${7:-1e-4}"
SEQ_LEN="${8:-1536}"
GRAD_CHECKPOINT="${9:-0}"

mkdir -p "$ADAPTER_DIR"

echo "Training: model=$BASE_MODEL data=$DATA_DIR -> $ADAPTER_DIR"
echo "batch=$BATCH layers=$LAYERS iters=$ITERS lr=$LR seq_len=$SEQ_LEN grad_checkpoint=$GRAD_CHECKPOINT"

EXTRA_ARGS=()
if [ "$GRAD_CHECKPOINT" = "1" ]; then
  EXTRA_ARGS+=(--grad-checkpoint)
fi

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
  "${EXTRA_ARGS[@]}" \
  | tee "$ADAPTER_DIR/train.log"

# Extract loss curve from the log (mlx_lm prints "Iter N: Train loss X, Val loss Y" lines)
grep -E "Iter [0-9]+: (Train|Val) loss" "$ADAPTER_DIR/train.log" > "$ADAPTER_DIR/loss_lines.txt" || true
echo "Done. Loss lines saved to $ADAPTER_DIR/loss_lines.txt -- plot with lora/plot_loss.py"
