#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT_PATH="${ROOT_PATH:-./dataset/ETT-small}"
PRED_LENS="${PRED_LENS:-96 192 336 720}"
SEED="${SEED:-2021}"
GPU="${GPU:-0}"
read -r -a HORIZONS <<< "$PRED_LENS"
mkdir -p logs/PatchTST/ETTh1

# The forecasting module uses 512/2048; only imputation uses 16/32.
# For DUET, we recommend --full_imputation to match forecasting width.
# This starter script runs PatchTST only; other backbones are available in run.py.
for pred_len in "${HORIZONS[@]}"; do
  if [[ ! "$pred_len" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid prediction length: $pred_len" >&2
    exit 1
  fi
  "$PYTHON_BIN" -u run.py \
    --task_name msar_forecasting \
    --model PatchTST \
    --data ETTh1 --root_path "$ROOT_PATH" --data_path ETTh1.csv \
    --features M --target OT --freq h \
    --seq_len 336 --label_len 0 --pred_len "$pred_len" \
    --interval_list 6,1 --seq_len_list 56,336 \
    --d_model 512 --d_ff 2048 --e_layers 2 --n_heads 8 \
    --patch_len 16 --stride 8 --dropout 0.2 \
    --slim_imputation --slim_d_model 16 --slim_d_ff 32 \
    --loss MAE --learning_rate 0.001 --lradj type1 \
    --train_epochs 10 --patience 3 --batch_size 32 \
    --num_workers 0 --seed "$SEED" --gpu "$GPU" \
    --checkpoints ./checkpoints --result_file ./results.txt \
    2>&1 | tee "logs/PatchTST/ETTh1/pred${pred_len}_seed${SEED}.log"
done
