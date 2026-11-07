#!/usr/bin/env bash
set -eo pipefail

# Set MODEL to choose a backbone, e.g. MODEL=TimeMixer bash scripts/run_msar.sh.
MODEL="${MODEL:-PatchTST}"
SLIM_ARGS=(--slim_imputation --slim_d_model 16 --slim_d_ff 32)
if [[ "$MODEL" == "DUET" ]]; then
  # DUET imputation slimming is not recommended based on our width comparison.
  SLIM_ARGS=()
fi

python run.py \
  --model "$MODEL" \
  --data ETTh1 \
  --root_path ./dataset/ETT-small \
  --data_path ETTh1.csv \
  --seq_len 336 \
  --label_len 0 \
  --pred_len 96 \
  --loss MAE \
  --interval_list 6,1 \
  --seq_len_list 56,336 \
  "${SLIM_ARGS[@]}"
