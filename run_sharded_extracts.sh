#!/usr/bin/env bash
set -euo pipefail

# Number of shards for each dataset (default: 3)
PARTS=${PARTS:-3}
BATCH_SIZE=${BATCH_SIZE:-32}
MODEL_TYPE=${MODEL_TYPE:-UNI}
FORCE=${FORCE:-0}

# Datasets to run in sequence
CONFIGS=(
  "configs/TCGA_LGG.yaml"
)
# GPUs to use in parallel per dataset (map part1->GPUS[0], part2->GPUS[1], ...)
GPUS=(0 1 2)

PREP_ARGS=("-p" "$PARTS")
if [[ "$FORCE" == "1" ]]; then
  PREP_ARGS+=("-f")
fi

run_one_dataset() {
  local cfg="$1"
  local base="${cfg%.yaml}"

  echo "Preparing shards for $cfg (PARTS=$PARTS)"
  python prepare_sharded_configs.py "${PREP_ARGS[@]}" "$cfg"

  local ngpus=${#GPUS[@]}
  local part=1

  # Launch parts in waves capped by number of GPUs
  while (( part <= PARTS )); do
    local pids=()
    for gpu_idx in "${!GPUS[@]}"; do
      (( part > PARTS )) && break
      local gpu="${GPUS[$gpu_idx]}"
      local part_cfg="${base}_part${part}.yaml"
      echo "[DATASET $(basename "$base"), GPU ${gpu}] Running ${part_cfg}"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        python 1_extract_pretrain_feats.py \
          --config "$part_cfg" \
          --batch_size "$BATCH_SIZE" \
          --model_type "$MODEL_TYPE"
      ) &
      pids+=($!)
      (( part++ ))
    done
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
  done

  echo "Dataset $(basename "$base") completed."
}

for cfg in "${CONFIGS[@]}"; do
  run_one_dataset "$cfg"
done

echo "All datasets completed."
