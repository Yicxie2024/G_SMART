#!/usr/bin/env bash
#SBATCH --job-name=smart-prm-only
#SBATCH --nodelist=penelope
#SBATCH --partition=gpu
#SBATCH --qos=gpu-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/storage/ukp/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/SMART/logs/slurm-%x-%j.out
#SBATCH --error=/storage/ukp/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/SMART/logs/slurm-%x-%j.err
#SBATCH --chdir=/mnt/beegfs/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/SMART

set -euo pipefail

echo "=== Activating env ==="
source /storage/ukp/work/xie12/miniconda3/bin/activate smart_clean

# ---------- Paths & caches ----------
BASE=/mnt/beegfs/work/xie12
if [[ -n "${SLURM_TMPDIR:-}" && -w "$SLURM_TMPDIR" ]]; then
  LOCAL="$SLURM_TMPDIR"
else
  LOCAL="$BASE/tmp"
fi

export XDG_CACHE_HOME="$BASE/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

export VLLM_TORCH_COMPILE_CACHE_DIR="$LOCAL/vllm/torch_compile_cache/${SLURM_JOB_ID:-local}"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL/torch/inductor/${SLURM_JOB_ID:-local}"
export TRITON_CACHE_DIR="$LOCAL/triton/${SLURM_JOB_ID:-local}"
export CUDA_CACHE_PATH="$LOCAL/nv/${SLURM_JOB_ID:-local}"
export MPLCONFIGDIR="$BASE/tmp/matplotlib"
unset LUH_DISABLE_TOKEN_PROBABILITIES
export VLLM_NO_USAGE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$HF_DATASETS_CACHE" \
         "$VLLM_TORCH_COMPILE_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
         "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export SEED=0

# Allow caller to specify which slice of experiments to run
START_OPTION=${1:-0}
END_OPTION=${2:-0}

if ! [[ "$START_OPTION" =~ ^[0-9]+$ ]] || ! [[ "$END_OPTION" =~ ^[0-9]+$ ]]; then
  echo "ERROR: Both arguments must be non-negative integers." >&2
  exit 1
fi
if [[ $START_OPTION -gt $END_OPTION ]]; then
  echo "ERROR: Start option ($START_OPTION) cannot be greater than end option ($END_OPTION)" >&2
  exit 1
fi

echo "=== Running options $START_OPTION to $END_OPTION sequentially ==="

# Shared arguments for all PRM-only SMART runs
BASE_ARGS=(
  --approach beam_search
  --smart_search true
  --score_method prm
  --use_prm_only true
  --model_path /storage/ukp/shared/shared_model_weights/models--Qwen2.5-7B-Instruct
  --draft_model_path /storage/ukp/shared/shared_model_weights/models--Qwen2.5-1.5B-Instruct
  --filter_duplicates true
  --dataset_name HuggingFaceH4/MATH-500
  --search_batch_size 1
  --n 1
  --seed 0
  --beam_width 1
  --num_iterations 40
  --run_random_baseline false
  --run_llm_baseline false
  --run_slm_baseline false
)

# Option-specific overrides. Add/edit cases as needed.
for OPTION in $(seq $START_OPTION $END_OPTION); do
  echo "=== Running OPTION=$OPTION ==="

  case "$OPTION" in
    0)
      EXTRA_ARGS=(
        --threshold 0.9
        --dataset_start 0
        --dataset_end 500
      )
      ;;
    1)
      EXTRA_ARGS=(
        --threshold 0.6
        --dataset_start 0
        --dataset_end 500
      )
      ;;
    2)
      EXTRA_ARGS=(
        --threshold 0.8
        --dataset_start 0
        --dataset_end 500
      )
      ;;
      3)
      EXTRA_ARGS=(
        --threshold 0.3
        --dataset_start 0
        --dataset_end 500
      )
      ;;
    *)
      echo "ERROR: Unknown OPTION=$OPTION. Extend the case statement for additional settings." >&2
      exit 1
      ;;
  esac

  echo "=== Using LOCAL cache at: $LOCAL ==="
  echo "=== Running: python scripts/test_time_compute.py ${BASE_ARGS[*]} ${EXTRA_ARGS[*]} ==="

  srun python scripts/test_time_compute.py "${BASE_ARGS[@]}" "${EXTRA_ARGS[@]}"
  status=$?
  if [[ $status -ne 0 ]]; then
    echo "ERROR: Option $OPTION failed with exit code $status"
    exit $status
  fi

  echo "=== Option $OPTION completed successfully ==="
done

echo "=== All requested options completed successfully ==="

