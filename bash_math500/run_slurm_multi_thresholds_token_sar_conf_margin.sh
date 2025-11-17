#!/usr/bin/env bash
#SBATCH --job-name=smart-csar
#SBATCH --nodelist=moe
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

export VLLM_NO_USAGE=1
export TOKENIZERS_PARALLELISM=false

if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  if [[ -f "$BASE/.secrets/hf_token" ]]; then
    export HUGGINGFACE_HUB_TOKEN="$(<"$BASE/.secrets/hf_token")"
    echo "Loaded HUGGINGFACE_HUB_TOKEN from secret file"
  else
    echo "Info: HUGGINGFACE_HUB_TOKEN not set. Some features may be limited."
  fi
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  if [[ -f "$BASE/.secrets/wandb_key" ]]; then
    export WANDB_API_KEY="$(<"$BASE/.secrets/wandb_key")"
    echo "Loaded WANDB_API_KEY from secret file"
  else
    echo "Info: WANDB_API_KEY not set. Logging may be limited."
  fi
fi

mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$HF_DATASETS_CACHE" \
         "$VLLM_TORCH_COMPILE_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
         "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export SEED=0

START_OPTION=${1:-0}
END_OPTION=${2:-5}

if ! [[ "$START_OPTION" =~ ^[0-9]+$ ]] || ! [[ "$END_OPTION" =~ ^[0-9]+$ ]]; then
  echo "ERROR: Both arguments must be numbers (0-5)" >&2
  echo "Usage: sbatch run_slurm_multi_thresholds_token_sar_conf_margin.sh [start] [end]" >&2
  exit 1
fi

if [[ $START_OPTION -gt $END_OPTION ]]; then
  echo "ERROR: start option ($START_OPTION) cannot be greater than end option ($END_OPTION)" >&2
  exit 1
fi

if [[ $START_OPTION -lt 0 ]] || [[ $END_OPTION -gt 5 ]]; then
  echo "ERROR: options must be between 0 and 5" >&2
  exit 1
fi

echo "=== Running options $START_OPTION to $END_OPTION sequentially ==="

for OPTION in $(seq $START_OPTION $END_OPTION); do
  echo "=== Running OPTION=$OPTION ==="

  case "$OPTION" in
    0) CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search_smart_conf.yaml; EXTRA=(--n=1 --beam_width=16 --score_method=token_sar_conf_margin --uq_threshold=100.0 --dataset_start=0 --dataset_end=2 --run_random_baseline=false --run_llm_baseline=false --run_slm_baseline=false) ;;
    1) CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search_smart_conf.yaml; EXTRA=(--n=1 --beam_width=16 --score_method=token_sar_conf_margin --uq_threshold=0.00 --dataset_start=51 --dataset_end=350 --run_random_baseline=false --run_llm_baseline=false --run_slm_baseline=false) ;;
    2) CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search_smart_conf.yaml; EXTRA=(--n=1 --beam_width=16 --score_method=token_sar_conf_margin --uq_threshold=0.50 --dataset_start=51 --dataset_end=350 --run_random_baseline=false --run_llm_baseline=false --run_slm_baseline=false) ;;
    3) CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search_smart_conf.yaml; EXTRA=(--n=1 --beam_width=16 --score_method=token_sar_conf_margin --uq_threshold=1.00 --dataset_start=51 --dataset_end=350 --run_random_baseline=false --run_llm_baseline=false --run_slm_baseline=false) ;;
    4) CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search_smart_conf.yaml; EXTRA=(--n=1 --beam_width=16 --score_method=token_sar_conf_margin --uq_threshold=1.50 --dataset_start=51 --dataset_end=350 --run_random_baseline=false --run_llm_baseline=false --run_slm_baseline=false) ;;
    5) CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search_smart_conf.yaml; EXTRA=(--n=1 --beam_width=16 --score_method=token_sar_conf_margin --uq_threshold=2.00 --dataset_start=51 --dataset_end=350 --run_random_baseline=false --run_llm_baseline=false --run_slm_baseline=false) ;;
    *) echo "Unknown OPTION=$OPTION. Valid options are 0-5." >&2; exit 1 ;;
  esac

  echo "=== Using LOCAL cache at: $LOCAL ==="
  echo "=== Running: python scripts/test_time_compute.py $CONFIG ${EXTRA[*]} ==="

  srun python scripts/test_time_compute.py "$CONFIG" "${EXTRA[@]}"

  if [[ $? -ne 0 ]]; then
    echo "ERROR: Option $OPTION failed with exit code $?"
    echo "Stopping execution of remaining options."
    exit 1
  fi

  echo "=== Option $OPTION completed successfully ==="
  echo ""
done

echo "=== All options ($START_OPTION to $END_OPTION) completed successfully ==="

