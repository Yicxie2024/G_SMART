#!/usr/bin/env bash
#SBATCH --job-name=smart
#SBATCH --nodelist=scratchy
#SBATCH --partition=gpu
#SBATCH --qos=gpu-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=/storage/ukp/work/xie12/uncertainty-guided-reasoning/logs/slurm-%x-%j.out
#SBATCH --error=/storage/ukp/work/xie12/uncertainty-guided-reasoning/logs/slurm-%x-%j.err
#SBATCH --chdir=/mnt/beegfs/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/SMART

set -euo pipefail

echo "=== Activating env ==="
source /storage/ukp/work/xie12/miniconda3/bin/activate smart2

# ---------- Paths & caches ----------
BASE=/mnt/beegfs/work/xie12
# 优先使用本地临时盘（更快、避免网络盘权限/并发问题）
if [[ -n "${SLURM_TMPDIR:-}" && -w "$SLURM_TMPDIR" ]]; then
  LOCAL="$SLURM_TMPDIR"
else
  LOCAL="$BASE/tmp"   # 回退到 beegfs
fi

# 统一缓存位置
export XDG_CACHE_HOME="$BASE/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
# 向后兼容（可保留/之后移除）
export TRANSFORMERS_CACHE="$HF_HOME/hub"

# 编译/内核缓存（放本地盘优先）
export VLLM_TORCH_COMPILE_CACHE_DIR="$LOCAL/vllm/torch_compile_cache/${SLURM_JOB_ID:-local}"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL/torch/inductor/${SLURM_JOB_ID:-local}"
export TRITON_CACHE_DIR="$LOCAL/triton/${SLURM_JOB_ID:-local}"
export CUDA_CACHE_PATH="$LOCAL/nv/${SLURM_JOB_ID:-local}"
export MPLCONFIGDIR="$BASE/tmp/matplotlib"

# 关闭 vLLM usage（避免去 ~ 写）
export VLLM_NO_USAGE=1

# Load API tokens from environment variables or secret files (optional)
# Priority: environment variables > secret files > continue without warning
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

# 确保目录存在
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$HF_DATASETS_CACHE" \
         "$VLLM_TORCH_COMPILE_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
         "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR"

# Python 包路径（你已在 SMART/ 目录下）
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

export SEED=0
OPTION=${1:-0}

# Clean the option parameter (remove any non-numeric characters)
OPTION=$(echo "$OPTION" | sed 's/[^0-9]//g')
if [[ -z "$OPTION" ]]; then
  OPTION=0
fi

echo "=== Using OPTION=$OPTION ==="

case "$OPTION" in
  0) CONFIG=recipes/Qwen2.5-7B-Instruct/best_of_n.yaml;         EXTRA=(--n=16 --beam_width=1 --score_method=prm) ;;
  1) CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search.yaml;       EXTRA=(--n=16 --beam_width=4 --score_method=prm) ;;
  2) CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search.yaml;       EXTRA=(--n=16 --beam_width=4 --score_method=conf) ;;
  3) CONFIG=recipes/Qwen2.5-1.5B-Instruct/best_of_n.yaml;       EXTRA=(--n=16 --beam_width=1 --score_method=prm) ;;
  4) CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search_smart.yaml; EXTRA=(--n=16 --beam_width=1 --score_method=prm) ;;
  5) CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search_smart.yaml; EXTRA=(--n=16 --beam_width=4 --score_method=prm) ;;
  *) echo "Unknown OPTION=$OPTION. Valid options are 0-5." >&2; exit 1 ;;
esac

echo "=== Using LOCAL cache at: $LOCAL ==="
srun python scripts/test_time_compute.py "$CONFIG" "${EXTRA[@]}"
