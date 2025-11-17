#!/usr/bin/env bash
#SBATCH --job-name=hybrid-prm
#SBATCH --nodelist=moe
#SBATCH --partition=yolo
#SBATCH --qos=yolo
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

# 禁用 tokenizers 并行处理以避免与多进程冲突
export TOKENIZERS_PARALLELISM=false

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

# 解析命令行参数
START_OPTION=${1:-0}
END_OPTION=${2:-11}

# 验证参数
if ! [[ "$START_OPTION" =~ ^[0-9]+$ ]] || ! [[ "$END_OPTION" =~ ^[0-9]+$ ]]; then
  echo "ERROR: Both arguments must be numbers (0-11)" >&2
  echo "Usage: sbatch run_slurm_hybrid_prm.sh [start_option] [end_option]" >&2
  echo "Example: sbatch run_slurm_hybrid_prm.sh 0 3" >&2
  echo "Available options: 0-11 (hybrid UQ+PRM threshold combinations)" >&2
  echo "  0-2:  uq_low=0.2 with prm_th=0.6,0.7,0.8 (Conservative strategy)" >&2
  echo "  3-5:  uq_low=0.3 with prm_th=0.6,0.7,0.8 (Balanced strategy)" >&2
  echo "  6-8:  uq_low=0.4 with prm_th=0.6,0.7,0.8 (Efficient strategy)" >&2
  echo "  9-11: uq_low=0.35 with prm_th=0.65,0.75,0.85 (Fine-tuned)" >&2
  exit 1
fi

if [[ $START_OPTION -gt $END_OPTION ]]; then
  echo "ERROR: Start option ($START_OPTION) cannot be greater than end option ($END_OPTION)" >&2
  exit 1
fi

if [[ $START_OPTION -lt 0 ]] || [[ $END_OPTION -gt 11 ]]; then
  echo "ERROR: Options must be between 0 and 11" >&2
  exit 1
fi

echo "=== Running Hybrid SMART Correction (UQ + PRM) ==="
echo "=== Options $START_OPTION to $END_OPTION sequentially ==="

# 循环运行指定范围的选项
for OPTION in $(seq $START_OPTION $END_OPTION); do
  echo "=== Running OPTION=$OPTION ==="
  
  # 配置不同的 UQ 和 PRM 阈值组合
  # Format: uq_threshold, prm_threshold
  case "$OPTION" in
    0) CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search_smart_hybrid_prm.yaml; EXTRA=(--n=1 --beam_width=16 --score_method=hybrid_prm --uq_threshold=0.5 --prm_threshold=0.98 --dataset_start=51 --dataset_end=500) ;;
    1) CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search_smart_hybrid_prm.yaml; EXTRA=(--n=1 --beam_width=16 --score_method=hybrid_prm --uq_threshold=0.4 --prm_threshold=0.98 --dataset_start=51 --dataset_end=500) ;;
    2) CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search_smart_hybrid_prm.yaml; EXTRA=(--n=1 --beam_width=16 --score_method=hybrid_prm --uq_threshold=0.5 --prm_threshold=0.5 --dataset_start=51 --dataset_end=500) ;;
    *) echo "Unknown OPTION=$OPTION. Valid options are 0-11." >&2; exit 1 ;;
  esac

  echo "=== Using LOCAL cache at: $LOCAL ==="
  echo "=== Running: python scripts/test_time_compute.py $CONFIG ${EXTRA[*]} ==="
  
  # 运行当前选项
  srun python scripts/test_time_compute.py "$CONFIG" "${EXTRA[@]}"
  
  # 检查上一个命令的退出状态
  if [[ $? -ne 0 ]]; then
    echo "ERROR: Option $OPTION failed with exit code $?"
    echo "Stopping execution of remaining options."
    exit 1
  fi
  
  echo "=== Option $OPTION completed successfully ==="
  echo ""
done

echo "=== All options ($START_OPTION to $END_OPTION) completed successfully ==="
echo "=== Results saved to: /storage/ukp/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/outputs/smart/hybrid_prm/ ==="

