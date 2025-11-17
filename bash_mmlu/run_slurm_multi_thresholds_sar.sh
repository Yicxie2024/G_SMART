#!/usr/bin/env bash
#SBATCH --job-name=smart-multi
#SBATCH --nodelist=scratchy
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
END_OPTION=${2:-7}

# 验证参数
if ! [[ "$START_OPTION" =~ ^[0-9]+$ ]] || ! [[ "$END_OPTION" =~ ^[0-9]+$ ]]; then
  echo "ERROR: Both arguments must be numbers (0-7)" >&2
  echo "Usage: sbatch run_slurm_multi.sh [start_option] [end_option]" >&2
  echo "Example: sbatch run_slurm_multi.sh 0 3" >&2
  echo "Available options: 0-7 (clean scoring methods only)" >&2
  echo "  0-1: PRM baseline methods" >&2
  echo "  2-3: Confidence scoring" >&2
  echo "  4-5: MSP scoring" >&2
  echo "  6-7: Top-2 margin scoring" >&2
  exit 1
fi

if [[ $START_OPTION -gt $END_OPTION ]]; then
  echo "ERROR: Start option ($START_OPTION) cannot be greater than end option ($END_OPTION)" >&2
  exit 1
fi

if [[ $START_OPTION -lt 0 ]] || [[ $END_OPTION -gt 7 ]]; then
  echo "ERROR: Options must be between 0 and 7" >&2
  exit 1
fi

echo "=== Running options $START_OPTION to $END_OPTION sequentially ==="

# 循环运行指定范围的选项
for OPTION in $(seq $START_OPTION $END_OPTION); do
  echo "=== Running OPTION=$OPTION ==="
  MODEL_PATH=/storage/ukp/shared/shared_model_weights/models--Qwen2.5-14B-Instruct
  #DRAFT_MODEL_PATH=/storage/ukp/shared/shared_model_weights/models--Qwen2.5-7B-Instruct
  
  case "$OPTION" in
    0) CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search_smart_mmlu_pro_conf.yaml; EXTRA=(--n=1 --beam_width=16 --score_method=conf --dataset_split=test --uq_threshold=0.09 --dataset_start=0 --dataset_end=1 --run_random_baseline=false --run_llm_baseline=false --run_slm_baseline=false --model_path=$MODEL_PATH) ;;
    #1) CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search_smart_mmlu_pro_conf.yaml; EXTRA=(--n=1 --beam_width=16 --score_method=top2_margin --dataset_split=test --uq_threshold=0.78 --dataset_start=0 --dataset_end=500 --run_random_baseline=false --run_llm_baseline=false --run_slm_baseline=false --model_path=$MODEL_PATH) ;;
    #2) CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search_smart_mmlu_pro_conf.yaml; EXTRA=(--n=1 --beam_width=16 --score_method=token_entropy --dataset_split=test --uq_threshold=0.0036 --dataset_start=0 --dataset_end=500 --run_random_baseline=false --run_llm_baseline=false --run_slm_baseline=false --model_path=$MODEL_PATH) ;;
    #3) CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search_smart_mmlu_pro_conf.yaml; EXTRA=(--n=1 --beam_width=16 --score_method=token_sar --dataset_split=test --uq_threshold=0.05 --dataset_start=0 --dataset_end=500 --run_random_baseline=false --run_llm_baseline=false --run_slm_baseline=false --model_path=$MODEL_PATH) ;;
    *) echo "Unknown OPTION=$OPTION. Valid options are 0-7." >&2; exit 1 ;;
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
