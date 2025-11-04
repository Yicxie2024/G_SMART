#!/usr/bin/env bash
#SBATCH --job-name=smart-random-baseline
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
# 如果只提供一个参数，默认只运行该选项；如果提供两个参数，运行范围
if [[ -z "${2:-}" ]]; then
  # 只提供了一个参数，只运行该选项
  START_OPTION=${1:-0}
  END_OPTION=${1:-0}
else
  # 提供了两个参数，运行范围
  START_OPTION=${1:-0}
  END_OPTION=${2:-3}
fi

# 验证参数
if ! [[ "$START_OPTION" =~ ^[0-9]+$ ]] || ! [[ "$END_OPTION" =~ ^[0-9]+$ ]]; then
  echo "ERROR: Both arguments must be numbers (0-3)" >&2
  echo "Usage: sbatch run_slurm_random_baseline.sh [option] OR sbatch run_slurm_random_baseline.sh [start_option] [end_option]" >&2
  echo "Examples:" >&2
  echo "  sbatch run_slurm_random_baseline.sh 0        # Run only option 0" >&2
  echo "  sbatch run_slurm_random_baseline.sh 0 2     # Run options 0, 1, 2" >&2
  echo "Available options: 0-3 (different random_thresholds configurations)" >&2
  echo "  0: random_thresholds [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]" >&2
  echo "  1: random_thresholds [0.6, 0.7, 0.8, 0.9, 1.0]" >&2
  echo "  2: random_thresholds [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]" >&2
  echo "  3: random_thresholds [0.1, 0.3, 0.5, 0.7, 0.9]" >&2
  exit 1
fi

if [[ $START_OPTION -gt $END_OPTION ]]; then
  echo "ERROR: Start option ($START_OPTION) cannot be greater than end option ($END_OPTION)" >&2
  exit 1
fi

if [[ $START_OPTION -lt 0 ]] || [[ $END_OPTION -gt 3 ]]; then
  echo "ERROR: Options must be between 0 and 3" >&2
  exit 1
fi

echo "=== Running options $START_OPTION to $END_OPTION sequentially ==="

# 基础参数配置（所有选项共用）
BASE_EXTRA=(
  --approach=beam_search
  --smart_search=True
  --score_method=conf
  --n=1
  --beam_width=16
  --num_iterations=40
  --temperature=0.8
  --top_p=1.0
  --max_tokens=2048
  --lookahead=0
  --seed=0
  --dataset_name=HuggingFaceH4/MATH-500
  --dataset_split=test
  --data_name=math
  --dataset_start=51
  --dataset_end=500
  --run_slm_baseline=False
  --run_llm_baseline=False
  --run_random_baseline=False
  --model_path=/storage/ukp/shared/shared_model_weights/models--Qwen2.5-7B-Instruct
  --draft_model_path=/storage/ukp/shared/shared_model_weights/models--Qwen2.5-1.5B-Instruct
  --gpu_memory_utilization=0.4
  --search_batch_size=1
  --agg_strategy=last
  --sort_completed=False
  --threshold=0.9
)

# 循环运行指定范围的选项
for OPTION in $(seq $START_OPTION $END_OPTION); do
  echo "=== Running OPTION=$OPTION ==="
  
  # 根据选项设置不同的 random_thresholds
  case "$OPTION" in
    0)
      # 低阈值范围
      RANDOM_THRESHOLDS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
      ;;
    1)
      # 高阈值范围
      RANDOM_THRESHOLDS=(0.6 0.7 0.8 0.9 1.0)
      ;;
    2)
      # 均匀分布
      RANDOM_THRESHOLDS=(0.0 0.2 0.4 0.6 0.8 1.0)
      ;;
    3)
      # 奇数阈值
      RANDOM_THRESHOLDS=(0.1 0.3 0.5 0.7 0.9)
      ;;
    *)
      echo "Unknown OPTION=$OPTION. Valid options are 0-3." >&2
      exit 1
      ;;
  esac
  
  echo "=== Random thresholds: ${RANDOM_THRESHOLDS[*]} ==="
  echo "=== Using LOCAL cache at: $LOCAL ==="
  echo "=== Running: python src/scripts/test_time_compute.py ${BASE_EXTRA[*]} --random_thresholds ${RANDOM_THRESHOLDS[*]} ==="
  
  # 运行当前选项
  # 注意：random_thresholds 需要使用空格分隔的多个参数值，而不是逗号分隔的字符串
  srun python scripts/test_time_compute.py "${BASE_EXTRA[@]}" --random_thresholds "${RANDOM_THRESHOLDS[@]}"
  
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

