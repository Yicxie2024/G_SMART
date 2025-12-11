#!/usr/bin/env bash
#SBATCH --job-name=smart-slm-only-baseline-bbh
#SBATCH --nodelist=bob
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

echo "=== Running SLM-only Baseline for BBH Dataset ==="

# 基础参数配置（BBH数据集专用）
# SLM-only baseline: 只需要 SLM，不需要 LLM 或 PRM，也不需要阈值
BASE_EXTRA=(
  --approach=beam_search
  --smart_search=True
  --score_method=conf
  --n=1
  --beam_width=16
  --num_iterations=40
  --temperature=0.8
  --top_p=1.0
  --max_tokens=4096
  --lookahead=0
  --seed=0
  --dataset_name=lukaemon/bbh
  --dataset_split=test
  --data_name=bbh
  --dataset_start=10
  --dataset_end=20
  --run_slm_baseline=False
  --run_llm_baseline=False
  --run_random_baseline=False
  --run_slm_only_baseline=True
  --draft_model_path=/storage/ukp/shared/shared_model_weights/models--Qwen2.5-1.5B-Instruct
  --gpu_memory_utilization=0.8
  --search_batch_size=1
  --agg_strategy=last
  --sort_completed=False
  --system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Solve the reasoning problem efficiently and clearly:\n\n- For simple problems: Provide a concise solution.\n- For complex problems: Use step-by-step reasoning."
)

echo "=== Using LOCAL cache at: $LOCAL ==="
echo "=== Running: python scripts/test_time_compute.py ${BASE_EXTRA[*]} ==="

# 运行实验
# SLM-only baseline 不需要阈值参数，因为没有纠错步骤
srun python scripts/test_time_compute.py "${BASE_EXTRA[@]}"

# 检查上一个命令的退出状态
if [[ $? -ne 0 ]]; then
  echo "ERROR: Experiment failed with exit code $?" >&2
  exit 1
fi

echo "=== Experiment completed successfully ==="
echo "=== Results saved to: /storage/ukp/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/outputs/smart/slm_only/smart_slm_only/ ==="

