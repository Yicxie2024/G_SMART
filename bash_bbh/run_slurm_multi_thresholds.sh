#!/usr/bin/env bash
#SBATCH --job-name=smart-multi-threshold
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

# ============================================================================
# 配置区域：请手动设置以下变量
# ============================================================================

# 配置文件名称（位于 configs/ 文件夹中）
#CONFIG_NAME="beam_search_conf_bbh.sh"
CONFIG_NAME="beam_search_cocoa_msp_bbh.sh"

# ============================================================================
# 加载配置
# ============================================================================

echo "=== Configuration ==="
echo "Config file: $CONFIG_NAME"
echo ""

# 加载基础参数配置文件
# 配置文件位于与脚本相同的目录下的 configs 子文件夹中
# 由于设置了 --chdir，工作目录已切换到 SMART 目录，使用相对路径
# 如果相对路径不存在，尝试使用绝对路径或脚本所在目录
if [[ -f "bash_bbh/configs/$CONFIG_NAME" ]]; then
  # 使用相对于 --chdir 的路径（推荐）
  CONFIG_FILE="bash_bbh/configs/$CONFIG_NAME"
elif [[ -f "/storage/ukp/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/SMART/bash_bbh/configs/$CONFIG_NAME" ]]; then
  # 使用绝对路径（fallback）
  CONFIG_FILE="/storage/ukp/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/SMART/bash_bbh/configs/$CONFIG_NAME"
else
  # 尝试使用脚本所在目录（最后的 fallback）
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "$PWD/bash_bbh")"
  CONFIG_FILE="$SCRIPT_DIR/configs/$CONFIG_NAME"
  
  if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Configuration file not found: $CONFIG_NAME" >&2
    echo "Tried paths:" >&2
    echo "  1. bash_bbh/configs/$CONFIG_NAME (relative to \$PWD: $PWD)" >&2
    echo "  2. /storage/ukp/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/SMART/bash_bbh/configs/$CONFIG_NAME" >&2
    echo "  3. $CONFIG_FILE (from script dir)" >&2
    echo "" >&2
    echo "Checking available config files:" >&2
    if [[ -d "bash_bbh/configs" ]]; then
      echo "  In bash_bbh/configs/:" >&2
      ls -1 "bash_bbh/configs/" 2>/dev/null | sed 's/^/    - /' || echo "    (none found)" >&2
    fi
    exit 1
  fi
fi

echo "=== Loading configuration from: $CONFIG_FILE ==="
source "$CONFIG_FILE"

# 验证配置文件是否提供了必需的变量
if [[ -z "${BASE_EXTRA:-}" ]]; then
  echo "ERROR: BASE_EXTRA not found in config file." >&2
  echo "Please ensure your config file defines BASE_EXTRA array." >&2
  exit 1
fi

if [[ -z "${UQ_THRESHOLDS:-}" ]]; then
  echo "ERROR: UQ_THRESHOLDS not found in config file." >&2
  echo "Please ensure your config file defines UQ_THRESHOLDS array." >&2
  exit 1
fi

echo "=== UQ thresholds: ${UQ_THRESHOLDS[*]} ==="
echo "=== Using LOCAL cache at: $LOCAL ==="
echo ""

echo "=== Running: python scripts/test_time_compute.py ${BASE_EXTRA[*]} --uq_thresholds ${UQ_THRESHOLDS[*]} ==="

# 运行实验
# 注意：uq_thresholds 需要使用空格分隔的多个参数值，而不是逗号分隔的字符串（与random_thresholds一致）
srun python scripts/test_time_compute.py "${BASE_EXTRA[@]}" --uq_thresholds "${UQ_THRESHOLDS[@]}"

# 检查上一个命令的退出状态
if [[ $? -ne 0 ]]; then
  echo "ERROR: Experiment failed with exit code $?" >&2
  exit 1
fi

echo "=== Experiment completed successfully ==="

