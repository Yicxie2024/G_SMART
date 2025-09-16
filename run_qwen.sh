#!/bin/bash
#SBATCH --job-name=llama-ttc
#SBATCH --nodelist=scratchy
#SBATCH --partition=gpu
#SBATCH --qos=gpu-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=/storage/ukp/work/xie12/uncertainty-guided-reasoning/logs/slurm-%x-%j.out
#SBATCH --error=/storage/ukp/work/xie12/uncertainty-guided-reasoning/logs/slurm-%x-%j.err
# 如需一次性跑 0..5 六个分支，取消下一行注释：
# #SBATCH --array=0-5

set -euo pipefail

echo "=== Job info ==="
echo "JOB: $SLURM_JOB_ID  ARRAY_ID: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "HOST: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
date

# ========= 环境激活 =========
echo "=== Activating env ==="
source /storage/ukp/work/xie12/miniconda3/bin/activate smart

# ========= 缓存/日志/令牌 =========
echo "=== Caches ==="
export HF_HOME=/mnt/beegfs/work/xie12/tmp/huggingface
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$HF_HOME/.cache"
export MPLCONFIGDIR=/mnt/beegfs/work/xie12/tmp/matplotlib
export VLLM_USAGE_DIR=/mnt/beegfs/work/xie12/tmp/vllm_usage/${SLURM_JOB_ID}

# ==== vLLM ====
export VLLM_NO_USAGE_COLLECTION=1            # 避免 /home 写权限报错
export VLLM_GPU_MEMORY_UTILIZATION=0.40



# （如需从 Hub 拉私有模型或用在线追踪才需要这些）
export WANDB_API_KEY=37a24d82af6d6090d998bf517067ef84c4575c85

# Triton 缓存按作业隔离，结束即清理
export TRITON_CACHE_DIR="/mnt/beegfs/work/xie12/triton_cache/${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"
trap 'echo "Cleaning TRITON_CACHE_DIR"; rm -rf "$TRITON_CACHE_DIR"' EXIT

mkdir -p /storage/ukp/work/xie12/uncertainty-guided-reasoning/logs
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TRANSFORMERS_CACHE" "$VLLM_USAGE_DIR"

# ========= 其他环境变量（保持与你原脚本一致）=========
export CUDA_LAUNCH_BLOCKING=1
export SEED=0

# ========= 选择 OPTION（优先用脚本参数；否则用数组下标；否则默认为 0）=========
OPTION="${1:-${SLURM_ARRAY_TASK_ID:-0}}"
echo "Using OPTION=$OPTION"

# ========= 与你原脚本等价的分支路由 =========
CONFIG=""
EXTRA_ARGS="--n=16"

case "$OPTION" in
  # ############ run large model ############
  # run large, best-of-n, score-method=prm
  0)
    CONFIG="recipes/Llama-3.1-8B-Instruct/best_of_n.yaml"
    EXTRA_ARGS="$EXTRA_ARGS --beam_width=1"
    ;;

  # run large, beam-search, score-method=prm
  1)
    CONFIG="recipes/Llama-3.1-8B-Instruct/beam_search.yaml"
    EXTRA_ARGS="$EXTRA_ARGS --beam_width=4"
    ;;

  # run large, beam-search, score-method=conf
  2)
    CONFIG="recipes/Llama-3.1-8B-Instruct/beam_search.yaml"
    EXTRA_ARGS="$EXTRA_ARGS --beam_width=4 --score_method=conf"
    ;;

  # ############ run small model ############
  # run small, best-of-n, score-method=prm
  3)
    CONFIG="recipes/Llama-3.2-1B-Instruct/best_of_n.yaml"
    EXTRA_ARGS="$EXTRA_ARGS --beam_width=1"
    ;;

  # ############ run smart ############
  # run smart, best-of-n, score-method=prm
  4)
    CONFIG="recipes/Qwen2.5-1.5B-Instruct/beam_search.yaml"
    EXTRA_ARGS="$EXTRA_ARGS --beam_width=1"
    ;;

  # run smart, beam-search, score-method=prm
  5)
    CONFIG="recipes/Qwen2.5-7B-Instruct/beam_search_smart.yaml"
    EXTRA_ARGS="$EXTRA_ARGS --beam_width=4"
    ;;
  
  # run smart, beam-search, score-method=sse 
  6)
    CONFIG="recipes/Qwen2.5-7B-Instruct/beam_search_smart_sse.yaml"
    EXTRA_ARGS=""   # YAML 里已经写好了，不需要再传
    ;;
    *)
    echo "Unknown OPTION=$OPTION (valid: 0..6)"; exit 1;;
esac

echo "CONFIG: $CONFIG"
echo "ARGS:   $EXTRA_ARGS"

echo "=== Running test_time_compute.py ==="
if [[ -n "$CONFIG" ]]; then
  # CONFIG 非空时，既传 config 文件也传额外参数
  srun --unbuffered python scripts/test_time_compute.py "$CONFIG" $EXTRA_ARGS
else
  # CONFIG 为空（例如 OPTION=6），只传命令行参数，避免空字符串参数
  srun --unbuffered python scripts/test_time_compute.py $EXTRA_ARGS
fi

echo "=== Done ==="
date
