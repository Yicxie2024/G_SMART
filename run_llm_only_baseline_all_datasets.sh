#!/usr/bin/env bash
#SBATCH --job-name=smart-llm-only-all-datasets
#SBATCH --nodelist=remus
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

echo "=== Running LLM-only Baseline for All Datasets ==="
echo "This script will run LLM-only baseline sequentially for:"
echo "  1. BBH"
echo "  2. MATH-500"
echo "  3. MMLU-Pro"
echo ""
echo "GPU Settings:"
echo "  - Using GPU settings from this script (see #SBATCH directives above)"
echo "  - All three datasets will run sequentially in the same SLURM job"
echo "  - Sub-scripts are called via 'bash' (not 'sbatch'), so they use this job's GPU"
echo "  - No additional GPU allocation will be requested"
echo ""

# Use the working directory set by #SBATCH --chdir
# The script is located in the SMART directory, so sub-scripts are in bash_bbh/, bash_math500/, bash_mmlu/
# Since #SBATCH --chdir is set to /mnt/beegfs/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/SMART
# We can use relative paths from $PWD
cd /mnt/beegfs/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/SMART || exit 1

# Run BBH
echo "=========================================="
echo "=== Step 1/3: Running BBH Dataset ==="
echo "=========================================="
bash bash_bbh/run_slurm_llm_only_baseline.sh
BBH_EXIT_CODE=$?

if [[ $BBH_EXIT_CODE -ne 0 ]]; then
  echo "ERROR: BBH experiment failed with exit code $BBH_EXIT_CODE" >&2
  exit $BBH_EXIT_CODE
fi

echo ""
echo "BBH experiment completed successfully!"
echo ""

# Run MATH-500
echo "=========================================="
echo "=== Step 2/3: Running MATH-500 Dataset ==="
echo "=========================================="
bash bash_math500/run_slurm_llm_only_baseline.sh
MATH_EXIT_CODE=$?

if [[ $MATH_EXIT_CODE -ne 0 ]]; then
  echo "ERROR: MATH-500 experiment failed with exit code $MATH_EXIT_CODE" >&2
  exit $MATH_EXIT_CODE
fi

echo ""
echo "MATH-500 experiment completed successfully!"
echo ""

# Run MMLU-Pro
echo "=========================================="
echo "=== Step 3/3: Running MMLU-Pro Dataset ==="
echo "=========================================="
bash bash_mmlu/run_slurm_llm_only_baseline.sh
MMLU_EXIT_CODE=$?

if [[ $MMLU_EXIT_CODE -ne 0 ]]; then
  echo "ERROR: MMLU-Pro experiment failed with exit code $MMLU_EXIT_CODE" >&2
  exit $MMLU_EXIT_CODE
fi

echo ""
echo "MMLU-Pro experiment completed successfully!"
echo ""

echo "=========================================="
echo "=== All Experiments Completed Successfully! ==="
echo "=========================================="
echo "Results saved to:"
echo "  - BBH: /storage/ukp/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/outputs/smart/llm_only/smart_llm_only/"
echo "  - MATH-500: /storage/ukp/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/outputs/smart/llm_only/smart_llm_only/"
echo "  - MMLU-Pro: /storage/ukp/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/outputs/smart/llm_only/smart_llm_only/"

