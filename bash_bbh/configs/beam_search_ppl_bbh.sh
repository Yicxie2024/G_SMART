#!/usr/bin/env bash
# 基础参数配置文件：beam_search + conf score method + bbh dataset
# 使用方法：在 bash 脚本中 source 此文件即可获得 BASE_EXTRA 数组和阈值配置

# 基础参数配置
# 参考 beam_search_smart_cocoa_bbh.yaml 中的配置
BASE_EXTRA=(
  --approach=beam_search
  --smart_search=True
  --score_method=perplexity
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
  --model_path=/storage/ukp/shared/shared_model_weights/models--Qwen2.5-7B-Instruct
  --draft_model_path=/storage/ukp/shared/shared_model_weights/models--Qwen2.5-1.5B-Instruct
  --gpu_memory_utilization=0.4
  --search_batch_size=1
  --agg_strategy=last
  --sort_completed=False
  --threshold=0.1
  --system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Solve the reasoning problem efficiently and clearly:\n\n- For simple problems: Provide a concise solution.\n- For complex problems: Use step-by-step reasoning."
)

# UQ thresholds 配置（用于多阈值实验）
# 可根据需要修改为不同的阈值组合
UQ_THRESHOLDS=(0.1 0.5 1.0 2.0 5.0)

