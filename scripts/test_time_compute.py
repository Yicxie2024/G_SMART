#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import random
import numpy as np
import os

import torch
from vllm import LLM
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score
from sal.search import (
    best_of_n,
    best_of_n_conf,
    best_of_n_smart,
    beam_search,
    beam_search_conf,
    beam_search_smart,
    beam_search_smart_conf,
    beam_search_smart_random_score,
    split_dataset_by_thresholds,
    beam_search_smart_conf_multi_threshold,
    split_dataset_by_uq_thresholds,
)
from sal.search.beam_search_smart_cocoa import smart_beam_search_cocoa as beam_search_smart_cocoa
from sal.search.beam_search_smart_cocoa_default import smart_beam_search_cocoa_default as beam_search_smart_cocoa_default
from sal.search.beam_search_smart_cocoa_multi_threshold import smart_beam_search_cocoa_multi_threshold as beam_search_smart_cocoa_multi_threshold
from datasets import Dataset

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

APPROACHES = {
    "beam_search": beam_search,
    "beam_search_smart": beam_search_smart,
    "beam_search_conf": beam_search_conf,
    "beam_search_smart_conf": beam_search_smart_conf,
    "beam_search_smart_random_score": beam_search_smart_random_score,
    "beam_search_smart_conf_multi_threshold": beam_search_smart_conf_multi_threshold,
    "beam_search_smart_cocoa": beam_search_smart_cocoa,
    "beam_search_smart_cocoa_default": beam_search_smart_cocoa_default,
    "beam_search_smart_cocoa_multi_threshold": beam_search_smart_cocoa_multi_threshold,
    "best_of_n": best_of_n,
    "best_of_n_smart": best_of_n_smart,
    "best_of_n_conf": best_of_n_conf,
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable optimizations for reproducibility


set_seed(42)


def main():
    # Set environment variables to avoid permission issues
    os.environ.setdefault('VLLM_USAGE_STATS_DISABLED', '1')
    os.environ.setdefault('TORCH_COMPILE_CACHE_DIR', '/mnt/beegfs/work/xie12/torch_compile_cache')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/mnt/beegfs/work/xie12/torch_inductor_cache')
    os.environ.setdefault('TORCH_LOGS_DIR', '/mnt/beegfs/work/xie12/torch_logs')
    
    # Ensure cache directories exist
    for cache_dir in [
        '/mnt/beegfs/work/xie12/torch_compile_cache',
        '/mnt/beegfs/work/xie12/torch_inductor_cache', 
        '/mnt/beegfs/work/xie12/torch_logs'
    ]:
        os.makedirs(cache_dir, exist_ok=True)
    
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    num_gpus = torch.cuda.device_count()
    print("=" * 20)
    print("The number of available GPUs:", num_gpus)

    # configure approach name
    # Check if random_thresholds is set for random score-based correction
    if getattr(config, 'random_thresholds', None) is not None and len(config.random_thresholds) > 0:
        approach_name = "beam_search_smart_random_score"
    # Check if uq_thresholds is set for multi-threshold correction
    elif getattr(config, 'uq_thresholds', None) is not None and len(config.uq_thresholds) > 0:
        # Determine if using cocoa or conf based on score_method
        if config.score_method.startswith("cocoa"):
            approach_name = "beam_search_smart_cocoa_multi_threshold"
        else:
            approach_name = "beam_search_smart_conf_multi_threshold"
    else:
        approach_suffix = "_smart" if config.smart_search else ""
        if config.score_method == "conf":
            approach_suffix += "_conf"
        elif config.score_method == "perplexity":
            approach_suffix += "_conf"  # perplexity uses the same approach as conf
        elif config.score_method == "msp":
            approach_suffix += "_conf"  # msp uses the same approach as conf
        elif config.score_method == "top2_margin":
            approach_suffix += "_conf"  # top2_margin uses the same approach as conf
        elif config.score_method.startswith("cocoa"):
            approach_suffix += "_cocoa" if not getattr(config, 'use_default_beam_search', False) else "_cocoa_default"
        elif config.score_method == "token_entropy":
            approach_suffix += "_conf"
        approach_name = config.approach + approach_suffix

    if approach_name not in APPROACHES:
        raise ValueError(f"Invalid score method: {config.score_method}")
    approach_fn = APPROACHES[approach_name]
    print("Approach name:", approach_name) 

    # log the search method and score method
    score_method_names = {
        "conf": "Confidence",
        "perplexity": "Perplexity", 
        "msp": "MSP",
        "top2_margin": "Top-2 Margin",
        "cocoa_msp": "CoCoA MSP",
        "cocoa_ppl": "CoCoA PPL", 
        "cocoa_entropy": "CoCoA Entropy",
        "token_entropy": "Token Entropy",
        "prm": "PRM"
    }
    score_name = score_method_names.get(config.score_method, "Unknown")
    if config.score_method.startswith("cocoa") and config.score_method not in score_method_names:
        score_name = "CoCoA"
    
    print(
        "\nUsing "
        + ("SMART" if config.smart_search else "Baseline")
        + " search.\nUsing "
        + score_name
        + " based score.\n"
    )
    if config.smart_search:
        print("Threshold:", config.threshold)
    print("N:", config.n)
    print("Beam width:", config.beam_width)
    print("=" * 20)

    if config.smart_search:
        mp.set_start_method("spawn", force=True)
        slm = LLM(
            model=config.draft_model_path,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=True,
            seed=config.seed,
            tensor_parallel_size=num_gpus,
            max_model_len=8192,
        )

        llm = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()

        if config.score_method == "prm":
            prm = load_prm(config)

            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "slm": slm, "prm": prm, "llm": llm},
                desc="Running search",
                load_from_cache_file=False,
            )
        elif config.score_method == "conf":
            # Confidence-based scoring doesn't need PRM model
            prm = None

            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "slm": slm, "prm": prm, "llm": llm},
                desc="Running search",
                load_from_cache_file=False,
            )
        elif config.score_method.startswith("cocoa"):
            # CoCoA methods don't need PRM model - they use semantic consistency
            prm = None

            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "slm": slm, "prm": prm, "llm": llm},
                desc="Running search",
                load_from_cache_file=False,
            )
        elif config.score_method == "perplexity":
            # Perplexity-based scoring doesn't need PRM model
            prm = None

            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "slm": slm, "prm": prm, "llm": llm},
                desc="Running search",
                load_from_cache_file=False,
            )
        elif config.score_method == "msp":
            # MSP-based scoring doesn't need PRM model
            prm = None

            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "slm": slm, "prm": prm, "llm": llm},
                desc="Running search",
                load_from_cache_file=False,
            )
        elif config.score_method == "top2_margin":
            # Top-2 margin-based scoring doesn't need PRM model
            prm = None

            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "slm": slm, "prm": prm, "llm": llm},
                desc="Running search",
                load_from_cache_file=False,
            )
        elif config.score_method == "token_entropy":
            # Token Entropy-based scoring doesn't need PRM model
            prm = None

            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "slm": slm, "prm": prm, "llm": llm},
                desc="Running search",
                load_from_cache_file=False,
            )
        elif approach_name == "beam_search_smart_random_score":
            # Random score-based correction doesn't need PRM model
            prm = None

            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "slm": slm, "prm": prm, "llm": llm},
                desc="Running search",
                load_from_cache_file=False,
            )
        elif approach_name == "beam_search_smart_conf_multi_threshold":
            # Multi-threshold confidence-based correction doesn't need PRM model
            prm = None

            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "slm": slm, "prm": prm, "llm": llm},
                desc="Running search",
                load_from_cache_file=False,
            )
        else:
            raise ValueError(f"Invalid score method: {config.score_method}")
    else:
        llm = LLM(
            model=config.model_path,
            revision="main",
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=True,
            seed=config.seed,
            tensor_parallel_size=num_gpus,
            max_model_len=8192,
        )

        if config.score_method == "prm":
            prm = load_prm(config)

            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "llm": llm, "prm": prm},
                desc="Running search",
                load_from_cache_file=False,
            )

        elif config.score_method == "conf":
            # Confidence-based scoring doesn't need PRM model
            prm = None

            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "llm": llm, "prm": prm},
                desc="Running search",
                load_from_cache_file=False,
            )
        elif config.score_method.startswith("cocoa"):
            prm = load_prm(config)

            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "llm": llm, "prm": prm},
                desc="Running search",
                load_from_cache_file=False,
            )
        elif config.score_method == "msp":
            # MSP-based scoring doesn't need PRM model
            prm = None

            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "llm": llm, "prm": prm},
                desc="Running search",
                load_from_cache_file=False,
            )
        elif config.score_method == "top2_margin":
            # Top-2 margin-based scoring doesn't need PRM model
            prm = None

            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "llm": llm, "prm": prm},
                desc="Running search",
                load_from_cache_file=False,
            )
        else:
            raise ValueError(f"Invalid score method: {config.score_method}")

    # Handle random score-based correction: split by thresholds and save separately
    if approach_name == "beam_search_smart_random_score":
        # Split dataset by thresholds
        threshold_datasets = split_dataset_by_thresholds(dataset, config)
        
        # Score and save each threshold's dataset separately
        import sys
        sys.path.append("src/evaluation")
        from evaluation.evaluate import evaluate
        import copy
        
        for threshold_str, threshold_dataset in threshold_datasets.items():
            # Create a copy of config for this threshold
            threshold_config = copy.deepcopy(config)
            threshold_config.uq_threshold = float(threshold_str.replace('_', '.'))
            # Set score_method for output file naming
            threshold_config.score_method = "random_score"
            # Set output_dir to ensure random_score uses a separate directory
            if threshold_config.output_dir is None:
                threshold_config.output_dir = "/storage/ukp/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/outputs/smart/random_score"
            
            # Score the threshold dataset
            threshold_dataset = score(threshold_dataset, threshold_config)
            
            # Save the threshold dataset
            save_dataset(threshold_dataset, threshold_config)
            
            # Evaluate for this threshold
            data_name_for_eval = getattr(threshold_config, 'data_name', 'math')
            keys = ["pred"]
            threshold_dataset, result = evaluate(
                data_name=data_name_for_eval, prompt_type=None, samples=threshold_dataset, pred_keys=keys
            )
            threshold_dataset = Dataset.from_list(
                [
                    {k: v for k, v in dict(sample).items() if k != "pred_completions"}
                    for sample in threshold_dataset
                ]
            )
            
            # Save evaluated dataset
            save_dataset(threshold_dataset, threshold_config)
            
            logger.info(f"Completed processing for threshold {threshold_str}: {result}")
        
        logger.info("Done 🔥!")
    # Handle multi-threshold correction: split by thresholds and save separately
    elif approach_name == "beam_search_smart_conf_multi_threshold" or approach_name == "beam_search_smart_cocoa_multi_threshold":
        # Split dataset by thresholds (works for both conf and cocoa)
        if approach_name == "beam_search_smart_cocoa_multi_threshold":
            # Import cocoa-specific split function
            from sal.search.beam_search_smart_cocoa_multi_threshold import split_dataset_by_thresholds as split_cocoa
            threshold_datasets = split_cocoa(dataset, config)
        else:
            threshold_datasets = split_dataset_by_uq_thresholds(dataset, config)
        
        # Score and save each threshold's dataset separately
        import sys
        sys.path.append("src/evaluation")
        from evaluation.evaluate import evaluate
        import copy
        
        for threshold_str, threshold_dataset in threshold_datasets.items():
            # Create a copy of config for this threshold
            threshold_config = copy.deepcopy(config)
            threshold_config.uq_threshold = float(threshold_str.replace('_', '.'))
            # Keep original score_method for output file naming
            # Set output_dir based on score_method
            score_method_name = getattr(threshold_config, 'score_method', 'conf')
            if threshold_config.output_dir is None:
                threshold_config.output_dir = f"/storage/ukp/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/outputs/smart/{score_method_name}_multi_threshold"
            
            # Score the threshold dataset
            threshold_dataset = score(threshold_dataset, threshold_config)
            
            # Save the threshold dataset
            save_dataset(threshold_dataset, threshold_config)
            
            # Evaluate for this threshold
            data_name_for_eval = getattr(threshold_config, 'data_name', 'math')
            keys = ["pred"]
            threshold_dataset, result = evaluate(
                data_name=data_name_for_eval, prompt_type=None, samples=threshold_dataset, pred_keys=keys
            )
            threshold_dataset = Dataset.from_list(
                [
                    {k: v for k, v in dict(sample).items() if k != "pred_completions"}
                    for sample in threshold_dataset
                ]
            )
            
            # Save evaluated dataset
            save_dataset(threshold_dataset, threshold_config)
            
            logger.info(f"Completed processing for threshold {threshold_str}: {result}")
        
        logger.info("Done 🔥!")
    else:
        # Original processing for other approaches
        dataset = score(dataset, config)
        save_dataset(dataset, config)

        import sys

        sys.path.append("src/evaluation")
        from evaluation.evaluate import evaluate

        if config.approach == "best_of_n" or config.approach == "beam_search":
            subsets = [2**i for i in range(config.n) if 2**i <= config.n]
            keys = []
            for n in subsets:
                keys.extend([f"pred_weighted@{n}", f"pred_maj@{n}", f"pred_naive@{n}"])
        else:
            keys = ["pred", "pred_randomg"]

        # Use data_name from config instead of hardcoding "math"
        data_name_for_eval = getattr(config, 'data_name', 'math')
        dataset, result = evaluate(
            data_name=data_name_for_eval, prompt_type=None, samples=dataset, pred_keys=keys
        )
        dataset = Dataset.from_list(
            [
                {k: v for k, v in dict(sample).items() if k != "pred_completions"}
                for sample in dataset
            ]
        )

        save_dataset(dataset, config)

        logger.info(result)
        logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
