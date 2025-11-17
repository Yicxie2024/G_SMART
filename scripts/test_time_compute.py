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
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

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
    beam_search_slm_only,
    beam_search_llm_only,
    beam_search_smart_prm_only,
)
from sal.search.beam_search_smart_uhead import smart_beam_search as beam_search_smart_uhead
from sal.search.beam_search_smart_cocoa import smart_beam_search_cocoa as beam_search_smart_cocoa
from sal.search.beam_search_smart_cocoa_default import smart_beam_search_cocoa_default as beam_search_smart_cocoa_default
from sal.search.beam_search_smart_cocoa_multi_threshold import smart_beam_search_cocoa_multi_threshold as beam_search_smart_cocoa_multi_threshold
from sal.search.beam_search_smart_conf_prm import smart_beam_search_conf as beam_search_smart_conf_prm
from datasets import Dataset

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

APPROACHES = {
    "beam_search": beam_search,
    "beam_search_smart": beam_search_smart,
    "beam_search_conf": beam_search_conf,
    "beam_search_smart_conf": beam_search_smart_conf,
    "beam_search_smart_conf_prm": beam_search_smart_conf_prm,
    "beam_search_smart_random_score": beam_search_smart_random_score,
    "beam_search_smart_conf_multi_threshold": beam_search_smart_conf_multi_threshold,
    "beam_search_smart_cocoa": beam_search_smart_cocoa,
    "beam_search_smart_cocoa_default": beam_search_smart_cocoa_default,
    "beam_search_smart_cocoa_multi_threshold": beam_search_smart_cocoa_multi_threshold,
    "beam_search_smart_uhead": beam_search_smart_uhead,
    "beam_search_slm_only": beam_search_slm_only,
    "beam_search_llm_only": beam_search_llm_only,
    "beam_search_smart_prm_only": beam_search_smart_prm_only,
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


def infer_model_scale(model_path: str):
    if not model_path:
        return None
    lowered = model_path.lower()
    if "14b" in lowered:
        return "14b"
    if "7b" in lowered:
        return "7b"
    # Default to 7B strategy for all non-14B models
    return "7b"


def plan_uhead_allocation(gpu_memory_gb: float):
    """Return (vllm_ratio, llm_memory_gb, uhead_memory_gb, strategy)."""
    if gpu_memory_gb >= 70:
        # 80GB class GPUs
        vllm_ratio = 0.28  # ~22GB for draft model + KV cache
        llm_memory_gb = max(22, int(gpu_memory_gb * 0.28))  # ~22GB for main LLM
        uhead_memory_gb = max(20, int(gpu_memory_gb * 0.25))  # Reserve for uHead passes/buffers
        strategy = "80GB: UHead (SLM + LLM + UHead buffers, ~75%)"
    elif gpu_memory_gb >= 35:
        # 40GB class GPUs
        vllm_ratio = 0.30  # ~12GB for draft model + KV cache
        llm_memory_gb = max(12, int(gpu_memory_gb * 0.28))  # ~11-12GB for main LLM
        uhead_memory_gb = max(11, int(gpu_memory_gb * 0.28))  # Reserve similar budget for UHead
        strategy = "40GB: UHead (SLM + LLM + UHead buffers, ~82%)"
    else:
        # Smaller GPUs: fall back to conservative shared budgeting
        vllm_ratio = 0.30
        llm_memory_gb = int(gpu_memory_gb * 0.30)
        uhead_memory_gb = int(gpu_memory_gb * 0.25)
        strategy = f"{gpu_memory_gb:.0f}GB: UHead (conservative allocation)"

    return vllm_ratio, llm_memory_gb, uhead_memory_gb, strategy


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
    # Check if SLM-only baseline is requested
    if getattr(config, 'run_slm_only_baseline', False):
        approach_name = "beam_search_slm_only"
    # Check if LLM-only baseline is requested
    elif getattr(config, 'run_llm_only_baseline', False):
        approach_name = "beam_search_llm_only"
    # Check if PRM-only method is requested
    elif getattr(config, 'use_prm_only', False) and config.score_method == "prm" and config.smart_search:
        approach_name = "beam_search_smart_prm_only"
    # Check if random_thresholds is set for random score-based correction
    elif getattr(config, 'random_thresholds', None) is not None and len(config.random_thresholds) > 0:
        approach_name = "beam_search_smart_random_score"
    # Check if uq_thresholds is set for multi-threshold correction
    elif getattr(config, 'uq_thresholds', None) is not None and len(config.uq_thresholds) > 0:
        # Determine if using cocoa or conf based on score_method
        if config.score_method.startswith("cocoa"):
            approach_name = "beam_search_smart_cocoa_multi_threshold"
        else:
            approach_name = "beam_search_smart_conf_multi_threshold"
    elif config.score_method == "uhead":
        if not config.smart_search:
            raise ValueError("uhead score_method requires smart_search=True")
        approach_name = "beam_search_smart_uhead"
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
        elif config.score_method == "token_sar":
            approach_suffix += "_conf"  # token_sar uses the same approach as conf
        elif config.score_method == "token_sar_conf_margin":
            approach_suffix += "_conf"  # fusion method uses confidence pipeline
        elif config.score_method == "hybrid_prm":
            approach_suffix += "_conf_prm"  # hybrid UQ + PRM correction
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
        "token_sar": "Token SAR",
        "token_sar_conf_margin": "Token SAR+Conf+Margin",
        "prm": "PRM",
    "hybrid_prm": "Hybrid UQ+PRM",
    "uhead": "UHead",
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

    # Handle SLM-only baseline (doesn't need SMART search setup)
    if approach_name == "beam_search_slm_only":
        # SLM-only baseline: only needs SLM, no LLM or PRM
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Detected GPU memory: {gpu_memory_gb:.1f} GB")
        logger.info("Running SLM-only baseline (single-pass generation)")
        
        # SLM-only: allocate more memory to SLM since it's the only model
        slm = LLM(
            model=config.draft_model_path,
            gpu_memory_utilization=0.8,  # Higher utilization since only SLM is used
            enable_prefix_caching=True,
            seed=config.seed,
            tensor_parallel_size=num_gpus,
            max_model_len=4096,  # Can use longer context for single-pass generation
        )
        prm = None
        llm = None
        
        dataset = get_dataset(config)
        # Remove columns that might conflict with returned fields
        columns_to_remove = [col for col in dataset.column_names 
                           if col in ["completions", "pred", "scores", "correction_counts", 
                                     "completion_times_uq", "llm_correction_tokens_uq", 
                                     "smart_step", "total_tokens", "correction_token_ratio"]]
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)
        dataset = dataset.map(
            approach_fn,
            batched=True,
            batch_size=config.search_batch_size,
            fn_kwargs={"config": config, "slm": slm, "prm": prm},
            desc="Running SLM-only baseline",
            load_from_cache_file=False,
        )
    elif approach_name == "beam_search_llm_only":
        # LLM-only baseline: only needs LLM, no SLM or PRM
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Detected GPU memory: {gpu_memory_gb:.1f} GB")
        logger.info("Running LLM-only baseline (single-pass generation)")
        
        # LLM-only: adaptive memory allocation based on GPU size
        # For 7B model (Qwen2.5-7B-Instruct):
        # - Model weights: ~14GB (BF16)
        # - KV cache: variable based on batch size and context length
        # - Activations: variable based on batch size
        if gpu_memory_gb > 70:
            # 80GB GPU: can use high utilization
            # 7B model: ~14GB weights + KV cache + activations = ~40-50GB total needed
            llm_memory_utilization = 0.75  # ~60GB, leaves good buffer for activations and KV cache
            max_model_len = 4096
            strategy = "80GB: LLM-only (7B model, 75% utilization, ~60GB)"
        elif gpu_memory_gb > 35:
            # 40GB GPU: need to be more conservative
            # 7B model needs ~14GB for weights, ~5-10GB for KV cache, ~3-5GB for activations
            # Total needed: ~22-29GB, so 62% gives ~25GB which is safer
            llm_memory_utilization = 0.62  # ~25GB, ensures model fits + buffer with margin
            max_model_len = 4096
            strategy = "40GB: LLM-only (7B model, 62% utilization, ~25GB)"
        else:
            # Small GPU: very conservative
            # May need to reduce context length or batch size for smaller GPUs
            llm_memory_utilization = 0.55  # Conservative allocation
            max_model_len = 2048  # Reduce context length for smaller GPUs
            strategy = f"{gpu_memory_gb:.0f}GB: LLM-only (7B model, 55% utilization, reduced context to 2048)"
        
        logger.info(f"Memory allocation strategy: {strategy}")
        logger.info(f"LLM memory utilization: {llm_memory_utilization*100:.0f}%")
        
        # LLM-only: allocate memory to LLM based on GPU size
        llm = LLM(
            model=config.model_path,
            gpu_memory_utilization=llm_memory_utilization,
            enable_prefix_caching=True,
            seed=config.seed,
            tensor_parallel_size=num_gpus,
            max_model_len=max_model_len,
        )
        prm = None
        slm = None
        
        dataset = get_dataset(config)
        # Remove columns that might conflict with returned fields
        columns_to_remove = [col for col in dataset.column_names 
                           if col in ["completions", "pred", "scores", "correction_counts", 
                                     "completion_times_uq", "llm_correction_tokens_uq", 
                                     "smart_step", "total_tokens", "correction_token_ratio"]]
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)
        dataset = dataset.map(
            approach_fn,
            batched=True,
            batch_size=config.search_batch_size,
            fn_kwargs={"config": config, "llm": llm, "prm": prm},
            desc="Running LLM-only baseline",
            load_from_cache_file=False,
        )
    elif config.smart_search:
        mp.set_start_method("spawn", force=True)
        
        # Calculate memory allocation based on GPU size
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Detected GPU memory: {gpu_memory_gb:.1f} GB")
        model_scale = infer_model_scale(config.model_path)
        logger.info(f"Detected model scale from path '{config.model_path}': {model_scale or 'unknown'}")
        prm_memory_gb = 0
        uhead_memory_gb = 0
        
        # Adaptive memory allocation strategy:
        # - Detect additional models needed based on score_method
        # - Adjust allocation to accommodate different model combinations
        # - Four scenarios: PRM, CoCoA, token_sar, pure UQ
        needs_prm = config.score_method in ["prm", "hybrid_prm"]
        needs_uhead = config.score_method == "uhead"
        needs_embedding = config.score_method.startswith("cocoa")  # Sentence Transformer
        needs_crossencoder = config.score_method in {"token_sar", "token_sar_conf_margin"}    # CrossEncoder
        use_8bit_llm = False
        
        if gpu_memory_gb > 70:
            # 80GB GPU
            if needs_prm:
                # PRM method: SLM + LLM + PRM (INT8)
                # SLM (1.5B): ~3GB model + KV cache ~5-6GB = ~9GB actual
                # LLM (7B): ~14GB model + activations ~3-5GB = ~19GB actual
                # PRM (8B INT8): ~8GB model + activations ~1-2GB = ~10GB actual
                # IMPORTANT: INT8 quantized PRM must be FULLY on GPU, cannot offload!
                vllm_ratio = 0.25      # 20GB
                llm_memory_gb = int(gpu_memory_gb * 0.25)   # 20GB
                prm_memory_gb = int(gpu_memory_gb * 0.15)   # 12GB (must fit INT8 model fully)
                strategy = "80GB: PRM (SLM + LLM + 8B PRM INT8, 65%)"
                # Total: 20 + 20 + 12 = 52GB (65%)
            elif needs_uhead:
                # UHead method: SLM + LLM + UHead adapter (shares base LLM forward)
                # Reserve additional headroom for UHead forward passes on full LLM weights
                vllm_ratio, llm_memory_gb, uhead_memory_gb, strategy = plan_uhead_allocation(gpu_memory_gb)
                prm_memory_gb = 0
                # Total: 22 + 22 + 20 ~= 64GB (80%)
            elif needs_embedding or needs_crossencoder:
                # CoCoA or token_sar: SLM + LLM + small model (~1-1.5GB)
                # Slightly reduce vLLM to reserve space for embedding/crossencoder
                # Small models auto-load to GPU
                vllm_ratio = 0.33      # 26GB (slightly reduced for extra model)
                llm_memory_gb = int(gpu_memory_gb * 0.30)   # 24GB
                prm_memory_gb = 0
                if needs_embedding:
                    strategy = "80GB: CoCoA (SLM + LLM + SentenceTransformer ~1GB, 65%)"
                else:
                    strategy = "80GB: TokenSAR (SLM + LLM + CrossEncoder ~1GB, 64%)"
                # Total: 26 + 24 + ~1.5 = 51.5GB (65%)
                
            else:
                # Pure UQ: SLM + LLM only (conf/perplexity/msp/top2_margin/token_entropy)
                # Maximize vLLM and LLM allocation
                vllm_ratio = 0.35      # 28GB (max KV cache)
                llm_memory_gb = int(gpu_memory_gb * 0.30)   # 24GB
                prm_memory_gb = 0
                strategy = "80GB: Pure UQ (SLM + LLM only, 65%)"
                # Total: 28 + 24 = 52GB (65%)
            
            if model_scale == "14b":
                # Force 14B to load in 8-bit even on 80GB GPUs to save memory headroom
                use_8bit_llm = True
                strategy += " [14B -> 8bit]"
                
        elif gpu_memory_gb > 35:
            # 40GB GPU
            if needs_prm:
                # PRM method: Tight allocation
                # PRM (8B INT8) must be fully on GPU, cannot offload
                # LLM may use offload if needed (BF16, not quantized)
                vllm_ratio = 0.30      # 12GB
                llm_memory_gb = max(18, int(gpu_memory_gb * 0.28))   # 11GB (may offload)
                prm_memory_gb = int(gpu_memory_gb * 0.28)   # 12GB (must fit INT8 fully)
                strategy = "40GB: PRM (SLM + LLM + 8B PRM INT8, 88%)"
                # Total: 12 + 11 + 12 = 35GB (88%)
                if model_scale == "14b":
                    use_8bit_llm = True
                    strategy += " [14B -> 8bit]"
            elif needs_uhead:
                # UHead method: reserve space for duplicate forward buffers
                vllm_ratio, llm_memory_gb, uhead_memory_gb, strategy = plan_uhead_allocation(gpu_memory_gb)
                prm_memory_gb = 0
                # Total: 12 + 12 + 11 ~= 35GB (82%)
                if model_scale == "14b":
                    use_8bit_llm = True
                    strategy += " [14B -> 8bit]"
            elif needs_embedding or needs_crossencoder:
                # CoCoA or token_sar: Reserve space for small model
                # Slightly reduce vLLM to avoid OOM with extra model
                vllm_ratio = 0.38      # 15GB (slightly reduced)
                llm_memory_gb = int(gpu_memory_gb * 0.35)   # 14GB
                prm_memory_gb = 0
                if needs_embedding:
                    strategy = "40GB: CoCoA (SLM + LLM + SentenceTransformer ~1GB, 76%)"
                else:
                    strategy = "40GB: TokenSAR (SLM + LLM + CrossEncoder ~1GB, 75%)"
                # Total: 15 + 14 + ~1.5 = 30.5GB (76%)
                if model_scale == "14b":
                    use_8bit_llm = True
                    strategy += " [14B -> 8bit]"
                
            else:
                # Pure UQ: Ensure LLM has enough memory to avoid offload
                # Qwen2.5-7B-Instruct needs 16-18GB to fully fit in GPU without offload
                vllm_ratio = 0.35      # 14GB for SLM
                llm_memory_gb = max(18, int(gpu_memory_gb * 0.38))  # At least 18GB for LLM to avoid offload
                prm_memory_gb = 0
                strategy = "40GB: Pure UQ (SLM + LLM only, ~80%)"
                # Total: 14 + 18 = 32GB (80%), ensures no offload for 7B model
                if model_scale == "14b":
                    # 14B on 40GB requires quantization to avoid offload
                    use_8bit_llm = True
                    strategy += " [14B -> 8bit]"
        else:
            # Small GPU: Conservative allocation
            vllm_ratio = 0.30
            llm_memory_gb = int(gpu_memory_gb * 0.30)
            prm_memory_gb = int(gpu_memory_gb * 0.30)
            if needs_uhead:
                _, _, uhead_memory_gb, strategy = plan_uhead_allocation(gpu_memory_gb)
            else:
                strategy = "Small GPU: Conservative allocation"
            if model_scale == "14b":
                use_8bit_llm = True
                strategy += " [14B -> 8bit]"
        
        if use_8bit_llm and model_scale == "14b":
            # Ensure enough budget for 14B 8bit by shrinking SLM allocation if needed
            buffer_ratio = 0.1 if gpu_memory_gb > 35 else 0.15
            target_llm_memory = 24 if gpu_memory_gb > 35 else max(20, int(gpu_memory_gb * 0.45))
            available_ratio_for_llm = max(0.0, 1 - vllm_ratio - buffer_ratio)
            available_llm_memory = int(gpu_memory_gb * available_ratio_for_llm)
            if available_llm_memory < target_llm_memory:
                deficit = target_llm_memory - available_llm_memory
                vllm_ratio = max(0.1, vllm_ratio - deficit / gpu_memory_gb)
                available_ratio_for_llm = max(0.0, 1 - vllm_ratio - buffer_ratio)
                available_llm_memory = int(gpu_memory_gb * available_ratio_for_llm)
            if available_llm_memory <= 0:
                raise ValueError(
                    "Not enough GPU memory to keep the SMART draft model and load the 14B LLM in 8-bit. "
                    "Consider reducing SMART parameters, using fewer thresholds, or switching to a larger GPU."
                )
            llm_memory_gb = max(llm_memory_gb, min(target_llm_memory, available_llm_memory))
            strategy += f" | Adjusted for 14B 8bit (LLM={llm_memory_gb}GB, vLLM={vllm_ratio*100:.0f}%)"
        
        logger.info(f"Memory allocation strategy: {strategy}")
        allocation_msg = f"Memory allocation: vLLM={vllm_ratio*100:.0f}%, LLM={llm_memory_gb}GB"
        if needs_prm and prm_memory_gb:
            allocation_msg += f", PRM={prm_memory_gb}GB"
        if needs_uhead and uhead_memory_gb:
            allocation_msg += f", UHead reserve={uhead_memory_gb}GB"
        logger.info(allocation_msg)
        
        # SLM: Dynamic allocation based on GPU size
        slm = LLM(
            model=config.draft_model_path,
            gpu_memory_utilization=vllm_ratio,
            enable_prefix_caching=True,
            seed=config.seed,
            tensor_parallel_size=num_gpus,
            max_model_len=2048,  # Reduced from 8192 to save KV cache memory
        )

        # LLM: Dynamic max memory based on GPU size
        max_memory_map = {i: f"{llm_memory_gb}GiB" for i in range(num_gpus or 1)}
        llm_load_kwargs = {
            "device_map": "auto",
            "max_memory": max_memory_map,
        }
        if use_8bit_llm:
            if BitsAndBytesConfig is None:
                raise ImportError("bitsandbytes is required for 8-bit loading but is not installed.")
            llm_load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            llm_load_kwargs["torch_dtype"] = torch.float16
            logger.info("Loading main LLM in 8-bit quantized mode for GPU fit.")
        else:
            llm_load_kwargs["torch_dtype"] = torch.bfloat16
        llm = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            **llm_load_kwargs,
        ).eval()

        if config.score_method == "prm" or approach_name == "beam_search_smart_prm_only":
            prm = load_prm(config, max_memory_gb=prm_memory_gb)

            dataset = get_dataset(config)
            # Remove columns that might conflict with returned fields for PRM-only
            if approach_name == "beam_search_smart_prm_only":
                columns_to_remove = [col for col in dataset.column_names 
                                   if col in ["completions", "pred", "scores", "correction_counts", 
                                             "completion_times_uq", "llm_correction_tokens_uq", 
                                             "smart_step", "total_tokens", "correction_token_ratio"]]
                if columns_to_remove:
                    dataset = dataset.remove_columns(columns_to_remove)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "slm": slm, "prm": prm, "llm": llm},
                desc="Running search",
                load_from_cache_file=False,
            )
        elif config.score_method == "uhead":
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
        elif config.score_method in {"token_sar", "token_sar_conf_margin"}:
            # Token SAR-based scoring (and its fusion variants) don't need PRM model
            prm = None

            # Initialize CrossEncoder once for all samples
            from sentence_transformers import CrossEncoder

            crossencoder = CrossEncoder(
                #"cross-encoder/stsb-roberta-large",
                "cross-encoder/nli-deberta-v3-large",
                device="cuda"
            )

            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "slm": slm, "prm": prm, "llm": llm, "crossencoder": crossencoder},
                desc="Running search",
                load_from_cache_file=False,
            )
        elif config.score_method == "hybrid_prm":
            # Hybrid UQ + PRM correction: needs PRM model for second-stage verification
            prm = load_prm(config, max_memory_gb=prm_memory_gb)

            dataset = get_dataset(config)
            # Get list of columns that might conflict with returned fields
            # Remove columns that will be overwritten by the map function
            columns_to_remove = [col for col in dataset.column_names 
                               if col in ["completions", "pred", "scores", "correction_counts", 
                                         "completion_times_uq", "llm_correction_tokens_uq", 
                                         "smart_step", "total_tokens", "correction_token_ratio"]]
            if columns_to_remove:
                dataset = dataset.remove_columns(columns_to_remove)
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
            
            # Initialize CrossEncoder if using token_sar method
            crossencoder = None
            if config.score_method in {"token_sar", "token_sar_conf_margin"}:
                from sentence_transformers import CrossEncoder
                crossencoder = CrossEncoder(
                    "cross-encoder/stsb-roberta-large",
                    device="cuda"
                )

            dataset = get_dataset(config)
            fn_kwargs = {"config": config, "slm": slm, "prm": prm, "llm": llm}
            if crossencoder is not None:
                fn_kwargs["crossencoder"] = crossencoder
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs=fn_kwargs,
                desc="Running search",
                load_from_cache_file=False,
            )
        else:
            raise ValueError(f"Invalid score method: {config.score_method}")
    else:
        # Calculate memory allocation for non-SMART search mode
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Detected GPU memory: {gpu_memory_gb:.1f} GB")
        
        # In non-SMART mode: LLM uses more memory, PRM uses less
        # - LLM: 60% of GPU memory
        # - PRM: 30% of GPU memory
        # - Buffer: 10%
        prm_memory_gb = int(gpu_memory_gb * 0.30)
        logger.info(f"Memory allocation (non-SMART): LLM=60%, PRM={prm_memory_gb}GB")
        
        llm = LLM(
            model=config.model_path,
            revision="main",
            gpu_memory_utilization=0.6,  # Higher ratio for LLM-only mode
            enable_prefix_caching=True,
            seed=config.seed,
            tensor_parallel_size=num_gpus,
            max_model_len=8192,
        )

        if config.score_method == "prm":
            prm = load_prm(config, max_memory_gb=prm_memory_gb)

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
            prm = load_prm(config, max_memory_gb=prm_memory_gb)

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
        elif config.score_method == "token_entropy":
            # Token Entropy-based scoring doesn't need PRM model
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
        elif config.score_method == "token_sar":
            # Token SAR-based scoring doesn't need PRM model (uses the same approach as conf)
            prm = None
            
            # Initialize CrossEncoder once for all samples
            from sentence_transformers import CrossEncoder
            crossencoder = CrossEncoder(
                "cross-encoder/stsb-roberta-large",
                device="cuda"
            )

            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "llm": llm, "prm": prm, "crossencoder": crossencoder},
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
    elif approach_name == "beam_search_slm_only":
        # SLM-only baseline: score and evaluate like other methods
        dataset = score(dataset, config)
        save_dataset(dataset, config)

        import sys
        sys.path.append("src/evaluation")
        from evaluation.evaluate import evaluate

        keys = ["pred"]
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
    elif approach_name == "beam_search_llm_only":
        # LLM-only baseline: score and evaluate like other methods
        dataset = score(dataset, config)
        save_dataset(dataset, config)

        import sys
        sys.path.append("src/evaluation")
        from evaluation.evaluate import evaluate

        keys = ["pred"]
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
