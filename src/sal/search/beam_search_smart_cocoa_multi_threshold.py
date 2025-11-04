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

"""
Multi-threshold experiment for CoCoA-based UQ-guided beam search.
Adapted from beam_search_smart_cocoa_default.py to support multiple UQ thresholds.
Baselines (random, slm_only, llm_only) have been removed.
"""

import copy
import logging
from collections import defaultdict
import time

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM
from sal.models.embedding_models import get_embedding_model

from .utils import (
    Beam,
    build_conv,
    generate_k_steps_with_responses,
    generate_k_steps_for_llm,
)

logger = logging.getLogger()
from sal.utils.score import aggregate_scores, calculate_cocoa_uq_scores

from transformers import AutoTokenizer


def _beam_search(
    batch_of_prompts, config: Config, slm: LLM, prm: PRM = None, llm: None = None, embedding_model = None
) -> tuple:
    """UQ-guided beam search with CoCoA-based uncertainty estimation."""
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
        logprobs=True,
    )

    # 只维护1个beam per prompt (不是config.n个)
    beams: list[Beam] = []
    start_time = time.time()  # Record start time for this beam
    for prompt in batch_of_prompts:
        beams.append(
            Beam(
                prompt=prompt,
                index=0,
                current_text="",
                next_texts=None,
                lookahead_texts=None,
                pruned=False,
                completed=False,  # New flag to track completion
                stop_reasons=None,
                history=[],
                best_scores=[],
                all_scores=[],
                previous_text=None,
                completion_tokens=[],
                smart_step=[],
                prm_update=[],  # we leave this empty
                gen_update=[],
                llm_tokens=[],
                llm_corrections=0,
                completion_time=0.0,  # Track total completion time
                llm_correction_tokens=0,  # Track total LLM correction tokens
            )
        )

    completed_beams: list[Beam] = []
    total_tokens = 0
    smart_done = False

    # Get embedding model (cached globally to avoid repeated loading)
    if embedding_model is None:
        embedding_model = get_embedding_model()

    for iterate_idx in tqdm(
        range(config.num_iterations), desc="UQ-guided generation", disable=False
    ):
        if iterate_idx == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        # 跳过扩展逻辑，因为只有1个beam per prompt
        # (原版这里会扩展到config.n个beams)

        if iterate_idx == config.num_iterations - 1:
            # Last iteration, generate to EOS
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                n=1,
                logprobs=True,
            )

        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in active_beams
        ]
        continue_final_message = iterate_idx > 0
        add_generation_prompt = iterate_idx == 0

        tokenizer = slm.get_tokenizer()
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        lookahead = 0 if iterate_idx == config.num_iterations - 1 else config.lookahead
        gen_results, responses = generate_k_steps_with_responses(
            templated_convs, lookahead, slm, sampling_params, config.beam_width, config.uq_sampling_temperature
        )

        prev_active_beams = copy.deepcopy(active_beams)

        # copy the active beams to regenerate the beams with llm
        prompts, completions = [], []
        for beam, gen_result in zip(active_beams, gen_results, strict=True):
            beam.next_texts = [gen_result.next_texts[0]]
            beam.stop_reasons = [gen_result.stop_reasons[0]]
            beam.lookahead_texts = [gen_result.lookahead_texts[0]]
            beam.completion_tokens += [gen_result.completion_tokens[0]]

            beam.current_text += beam.next_texts[0]
            beam.history.append(beam.next_texts[0])
            total_tokens += sum([gen_result.completion_tokens[0]])

            history_text = " ".join(beam.history)
            if len(tokenizer.encode(history_text)) > 2048:
                beam.completed = True
                beam.stop_reasons = ["length"]
                completed_beams.append(beam)
            elif (
                beam.stop_reasons[0] == "EOS"
                or beam.stop_reasons[0] == "length"
                or beam.next_texts[0] == ""
            ):
                beam.completed = True
                completed_beams.append(beam)
            prompts.append(beam.prompt)
            completions.append([beam.current_text])
        
        # Use the embedding model for semantic consistency calculation
        def _detok(ids):
            return tokenizer.decode(ids, skip_special_tokens=True)
        def _embed_fn(texts):
            vecs = embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
            return vecs
        
        # 只有1个beam，直接计算其UQ分数
        all_outputs = [o for r in responses for o in r.outputs]
        assert len(all_outputs) == config.beam_width, f"Expected {config.beam_width} outputs, got {len(all_outputs)}"
        
        # 提取logprobs（1个greedy + beam_width-1个samples）
        beam_logprobs_list = [output.logprobs for output in all_outputs]
        cocoa_msp, cocoa_ppl, cocoa_entropy, cocoa_confidence = calculate_cocoa_uq_scores(
            beam_logprobs_list, detok=_detok, embed_fn=_embed_fn
        )
        
        if config.score_method == "cocoa_msp":
            uq_score = cocoa_msp
        elif config.score_method == "cocoa_ppl":
            uq_score = cocoa_ppl
        elif config.score_method == "cocoa_entropy":
            uq_score = cocoa_entropy
        else:
            raise ValueError(f"Invalid score method: {config.score_method}")
        
        conf_agg_scores = [[uq_score]]  # 只有1个beam
        active_beams[0].all_scores.append(uq_score)

        # Now filter active_beams and agg_scores for beams that are completed
        conf_agg_scores = [
            conf_agg_scores[i] for i, b in enumerate(active_beams) if not b.completed
        ]

        prev_active_beams = [
            b
            for idx, b in enumerate(prev_active_beams)
            if not active_beams[idx].completed
        ]
        active_beams = [b for b in active_beams if not b.completed]

        # Early stopping if all beams are completed (只有1个beam，完成就停止)
        if len(active_beams) == 0:
            break

        # SMART beam search implementation
        # 只有1个beam，conf_agg_scores长度为1，直接判断是否需要LLM纠错
        assert len(conf_agg_scores) == 1, f"Expected 1 beam, got {len(conf_agg_scores)}"
        assert len(active_beams) == 1, f"Expected 1 active beam, got {len(active_beams)}"
        
        uq_score = conf_agg_scores[0][0]
        if uq_score <= config.uq_threshold:
            # UQ分数低，SLM生成的结果可信，不需要LLM纠错
            continue

        # UQ分数高，需要用LLM纠错
        smart_done = True
        beam = prev_active_beams[0]  # 只有1个beam

        conv = build_conv(beam.prompt, beam.current_text, config.system_prompt)
        continue_final_message = iterate_idx > 0
        add_generation_prompt = iterate_idx == 0

        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_conv = tokenizer.apply_chat_template(
            [conv],
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        lookahead = 0 if iterate_idx == config.num_iterations - 1 else config.lookahead
        gen_results = generate_k_steps_for_llm(
            tokenizer, templated_conv, lookahead, llm, config, 1
        )
        gen_result = gen_results[0]

        # 记录SLM生成的step（修正前）
        slm_text = active_beams[0].next_texts[0]
        # LLM修正后的step
        llm_text = gen_result.next_texts[0]
        
        # update the beam
        beam.next_texts = gen_result.next_texts
        beam.stop_reasons = gen_result.stop_reasons
        beam.lookahead_texts = gen_result.lookahead_texts
        beam.current_text += beam.next_texts[0]
        beam.history.append(beam.next_texts[0])

        if (
            beam.stop_reasons[0] == "EOS"
            or beam.stop_reasons[0] == "length"
            or beam.next_texts[0] == ""
        ):
            beam.completed = True
            completed_beams.append(beam)

        # log correction information
        beam.smart_step.append(iterate_idx)
        beam.gen_update.append((slm_text, llm_text))
        llm_token_count = len(tokenizer.encode(beam.next_texts[0]))
        beam.llm_tokens.append(llm_token_count)
        beam.llm_correction_tokens = getattr(beam, "llm_correction_tokens", 0) + llm_token_count
        total_tokens += llm_token_count
        # reuse the original confidence scores
        beam.all_scores = active_beams[0].all_scores
        
        active_beams[0] = beam
        beam.llm_corrections = getattr(beam, "llm_corrections", 0) + 1

    # Record completion time for all beams
    end_time = time.time()
    completion_time = end_time - start_time
    for beam in completed_beams:
        beam.completion_time = completion_time
    
    if smart_done == False:
        for beam in completed_beams:
            beam.smart_step = [-1]
            beam.gen_update = [("-1", "-1")]
            beam.llm_tokens = [-1]
            beam.llm_correction_tokens = 0  # No LLM corrections

    # recalculate prm scores for completed beams (optional for CoCoA methods)
    if prm is not None:
        prompts = [b.prompt for b in completed_beams]
        completions = [[b.current_text] for b in completed_beams]
        prm_scores = prm.score(prompts, completions)
    else:
        # CoCoA methods don't use PRM scores
        prm_scores = [[0.0] for _ in completed_beams]

    # Don't delete sbert here - it will be reused across samples

    return completed_beams, total_tokens, prm_scores


def smart_beam_search_cocoa_multi_threshold(examples, config: Config, slm: LLM, prm: PRM = None, llm: None = None):
    """
    Multi-threshold experiment for CoCoA-based UQ-guided beam search.
    Runs the UQ-guided method with multiple thresholds.
    Baselines (random, slm_only, llm_only) are removed.
    
    Returns a flattened dictionary with threshold-prefixed field names
    (e.g., "0_1_completions", "0_2_completions", etc.) that can be
    split later using split_dataset_by_thresholds.
    """
    problems = examples["problem"]
    
    # Get embedding model (cached globally to avoid repeated loading across samples)
    embedding_model = get_embedding_model()
    
    # Get UQ thresholds from config
    uq_thresholds = getattr(config, 'uq_thresholds', [config.uq_threshold])
    
    logger.info(f"Running multi-threshold experiment with thresholds: {uq_thresholds}")
    
    # Results dictionary: flatten all threshold results into a single dict
    # with threshold-prefixed field names
    all_results = {}
    
    for uq_threshold in uq_thresholds:
        logger.info(f"Processing threshold: {uq_threshold}")
        
        # Format threshold string for field names (e.g., 0.1 -> "0_1")
        threshold_str = f"{uq_threshold:.6f}".rstrip('0').rstrip('.')
        if threshold_str == "":
            threshold_str = "0"
        threshold_str = threshold_str.replace('.', '_')
        
        # Update config with current threshold
        config.uq_threshold = uq_threshold
        
        # Run UQ-guided correction with current threshold
        beam_results, total_tokens, prm_scores = _beam_search(
            problems, config, slm, prm, llm, embedding_model
        )
        
        # Group results by prompt
        grouped_results = defaultdict(list)
        for results in beam_results:
            grouped_results[results.prompt].append(results)
        
        # Build results lists for this threshold
        completions = []
        preds = []
        scores = []
        correction_counts = []
        completion_times = []
        llm_correction_tokens = []
        smart_steps = []
        correction_token_ratios = []
        
        for p in problems:
            beams = grouped_results[p]
            beam_completions = [b.current_text for b in beams]
            beam_scores = [b.all_scores for b in beams]
            pred = beam_completions[0] if len(beam_completions) > 0 else ""
            counts = [getattr(b, "llm_corrections", 0) for b in beams]
            times = [getattr(b, "completion_time", 0.0) for b in beams]
            tokens = [getattr(b, "llm_correction_tokens", 0) for b in beams]
            steps = [getattr(b, "smart_step", []) for b in beams]
            
            # Calculate correction_token_ratio for each beam
            beam_correction_ratios = []
            for b in beams:
                slm_tokens = sum(getattr(b, "completion_tokens", []))
                llm_corr_tokens_single = getattr(b, "llm_correction_tokens", 0)
                total_tokens_single = slm_tokens + llm_corr_tokens_single
                # Avoid division by zero
                if total_tokens_single > 0:
                    ratio = llm_corr_tokens_single / total_tokens_single
                else:
                    ratio = 0.0
                beam_correction_ratios.append(ratio)
            
            completions.append(beam_completions)
            preds.append(pred)
            scores.append(beam_scores)
            correction_counts.append(counts)
            completion_times.append(times)
            llm_correction_tokens.append(tokens)
            smart_steps.append(steps)
            correction_token_ratios.append(beam_correction_ratios)
        
        # Store results with threshold-prefixed field names
        all_results[f"{threshold_str}_completions"] = completions
        all_results[f"{threshold_str}_pred"] = preds
        all_results[f"{threshold_str}_scores"] = scores
        all_results[f"{threshold_str}_correction_counts"] = correction_counts
        all_results[f"{threshold_str}_completion_times"] = completion_times
        all_results[f"{threshold_str}_llm_correction_tokens"] = llm_correction_tokens
        all_results[f"{threshold_str}_smart_step"] = smart_steps
        all_results[f"{threshold_str}_correction_token_ratio"] = correction_token_ratios
        
        logger.info(f"Completed threshold: {uq_threshold}")
    
    # Embedding model is managed globally, no cleanup needed here
    
    return all_results


def split_dataset_by_thresholds(dataset, config: Config):
    """Split a dataset containing multiple threshold results into separate datasets.
    
    This function takes a dataset that was processed with smart_beam_search_cocoa_multi_threshold
    (which contains results for multiple thresholds) and splits it into separate datasets,
    one for each threshold. Each threshold's dataset will have standard field names
    (pred, completions, scores, etc.) instead of threshold-prefixed names.
    
    Args:
        dataset: Dataset containing results with threshold-prefixed field names
        config: Config object with uq_thresholds attribute
        
    Returns:
        dict: Dictionary mapping threshold strings to separate datasets
    """
    from datasets import Dataset
    
    uq_thresholds = getattr(config, 'uq_thresholds', [])
    if not uq_thresholds:
        raise ValueError("config.uq_thresholds must be provided and non-empty")
    
    # First, compute all threshold strings to know which fields to exclude
    all_threshold_strs = []
    for uq_threshold in uq_thresholds:
        threshold_str = f"{uq_threshold:.6f}".rstrip('0').rstrip('.')
        if threshold_str == "":
            threshold_str = "0"
        threshold_str = threshold_str.replace('.', '_')
        all_threshold_strs.append(threshold_str)
    
    threshold_datasets = {}
    
    for uq_threshold in uq_thresholds:
        # Format threshold string to match field names
        threshold_str = f"{uq_threshold:.6f}".rstrip('0').rstrip('.')
        if threshold_str == "":
            threshold_str = "0"
        threshold_str = threshold_str.replace('.', '_')
        
        # Get list of other threshold strings (to exclude their fields)
        other_threshold_strs = [t for t in all_threshold_strs if t != threshold_str]
        
        # Extract fields for this threshold
        threshold_data = []
        for i in range(len(dataset)):
            example = dataset[i]
            threshold_example = {}
            # Copy original fields (problem, etc.)
            for key, value in example.items():
                # Skip uq_thresholds field
                if key == "uq_thresholds":
                    continue
                # Skip fields that start with any other threshold prefix
                if any(key.startswith(f"{other_threshold_str}_") for other_threshold_str in other_threshold_strs):
                    continue
                # Skip fields that start with current threshold prefix (we'll rename them below)
                if key.startswith(f"{threshold_str}_"):
                    continue
                # Copy all other fields (e.g., problem, solution, answer, etc.)
                threshold_example[key] = value
            
            # Rename threshold-prefixed fields to standard names
            if f"{threshold_str}_completions" in example:
                threshold_example["completions"] = example[f"{threshold_str}_completions"]
            if f"{threshold_str}_pred" in example:
                threshold_example["pred"] = example[f"{threshold_str}_pred"]
            if f"{threshold_str}_scores" in example:
                threshold_example["scores"] = example[f"{threshold_str}_scores"]
            if f"{threshold_str}_correction_counts" in example:
                threshold_example["correction_counts"] = example[f"{threshold_str}_correction_counts"]
            if f"{threshold_str}_completion_times" in example:
                threshold_example["completion_times_uq"] = example[f"{threshold_str}_completion_times"]
            if f"{threshold_str}_llm_correction_tokens" in example:
                threshold_example["llm_correction_tokens_uq"] = example[f"{threshold_str}_llm_correction_tokens"]
            if f"{threshold_str}_smart_step" in example:
                threshold_example["smart_step"] = example[f"{threshold_str}_smart_step"]
            if f"{threshold_str}_correction_token_ratio" in example:
                threshold_example["correction_token_ratio"] = example[f"{threshold_str}_correction_token_ratio"]
            
            threshold_data.append(threshold_example)
        
        # Create a Dataset for this threshold
        threshold_dataset = Dataset.from_list(threshold_data)
        threshold_datasets[threshold_str] = threshold_dataset
        
        logger.info(f"Created dataset for threshold {uq_threshold} (key: {threshold_str}) with {len(threshold_dataset)} examples")
    
    return threshold_datasets

