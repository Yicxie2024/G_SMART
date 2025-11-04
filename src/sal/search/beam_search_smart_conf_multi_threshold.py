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
import copy
import logging
from collections import defaultdict
import time

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM

from .utils import (
    Beam,
    build_conv,
    generate_k_steps_with_responses,
    last,
    generate_k_steps_for_llm,
)

logger = logging.getLogger()
from sal.utils.score import (
    aggregate_scores, 
    calculate_confidence_score, 
    calculate_perplexity_score,
    calculate_top2_margin_scores,
    calculate_msp_scores,
    calculate_token_entropy_scores
)

from transformers import AutoTokenizer


def _beam_search_conf(
    batch_of_prompts, config: Config, slm: LLM, prm: PRM, llm: None, uq_threshold: float
) -> tuple:
    """Beam search with confidence-based correction.
    
    Args:
        uq_threshold: Threshold for uncertainty score. Behavior depends on score_method:
            - conf: correct if score < threshold
            - perplexity: correct if score > threshold
            - top2_margin: correct if score < threshold
            - msp: correct if score < threshold
            - token_entropy: correct if score > threshold
    """
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
        logprobs=True,
    )

    # Only maintain a single beam per prompt
    beams: list[Beam] = []
    start_time = time.time()
    for prompt in batch_of_prompts:
        beams.append(
            Beam(
                prompt=prompt,
                index=0,
                current_text="",
                next_texts=None,
                lookahead_texts=None,
                pruned=False,
                completed=False,
                stop_reasons=None,
                history=[],
                best_scores=[],
                all_scores=[],
                previous_text=None,
                completion_tokens=[],
                smart_step=[],
                prm_update=[],
                gen_update=[],
                llm_tokens=[],
                llm_corrections=0,
                completion_time=0.0,
                llm_correction_tokens=0,
            )
        )

    completed_beams: list[Beam] = []
    total_tokens = 0
    smart_done = False

    for iterate_idx in tqdm(
        range(config.num_iterations), desc=f"Confidence beam search (threshold={uq_threshold:.3f})"
    ):
        # With single beam per prompt, active_beams are simply the non-pruned, non-completed beams
        if iterate_idx == 0:
            active_beams = [b for b in beams if not b.pruned and not b.completed]
        else:
            active_beams = [b for b in active_beams if not b.pruned]
            
        if len(active_beams) == 0:
            break

        # Last iteration, generate to EOS
        if iterate_idx == config.num_iterations - 1:
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
            templated_convs, lookahead, slm, sampling_params, 1
        )

        prev_active_beams = copy.deepcopy(active_beams)

        prompts, completions = [], []
        for beam, gen_result in zip(active_beams, gen_results, strict=True):
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            beam.completion_tokens += gen_result.completion_tokens

            beam.current_text += beam.next_texts[0]
            beam.history.append(beam.next_texts[0])
            total_tokens += sum(gen_result.completion_tokens)

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

        # Confidence scores based on token logprobs
        conf_scores = []
        for output in [o for r in responses for o in r.outputs]:
            if config.score_method == "conf":
                likelihood, likelihood_mean, probs_mean = calculate_confidence_score(output.logprobs)
                conf_scores.append([likelihood_mean])  # Use normalized likelihood
            elif config.score_method == "perplexity":
                perplexity, normalized_perplexity, token_perplexity = calculate_perplexity_score(output.logprobs)
                conf_scores.append([normalized_perplexity])  # Use normalized perplexity
            elif config.score_method == "top2_margin":
                # Top-2 margin method: use min margin as uncertainty score
                min_margin, mean_margin, margins = calculate_top2_margin_scores([output.logprobs])
                conf_scores.append([min_margin])
            elif config.score_method == "msp":
                # MSP method: use MSP score as uncertainty score
                log_likelihood, probability = calculate_msp_scores([output.logprobs])
                conf_scores.append([probability])
            elif config.score_method == "token_entropy":
                # Token Entropy method: use base token entropy score (simplified for single beam)
                max_entropy, mean_entropy, entropies = calculate_token_entropy_scores([output.logprobs])
                conf_scores.append([mean_entropy])
            else:
                # Default to confidence score for backward compatibility
                likelihood, likelihood_mean, probs_mean = calculate_confidence_score(output.logprobs)
                conf_scores.append([likelihood_mean])

        # All score methods now return single values, no need for format detection
        conf_agg_scores = [[score[0]] for score in conf_scores]

        for beam, score in zip(active_beams, conf_scores, strict=True):
            beam.all_scores.append(score[0])

        # Filter for incomplete beams for potential correction
        conf_agg_scores = [
            conf_agg_scores[i] for i, b in enumerate(active_beams) if not b.completed
        ]

        prev_active_beams = [
            b
            for idx, b in enumerate(prev_active_beams)
            if not active_beams[idx].completed
        ]
        active_beams = [b for b in active_beams if not b.completed]

        # Early stopping if all beams are completed
        if len(active_beams) == 0:
            break

        # SMART single-beam correction: check if correction is needed based on score_method and threshold
        if config.score_method == "conf":
            need_correction = conf_agg_scores and conf_agg_scores[0][0] < uq_threshold
        elif config.score_method == "perplexity":
            need_correction = conf_agg_scores and conf_agg_scores[0][0] > uq_threshold
        elif config.score_method == "top2_margin":
            # For top2 margin, lower values indicate more uncertainty, so correct if below threshold
            need_correction = conf_agg_scores and conf_agg_scores[0][0] < uq_threshold
        elif config.score_method == "msp":
            # For MSP, lower values indicate more uncertainty, so correct if below threshold
            need_correction = conf_agg_scores and conf_agg_scores[0][0] < uq_threshold
        elif config.score_method == "token_entropy":
            # For Token Entropy, higher values indicate more uncertainty, so correct if above threshold
            need_correction = conf_agg_scores and conf_agg_scores[0][0] > uq_threshold
        else:
            # Default to confidence score behavior
            need_correction = conf_agg_scores and conf_agg_scores[0][0] < uq_threshold
        
        if not need_correction:
            continue

        smart_done = True
        re_beams = [prev_active_beams[0]]

        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt) for b in re_beams
        ]
        continue_final_message = iterate_idx > 0
        add_generation_prompt = iterate_idx == 0

        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        lookahead = 0 if iterate_idx == config.num_iterations - 1 else config.lookahead
        gen_results = generate_k_steps_for_llm(
            tokenizer, templated_convs, lookahead, llm, config, 1
        )

        for beam, gen_result in zip(re_beams, gen_results, strict=True):
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

        # Log correction information and replace the active beam
        beam = re_beams[0]
        beam.smart_step.append(iterate_idx)
        beam.gen_update.append((prev_active_beams[0].next_texts[0], beam.next_texts[0]))
        llm_token_count = len(tokenizer.encode(beam.next_texts[0]))
        beam.llm_tokens.append(llm_token_count)
        beam.llm_correction_tokens = getattr(beam, "llm_correction_tokens", 0) + llm_token_count
        total_tokens += llm_token_count
        # reuse the original confidence scores
        beam.all_scores = prev_active_beams[0].all_scores
        active_beams[0] = beam
        beam.llm_corrections = getattr(beam, "llm_corrections", 0) + 1

    # If no beam was marked completed, finalize the current state
    if len(completed_beams) == 0 and len(beams) > 0:
        completed_beams = [beams[0]]

    # We do not duplicate to match config.n; keep a single completion per prompt
    if config.sort_completed and len(completed_beams) > 1:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[:1]
    else:
        completed_beams = completed_beams[:1]

    # Record completion time for all beams
    end_time = time.time()
    completion_time = end_time - start_time
    for beam in completed_beams:
        beam.completion_time = completion_time

    if smart_done is False:
        for beam in completed_beams:
            beam.smart_step = [-1]
            beam.gen_update = [("-1", "-1")]
            beam.llm_tokens = [-1]
            beam.llm_correction_tokens = 0

    # recalculate prm scores for completed beams (if PRM model is available)
    if prm is not None:
        prompts = [b.prompt for b in completed_beams]
        completions = [[b.current_text] for b in completed_beams]
        prm_scores = prm.score(prompts, completions)
    else:
        prm_scores = None

    return completed_beams, total_tokens, prm_scores


def smart_beam_search_conf_multi_threshold(examples, config: Config, slm: LLM, prm: PRM, llm: None):
    """Confidence-based SMART beam search for multiple thresholds.
    
    This function runs confidence-based correction for each threshold in config.uq_thresholds.
    Each threshold will be tested independently, and results are organized by threshold.
    
    Returns:
        dict: Dictionary containing results for all thresholds. Keys are formatted as:
              - "uq_thresholds": list of threshold values used
              - For each threshold, fields are prefixed with threshold string:
                - "{threshold}_completions", "{threshold}_pred", "{threshold}_scores", etc.
              This structure allows the calling code to extract results for each threshold
              and save them to separate files.
    """
    problems = examples["problem"]
    
    # Get uq thresholds from config
    uq_thresholds = getattr(config, 'uq_thresholds', [])
    if not uq_thresholds:
        raise ValueError("config.uq_thresholds must be provided and non-empty")
    
    # Results dictionary: flatten all threshold results into a single dict
    # with threshold-prefixed field names
    # Note: uq_thresholds is not stored per-sample to avoid Arrow schema issues
    # It's available in config.uq_thresholds and used by split_dataset_by_thresholds
    all_results = {}
    
    # Run beam search for each threshold
    for uq_threshold in uq_thresholds:
        # Format threshold as string for use in field names
        # Use a format that's filesystem-safe and clear
        threshold_str = f"{uq_threshold:.6f}".rstrip('0').rstrip('.')
        if threshold_str == "":
            threshold_str = "0"
        threshold_str = threshold_str.replace('.', '_')  # Replace dots with underscores for field names
        
        # Run beam search with this threshold
        beam_results, total_tokens, prm_scores = _beam_search_conf(
            problems, config, slm, prm, llm, uq_threshold
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
            # total_tokens = SLM tokens (sum of completion_tokens) + LLM correction tokens
            beam_correction_ratios = []
            for b in beams:
                slm_tokens = sum(getattr(b, "completion_tokens", []))
                llm_corr_tokens_single = getattr(b, "llm_correction_tokens", 0)
                total_tokens = slm_tokens + llm_corr_tokens_single
                # Avoid division by zero
                if total_tokens > 0:
                    ratio = llm_corr_tokens_single / total_tokens
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
    
    return all_results


def split_dataset_by_thresholds(dataset, config: Config):
    """Split a dataset containing multiple threshold results into separate datasets.
    
    This function takes a dataset that was processed with smart_beam_search_conf_multi_threshold
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
        
        threshold_datasets[threshold_str] = Dataset.from_list(threshold_data)
    
    return threshold_datasets

