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
    calculate_token_entropy_scores,
    calculate_token_similarity,
    calculate_token_sar_score,
    combine_token_sar_conf_margin,
)

from transformers import AutoTokenizer


def _beam_search(
    batch_of_prompts, config: Config, slm: LLM, prm: PRM, llm: None, crossencoder=None
) -> tuple:
    # Get special tokens for TokenSAR if needed
    special_tokens = None
    if config.score_method in {"token_sar", "token_sar_conf_margin"} and crossencoder is not None:
        tokenizer_for_special = slm.get_tokenizer()
        special_tokens = set(tokenizer_for_special.added_tokens_decoder.keys())
    
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
        beam = Beam(
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
        beams.append(beam)

    completed_beams: list[Beam] = []
    total_tokens = 0
    smart_done = False

    for iterate_idx in tqdm(
        range(config.num_iterations), desc="Beam search iterations"
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
        for idx, output in enumerate([o for r in responses for o in r.outputs]):
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
            elif config.score_method == "token_sar":
                # TokenSAR method: calculate token similarity and weighted uncertainty
                if crossencoder is not None and hasattr(output, 'token_ids'):
                    beam = active_beams[idx] if idx < len(active_beams) else active_beams[-1]
                    token_ids = output.token_ids
                    input_text = beam.prompt
                    
                    # Calculate token similarity
                    token_similarity = calculate_token_similarity(
                        token_ids,
                        input_text,
                        tokenizer,
                        crossencoder,
                        special_tokens
                    )
                    
                    # Calculate TokenSAR score
                    sar_score = calculate_token_sar_score(output.logprobs, token_similarity)
                    conf_scores.append([sar_score])
                else:
                    # Fallback to mean entropy if CrossEncoder not available
                    max_entropy, mean_entropy, entropies = calculate_token_entropy_scores([output.logprobs])
                    conf_scores.append([mean_entropy])
            elif config.score_method == "token_sar_conf_margin":
                if crossencoder is not None and hasattr(output, 'token_ids'):
                    beam = active_beams[idx] if idx < len(active_beams) else active_beams[-1]
                    token_ids = output.token_ids
                    input_text = beam.prompt

                    token_similarity = calculate_token_similarity(
                        token_ids,
                        input_text,
                        tokenizer,
                        crossencoder,
                        special_tokens,
                    )

                    combine_kwargs = getattr(config, "token_sar_conf_margin_params", None) or {}
                    combined = combine_token_sar_conf_margin(
                        [output.logprobs],
                        token_similarity,
                        skip_token_ids=special_tokens,
                        **combine_kwargs,
                    )
                    conf_scores.append([combined["final_score"]])
                    beam.extra_info = getattr(beam, "extra_info", []) + [combined]
                else:
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

        # ============================================================
        # HYBRID SMART CORRECTION: Two-stage decision
        # Stage 1: UQ low threshold (filter obvious bad steps)
        # Stage 2: PRM high threshold (catch overconfident errors)
        # ============================================================
        
        if not conf_agg_scores:
            continue
            
        uq_score = conf_agg_scores[0][0]
        
        # Stage 1: Check if UQ indicates obvious uncertainty (low threshold)
        # Lower threshold means we only catch obviously bad steps here
        uq_threshold_low = getattr(config, 'uq_threshold', config.uq_threshold)
        
        if config.score_method == "conf":
            fails_uq_check = uq_score < uq_threshold_low
        elif config.score_method == "perplexity":
            fails_uq_check = uq_score > uq_threshold_low
        elif config.score_method == "top2_margin":
            fails_uq_check = uq_score < uq_threshold_low
        elif config.score_method == "msp":
            fails_uq_check = uq_score < uq_threshold_low
        elif config.score_method == "token_entropy":
            fails_uq_check = uq_score > uq_threshold_low
        elif config.score_method == "token_sar":
            fails_uq_check = uq_score > uq_threshold_low
        elif config.score_method == "token_sar_conf_margin":
            fails_uq_check = uq_score > uq_threshold_low
        elif config.score_method == "hybrid_prm":
            # Hybrid method uses confidence score for Stage 1 UQ check
            # If UQ < threshold, directly correct (Stage 1)
            fails_uq_check = uq_score < uq_threshold_low
        else:
            fails_uq_check = False
        
        need_correction = False
        correction_reason = "accepted"
        
        if fails_uq_check:
            # Stage 1: Obvious uncertainty detected, directly correct with LLM
            need_correction = True
            correction_reason = "low_uq"
        else:
            # Stage 2: UQ looks OK, but check with PRM for overconfident errors
            # Use PRM to evaluate the quality of SLM's step
            if prm is not None:
                prm_threshold_high = getattr(config, 'prm_threshold', 0.5)
                
                # Score current step with PRM
                current_prompts = [active_beams[0].prompt]
                current_completions = [[active_beams[0].current_text]]
                prm_step_scores = prm.score(current_prompts, current_completions)
                # prm_step_scores[0][0] is a list of scores for all steps, take the last one for current step
                prm_score = prm_step_scores[0][0][-1] if prm_step_scores and len(prm_step_scores[0]) > 0 and len(prm_step_scores[0][0]) > 0 else 0.0
                print(f"PRM score (current step): {prm_score}")
                if prm_score <= prm_threshold_high:
                    # Stage 2: PRM detected quality issue despite high UQ confidence
                    need_correction = True
                    correction_reason = "low_prm_despite_high_uq"
                else:
                    # Both UQ and PRM checks passed
                    correction_reason = "passed_both"
            else:
                # No PRM available, accept based on UQ alone
                correction_reason = "passed_uq_only"
        
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
        # Copy tracking information from previous beam
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


def smart_beam_search_conf(examples, config: Config, slm: LLM, prm: PRM, llm: None, crossencoder=None):
    """
    Hybrid SMART Correction with two-stage decision:
    - Stage 1: UQ low threshold (filter obvious bad steps)
    - Stage 2: PRM high threshold (catch overconfident errors)
    """
    problems = examples["problem"]

    # Run hybrid SMART correction
    beam_results, total_tokens, prm_scores_final = _beam_search(
        problems, config, slm, prm, llm, crossencoder
    )
    
    # Group results by prompt
    grouped_results = defaultdict(list)
    for beam in beam_results:
        grouped_results[beam.prompt].append(beam)

    # Collect results
    results = {
        # Main results
        "completions": [],
        "pred": [],
        "scores": [],
        "correction_counts": [],
        "completion_times_uq": [],
        "llm_correction_tokens_uq": [],
        "smart_step": [],
        "total_tokens": [],
        "correction_token_ratio": [],
    }

    for p in problems:
        beams = grouped_results[p]
        if len(beams) == 0:
            # Handle empty beams case - ensure all fields have consistent types
            results["completions"].append([])
            results["pred"].append("")
            results["scores"].append([])
            results["correction_counts"].append([])
            results["completion_times_uq"].append([])
            results["llm_correction_tokens_uq"].append([])
            results["smart_step"].append([])
            results["total_tokens"].append([])
            results["correction_token_ratio"].append([])
            continue
            
        completions = [b.current_text for b in beams]
        scores = [b.all_scores for b in beams]
        pred = completions[0] if len(completions) > 0 else ""
        counts = [getattr(b, "llm_corrections", 0) for b in beams]
        times = [getattr(b, "completion_time", 0.0) for b in beams]
        tokens = [getattr(b, "llm_correction_tokens", 0) for b in beams]
        smart_steps = [getattr(b, "smart_step", []) for b in beams]
        
        # Calculate total_tokens and correction_token_ratio for each beam
        # Note: total_tokens is per-problem, but we need per-beam tokens
        # We'll track total tokens per beam by summing SLM tokens + LLM correction tokens
        beam_total_tokens = []
        beam_correction_ratios = []
        for beam in beams:
            # Calculate total tokens for this beam: SLM tokens + LLM correction tokens
            slm_tokens = sum(getattr(beam, "completion_tokens", []))
            llm_correction_tokens_beam = getattr(beam, "llm_correction_tokens", 0)
            total_tokens_beam = slm_tokens + llm_correction_tokens_beam
            beam_total_tokens.append(total_tokens_beam)
            
            # Calculate correction token ratio
            if total_tokens_beam > 0:
                correction_ratio = llm_correction_tokens_beam / total_tokens_beam
            else:
                correction_ratio = 0.0
            beam_correction_ratios.append(correction_ratio)

        results["completions"].append(completions)
        results["pred"].append(pred)
        results["scores"].append(scores)
        results["correction_counts"].append(counts)
        results["completion_times_uq"].append(times)
        results["llm_correction_tokens_uq"].append(tokens)
        results["smart_step"].append(smart_steps)
        results["total_tokens"].append(beam_total_tokens)
        results["correction_token_ratio"].append(beam_correction_ratios)

    return results


