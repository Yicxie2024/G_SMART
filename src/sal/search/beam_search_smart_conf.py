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
    calculate_token_sar_score
)

from transformers import AutoTokenizer


def _beam_search(
    batch_of_prompts, config: Config, slm: LLM, prm: PRM, llm: None, crossencoder=None
) -> tuple:
    # Get special tokens for TokenSAR if needed
    special_tokens = None
    if config.score_method == "token_sar" and crossencoder is not None:
        tokenizer_for_special = slm.get_tokenizer()
        special_tokens = list(tokenizer_for_special.added_tokens_decoder.keys())
    
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

        # SMART single-beam correction: if confidence below threshold, ask llm to correct
        if config.score_method == "conf":
            need_correction = conf_agg_scores and conf_agg_scores[0][0] < config.uq_threshold
        elif config.score_method == "perplexity":
            need_correction = conf_agg_scores and conf_agg_scores[0][0] > config.uq_threshold
        elif config.score_method == "top2_margin":
            # For top2 margin, lower values indicate more uncertainty, so correct if below threshold
            need_correction = conf_agg_scores and conf_agg_scores[0][0] < config.uq_threshold
        elif config.score_method == "msp":
            # For MSP, higher values indicate more uncertainty, so correct if above threshold
            need_correction = conf_agg_scores and conf_agg_scores[0][0] < config.uq_threshold
        elif config.score_method == "token_entropy":
            # For Token Entropy, higher values indicate more uncertainty, so correct if above threshold
            need_correction = conf_agg_scores and conf_agg_scores[0][0] > config.uq_threshold
        elif config.score_method == "token_sar":
            # For TokenSAR, higher values indicate more uncertainty, so correct if above threshold
            need_correction = conf_agg_scores and conf_agg_scores[0][0] > config.uq_threshold
        else:
            need_correction = False
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


def _beam_search_llm_only(
    batch_of_prompts, config: Config, llm, prm: PRM = None
) -> tuple:
    """LLM-only baseline: use LLM to generate every step (no SLM)."""

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

    for iterate_idx in tqdm(
        range(config.num_iterations), desc="LLM-only generation", disable=False
    ):
        if iterate_idx == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]
        
        # Build conversations and generate with LLM
        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in active_beams
        ]
        continue_final_message = iterate_idx > 0
        add_generation_prompt = iterate_idx == 0

        tokenizer_llm = AutoTokenizer.from_pretrained(config.model_path)
        if config.custom_chat_template is not None:
            tokenizer_llm.chat_template = config.custom_chat_template
        templated_convs = tokenizer_llm.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )

        lookahead = 0 if iterate_idx == config.num_iterations - 1 else config.lookahead
        gen_results_llm = generate_k_steps_for_llm(
            tokenizer_llm, templated_convs, lookahead, llm, config, 1
        )

        for beam, gen_result in zip(active_beams, gen_results_llm, strict=True):
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            beam.completion_tokens += gen_result.completion_tokens

            beam.current_text += beam.next_texts[0]
            beam.history.append(beam.next_texts[0])
            total_tokens += sum(gen_result.completion_tokens)

            # track tokens generated by LLM
            llm_token_count = len(tokenizer_llm.encode(beam.next_texts[0]))
            beam.llm_tokens.append(llm_token_count)
            beam.llm_correction_tokens += llm_token_count

            history_text = " ".join(beam.history)
            if len(tokenizer_llm.encode(history_text)) > 2048:
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

        active_beams = [b for b in active_beams if not b.completed]
        if len(active_beams) == 0:
            break

    # Record completion time for LLM-only baseline
    end_time = time.time()
    completion_time = end_time - start_time
    for beam in completed_beams:
        beam.completion_time = completion_time
        # Fill all_scores with placeholder values since LLM-only doesn't have logprobs
        # all_scores should be list[float], e.g., [0.0, 0.0, ...]
        beam.all_scores = [0.0] * len(beam.history) if len(beam.history) > 0 else [0.0]

    if prm is not None:
        prompts = [b.prompt for b in completed_beams]
        completions = [[b.current_text] for b in completed_beams]
        prm_scores = prm.score(prompts, completions)
    else:
        prm_scores = [[0.0] for _ in completed_beams]

    return completed_beams, total_tokens, prm_scores


def _beam_search_slm_only(
    batch_of_prompts, config: Config, slm: LLM, prm: PRM = None
) -> tuple:
    """SLM-only baseline: no corrections at all."""
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
        logprobs=True,
    )

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

    for iterate_idx in tqdm(
        range(config.num_iterations), desc="SLM-only generation", disable=False
    ):
        if iterate_idx == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

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

        active_beams = [b for b in active_beams if not b.completed]

        if len(active_beams) == 0:
            break

    # Record completion time for SLM-only baseline
    end_time = time.time()
    completion_time = end_time - start_time
    for beam in completed_beams:
        beam.smart_step = [-1]
        beam.gen_update = [("-1", "-1")]
        beam.llm_tokens = [-1]
        beam.completion_time = completion_time
        beam.llm_correction_tokens = 0
        # Fill all_scores with placeholder values since SLM-only doesn't calculate scores
        beam.all_scores = [0.0] * len(beam.history) if len(beam.history) > 0 else [0.0]

    if prm is not None:
        prompts = [b.prompt for b in completed_beams]
        completions = [[b.current_text] for b in completed_beams]
        prm_scores = prm.score(prompts, completions)
    else:
        prm_scores = [[0.0] for _ in completed_beams]

    return completed_beams, total_tokens, prm_scores


def _beam_search_random_correction(
    batch_of_prompts, config: Config, slm: LLM, prm: PRM = None, llm: None = None,
    target_correction_counts_per_prompt: list[int] = None,
    actual_steps_per_prompt: list[int] = None,
) -> tuple:
    """Random correction: pre-select steps for LLM correction and apply during generation."""
    import random

    correction_steps_per_prompt = []
    if target_correction_counts_per_prompt is None:
        target_correction_counts_per_prompt = [0 for _ in batch_of_prompts]
    if actual_steps_per_prompt is None:
        actual_steps_per_prompt = [config.num_iterations for _ in batch_of_prompts]

    for i, prompt in enumerate(batch_of_prompts):
        max_possible_steps = max(0, int(actual_steps_per_prompt[i]))
        target_correction_count = int(target_correction_counts_per_prompt[i])
        if max_possible_steps <= 0 or target_correction_count <= 0:
            selected_steps = []
        elif max_possible_steps <= target_correction_count:
            selected_steps = list(range(max_possible_steps))
        else:
            selected_steps = sorted(random.sample(range(max_possible_steps), target_correction_count))
        correction_steps_per_prompt.append(selected_steps)

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
        logprobs=True,
    )

    beams: list[Beam] = []
    start_time = time.time()
    for i, prompt in enumerate(batch_of_prompts):
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
        beam.pre_selected_correction_steps = correction_steps_per_prompt[i]
        beam.early_stop_unused_corrections = False
        beams.append(beam)

    completed_beams: list[Beam] = []
    total_tokens = 0

    for iterate_idx in tqdm(
        range(config.num_iterations), desc="Random: Pre-selected correction", disable=False
    ):
        if iterate_idx == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        if iterate_idx == config.num_iterations - 1:
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                n=1,
                logprobs=True,
            )

        for beam in active_beams:
            if iterate_idx in beam.pre_selected_correction_steps:
                conv = build_conv(beam.prompt, beam.current_text, config.system_prompt)
                continue_final_message = iterate_idx > 0
                add_generation_prompt = iterate_idx == 0

                tokenizer_llm = AutoTokenizer.from_pretrained(config.model_path)
                if config.custom_chat_template is not None:
                    tokenizer_llm.chat_template = config.custom_chat_template
                templated_conv = tokenizer_llm.apply_chat_template(
                    [conv],
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    tokenize=False,
                )

                gen_results_llm = generate_k_steps_for_llm(
                    tokenizer_llm, templated_conv, 0, llm, config, 1
                )
                gen_result_llm = gen_results_llm[0]

                beam.next_texts = gen_result_llm.next_texts
                beam.stop_reasons = gen_result_llm.stop_reasons
                beam.lookahead_texts = gen_result_llm.lookahead_texts
                beam.completion_tokens += gen_result_llm.completion_tokens

                beam.current_text += beam.next_texts[0]
                beam.history.append(beam.next_texts[0])
                total_tokens += sum(gen_result_llm.completion_tokens)

                llm_token_count = len(tokenizer_llm.encode(beam.next_texts[0]))
                beam.smart_step.append(iterate_idx)
                beam.gen_update.append(("", beam.next_texts[0]))
                beam.llm_tokens.append(llm_token_count)
                beam.llm_correction_tokens += llm_token_count
                beam.llm_corrections += 1

            else:
                conv = build_conv(beam.prompt, beam.current_text, config.system_prompt)
                continue_final_message = iterate_idx > 0
                add_generation_prompt = iterate_idx == 0

                tokenizer = slm.get_tokenizer()
                if config.custom_chat_template is not None:
                    tokenizer.chat_template = config.custom_chat_template
                templated_conv = tokenizer.apply_chat_template(
                    [conv],
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    tokenize=False,
                )

                lookahead = 0 if iterate_idx == config.num_iterations - 1 else config.lookahead
                gen_results, responses = generate_k_steps_with_responses(
                    templated_conv, lookahead, slm, sampling_params, 1
                )
                gen_result = gen_results[0]

                beam.next_texts = [gen_result.next_texts[0]]
                beam.stop_reasons = [gen_result.stop_reasons[0]]
                beam.lookahead_texts = [gen_result.lookahead_texts[0]]
                beam.completion_tokens += [gen_result.completion_tokens[0]]

                beam.current_text += beam.next_texts[0]
                beam.history.append(beam.next_texts[0])
                total_tokens += sum([gen_result.completion_tokens[0]])

        for beam in active_beams:
            history_text = " ".join(beam.history)
            tokenizer = slm.get_tokenizer()
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

        active_beams = [b for b in active_beams if not b.completed]

        if len(active_beams) == 0:
            break

    for beam in completed_beams:
        used_corrections = set(beam.smart_step)
        pre_selected = set(beam.pre_selected_correction_steps)
        unused_corrections = pre_selected - used_corrections
        if len(unused_corrections) > 0:
            beam.early_stop_unused_corrections = True
        else:
            beam.early_stop_unused_corrections = False

    end_time = time.time()
    completion_time = end_time - start_time
    for beam in completed_beams:
        beam.completion_time = completion_time
        # Fill all_scores with placeholder values since Random doesn't calculate scores
        beam.all_scores = [0.0] * len(beam.history) if len(beam.history) > 0 else [0.0]

    if prm is not None:
        prompts = [b.prompt for b in completed_beams]
        completions = [[b.current_text] for b in completed_beams]
        prm_scores = prm.score(prompts, completions)
    else:
        prm_scores = [[0.0] for _ in completed_beams]

    return completed_beams, total_tokens, prm_scores

def smart_beam_search_conf(examples, config: Config, slm: LLM, prm: PRM, llm: None, crossencoder=None):
    problems = examples["problem"]

    # 1) Original SMART confidence-guided
    beam_results_uq, total_tokens_uq, prm_scores_uq = _beam_search(
        problems, config, slm, prm, llm, crossencoder
    )
    grouped_results_uq = defaultdict(list)
    for results in beam_results_uq:
        grouped_results_uq[results.prompt].append(results)

    # Derive counts and steps for random baseline
    correction_counts = []
    actual_steps = []
    for p in problems:
        beams = grouped_results_uq[p]
        count = sum(getattr(b, "llm_corrections", 0) for b in beams)
        correction_counts.append(count)
        steps = sum(len(getattr(b, "history", [])) for b in beams)
        actual_steps.append(steps if steps > 0 else config.num_iterations)

    # Baseline flags with defaults
    run_slm_baseline = getattr(config, 'run_slm_baseline', True)
    run_random_baseline = getattr(config, 'run_random_baseline', True)
    run_llm_baseline = getattr(config, 'run_llm_baseline', True)

    # 2) SLM-only baseline
    if run_slm_baseline:
        beam_results_slm, total_tokens_slm, prm_scores_slm = _beam_search_slm_only(
            problems, config, slm, prm
        )
        grouped_results_slm = defaultdict(list)
        for results in beam_results_slm:
            grouped_results_slm[results.prompt].append(results)
    else:
        grouped_results_slm = defaultdict(list)

    # 3) Random correction baseline
    if run_random_baseline:
        beam_results_random, total_tokens_random, prm_scores_random = _beam_search_random_correction(
            problems, config, slm, prm, llm,
            target_correction_counts_per_prompt=correction_counts,
            actual_steps_per_prompt=actual_steps,
        )
        grouped_results_random = defaultdict(list)
        for results in beam_results_random:
            grouped_results_random[results.prompt].append(results)
    else:
        grouped_results_random = defaultdict(list)

    # 4) LLM-only baseline
    if run_llm_baseline:
        beam_results_llm, total_tokens_llm, prm_scores_llm = _beam_search_llm_only(
            problems, config, llm, prm
        )
        grouped_results_llm = defaultdict(list)
        for results in beam_results_llm:
            grouped_results_llm[results.prompt].append(results)
    else:
        grouped_results_llm = defaultdict(list)

    results = {
        # confidence-guided (original)
        "completions": [],
        "pred": [],
        "scores": [],
        "correction_counts": [],
        "completion_times_uq": [],
        "llm_correction_tokens_uq": [],
        "smart_step": [],
        # SLM-only
        "completions_slm": [],
        "pred_slm": [],
        "scores_slm": [],
        "completion_times_slm": [],
        # LLM-only
        "completions_llm": [],
        "pred_llm": [],
        "scores_llm": [],
        "completion_times_llm": [],
        "llm_tokens_llm": [],
        # Random
        "completions_random": [],
        "pred_random": [],
        "scores_random": [],
        "correction_counts_random": [],
        "correction_counts_random_preselected": [],
        "completion_times_random": [],
        "llm_correction_tokens_random": [],
        "early_stop_unused_corrections_random": [],
    }

    for p in problems:
        # UQ-guided results
        beams_uq = grouped_results_uq[p]
        completions_uq = [b.current_text for b in beams_uq]
        scores_uq = [b.all_scores for b in beams_uq]
        pred_uq = completions_uq[0] if len(completions_uq) > 0 else ""
        counts_uq = [getattr(b, "llm_corrections", 0) for b in beams_uq]
        times_uq = [getattr(b, "completion_time", 0.0) for b in beams_uq]
        tokens_uq = [getattr(b, "llm_correction_tokens", 0) for b in beams_uq]
        smart_steps_uq = [getattr(b, "smart_step", []) for b in beams_uq]

        results["completions"].append(completions_uq)
        results["pred"].append(pred_uq)
        results["scores"].append(scores_uq)
        results["correction_counts"].append(counts_uq)
        results["completion_times_uq"].append(times_uq)
        results["llm_correction_tokens_uq"].append(tokens_uq)
        results["smart_step"].append(smart_steps_uq)

        # SLM-only
        if run_slm_baseline:
            beams_slm = grouped_results_slm[p]
            completions_slm = [b.current_text for b in beams_slm]
            scores_slm = [b.all_scores for b in beams_slm]
            pred_slm = completions_slm[0] if len(completions_slm) > 0 else ""
            times_slm = [getattr(b, "completion_time", 0.0) for b in beams_slm]

            results["completions_slm"].append(completions_slm)
            results["pred_slm"].append(pred_slm)
            results["scores_slm"].append(scores_slm)
            results["completion_times_slm"].append(times_slm)
        else:
            results["completions_slm"].append([])
            results["pred_slm"].append("")
            results["scores_slm"].append([])
            results["completion_times_slm"].append([])

        # Random
        if run_random_baseline:
            beams_random = grouped_results_random[p]
            completions_random = [b.current_text for b in beams_random]
            scores_random = [b.all_scores for b in beams_random]
            pred_random = completions_random[0] if len(completions_random) > 0 else ""
            counts_random = [getattr(b, "llm_corrections", 0) for b in beams_random]
            counts_random_preselected = [len(getattr(b, "pre_selected_correction_steps", [])) for b in beams_random]
            times_random = [getattr(b, "completion_time", 0.0) for b in beams_random]
            tokens_random = [getattr(b, "llm_correction_tokens", 0) for b in beams_random]
            early_stop_flags = [getattr(b, "early_stop_unused_corrections", False) for b in beams_random]

            results["completions_random"].append(completions_random)
            results["pred_random"].append(pred_random)
            results["scores_random"].append(scores_random)
            results["correction_counts_random"].append(counts_random)
            results["correction_counts_random_preselected"].append(counts_random_preselected)
            results["completion_times_random"].append(times_random)
            results["llm_correction_tokens_random"].append(tokens_random)
            results["early_stop_unused_corrections_random"].append(early_stop_flags)
        else:
            results["completions_random"].append([])
            results["pred_random"].append("")
            results["scores_random"].append([])
            results["correction_counts_random"].append([])
            results["correction_counts_random_preselected"].append([])
            results["completion_times_random"].append([])
            results["llm_correction_tokens_random"].append([])
            results["early_stop_unused_corrections_random"].append([])

        # LLM-only
        if run_llm_baseline:
            beams_llm = grouped_results_llm[p]
            completions_llm = [b.current_text for b in beams_llm]
            scores_llm = [b.all_scores for b in beams_llm]
            pred_llm = completions_llm[0] if len(completions_llm) > 0 else ""
            times_llm = [getattr(b, "completion_time", 0.0) for b in beams_llm]
            llm_tokens = [getattr(b, "llm_correction_tokens", 0) for b in beams_llm]

            results["completions_llm"].append(completions_llm)
            results["pred_llm"].append(pred_llm)
            results["scores_llm"].append(scores_llm)
            results["completion_times_llm"].append(times_llm)
            results["llm_tokens_llm"].append(llm_tokens)
        else:
            results["completions_llm"].append([])
            results["pred_llm"].append("")
            results["scores_llm"].append([])
            results["completion_times_llm"].append([])
            results["llm_tokens_llm"].append([])

    return results


