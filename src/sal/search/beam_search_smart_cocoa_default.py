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

        # scores = prm.score(prompts, completions)
        # agg_scores = [
        #     [aggregate_scores(s, config.agg_strategy) for s in score]
        #     for score in scores
        # ]
        
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
        cocoa_msp, cocoa_ppl, cocoa_entropy = calculate_cocoa_uq_scores(
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

        # 跳过去重逻辑，因为只有1个beam
        # if config.filter_duplicates:
        #     ...

        # 跳过pruning逻辑，因为只有1个beam
        # Get indices for top (config.n / config.beam_width) completions
        # top_indices = np.argsort(np.array(conf_agg_scores).flatten())[
        #     -(config.n // config.beam_width) :
        # ]
        # for idx, beam in enumerate(active_beams):
        #     if idx not in top_indices:
        #         beam.pruned = True

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

    # 跳过排序逻辑，因为只有1个beam
    # Filter completed beams for those with top config.n scores
    # if config.sort_completed:
    #     completed_beams = sorted(
    #         completed_beams,
    #         key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
    #         reverse=True,
    #     )[: config.n]
    # else:
    #     completed_beams = completed_beams[: config.n]

    # 跳过扩展逻辑，因为只有1个beam
    # if len(completed_beams) != config.n:
    #     ...

    # Print the problem information
    # for problem, info in problem_info.items():
    #     print(f"{{question: {problem}, generate_llm: {info['generate_llm']}, score_changed: {info['score_changed']}, text_changed: {info['text_changed']}}}")

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
    start_time = time.time()  # Record start time for SLM-only baseline
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
            templated_convs, lookahead, slm, sampling_params, config.beam_width, config.uq_sampling_temperature
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
    
    # Mark as no corrections done
    for beam in completed_beams:
        beam.smart_step = [-1]
        beam.gen_update = [("-1", "-1")]
        beam.llm_tokens = [-1]
        beam.completion_time = completion_time
        beam.llm_correction_tokens = 0  # No LLM corrections in SLM-only

    if prm is not None:
        prompts = [b.prompt for b in completed_beams]
        completions = [[b.current_text] for b in completed_beams]
        prm_scores = prm.score(prompts, completions)
    else:
        prm_scores = [[0.0] for _ in completed_beams]

    return completed_beams, total_tokens, prm_scores


def _beam_search_random_correction(
    batch_of_prompts, config: Config, slm: LLM, prm: PRM = None, llm: None = None, target_correction_count: int = 0
) -> tuple:
    """Random correction: randomly select steps to apply LLM correction.
    
    Strategy: 
    1. First, generate complete sequence with SLM only
    2. Record all steps  
    3. Randomly select target_correction_count steps (ensuring we have enough steps)
    4. For each selected step, re-generate with LLM from that point
    5. Reconstruct the sequence with selected steps replaced by LLM outputs
    """
    import random
    
    # Phase 1: Generate complete SLM-only sequence to know how many steps we have
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
    start_time = time.time()  # Record start time for random correction
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

    # Phase 1: SLM-only generation
    for iterate_idx in tqdm(
        range(config.num_iterations), desc="Random: SLM generation", disable=False
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
            templated_convs, lookahead, slm, sampling_params, config.beam_width, config.uq_sampling_temperature
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

    # Phase 2: Randomly select steps to correct
    for beam in completed_beams:
        num_steps = len(beam.history)
        
        if num_steps <= target_correction_count:
            selected_steps = list(range(num_steps))
        else:
            # Randomly select target_correction_count steps
            selected_steps = sorted(random.sample(range(num_steps), target_correction_count))
        
        # Phase 3: Re-generate selected steps with LLM
        for step_idx in selected_steps:
            # Reconstruct context up to (but not including) this step
            context_text = "".join(beam.history[:step_idx])
            
            conv = build_conv(beam.prompt, context_text, config.system_prompt)
            continue_final_message = step_idx > 0
            add_generation_prompt = step_idx == 0
            
            tokenizer_llm = AutoTokenizer.from_pretrained(config.model_path)
            if config.custom_chat_template is not None:
                tokenizer_llm.chat_template = config.custom_chat_template
            templated_conv = tokenizer_llm.apply_chat_template(
                [conv],
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                tokenize=False,
            )
            
            # Generate with LLM
            gen_results_llm = generate_k_steps_for_llm(
                tokenizer_llm, templated_conv, 0, llm, config, 1
            )
            gen_result_llm = gen_results_llm[0]
            
            slm_text = beam.history[step_idx]
            llm_text = gen_result_llm.next_texts[0]
            
            # Record correction info
            llm_token_count = len(tokenizer_llm.encode(llm_text))
            beam.smart_step.append(step_idx)
            beam.gen_update.append((slm_text, llm_text))
            beam.llm_tokens.append(llm_token_count)
            beam.llm_correction_tokens += llm_token_count
            total_tokens += llm_token_count
            beam.llm_corrections += 1
            
            # Replace the step in history
            beam.history[step_idx] = llm_text
        
        # Reconstruct final text
        beam.current_text = "".join(beam.history)

    # Record completion time for random correction
    end_time = time.time()
    completion_time = end_time - start_time
    for beam in completed_beams:
        beam.completion_time = completion_time

    if prm is not None:
        prompts = [b.prompt for b in completed_beams]
        completions = [[b.current_text] for b in completed_beams]
        prm_scores = prm.score(prompts, completions)
    else:
        prm_scores = [[0.0] for _ in completed_beams]

    return completed_beams, total_tokens, prm_scores


def smart_beam_search_cocoa_default(examples, config: Config, slm: LLM, prm: PRM = None, llm: None = None):
    problems = examples["problem"]
    
    # Get embedding model (cached globally to avoid repeated loading across samples)
    embedding_model = get_embedding_model()
    
    # 1. UQ-guided correction (original)
    beam_results_uq, total_tokens_uq, prm_scores_uq = _beam_search(
        problems, config, slm, prm, llm, embedding_model
    )
    
    # Get correction counts from UQ-guided results
    grouped_results_uq = defaultdict(list)
    for results in beam_results_uq:
        grouped_results_uq[results.prompt].append(results)
    
    correction_counts = []
    for p in problems:
        beams = grouped_results_uq[p]
        count = sum(getattr(b, "llm_corrections", 0) for b in beams)
        correction_counts.append(count)
    
    # Run baselines based on individual flags
    run_slm_baseline = getattr(config, 'run_slm_baseline', True)
    run_random_baseline = getattr(config, 'run_random_baseline', True)
    
    # 2. SLM-only baseline
    if run_slm_baseline:
        beam_results_slm, total_tokens_slm, prm_scores_slm = _beam_search_slm_only(
            problems, config, slm, prm
        )
        grouped_results_slm = defaultdict(list)
        for results in beam_results_slm:
            grouped_results_slm[results.prompt].append(results)
    else:
        grouped_results_slm = defaultdict(list)
    
    # 3. Random correction (using same correction counts as UQ-guided)
    if run_random_baseline:
        beam_results_random, total_tokens_random, prm_scores_random = _beam_search_random_correction(
            problems, config, slm, prm, llm, correction_counts[0]  # Assuming single problem
        )
        grouped_results_random = defaultdict(list)
        for results in beam_results_random:
            grouped_results_random[results.prompt].append(results)
    else:
        grouped_results_random = defaultdict(list)

    # Prepare output with all three methods
    results = {
        # UQ-guided results
        "completions": [],
        "pred": [],
        "scores": [],
        "correction_counts": [],
        "completion_times_uq": [],  # Timing for UQ-guided method
        "llm_correction_tokens_uq": [],  # LLM correction tokens for UQ-guided
        # SLM-only results
        "completions_slm": [],
        "pred_slm": [],
        "scores_slm": [],
        "completion_times_slm": [],  # Timing for SLM-only
        # Random correction results
        "completions_random": [],
        "pred_random": [],
        "scores_random": [],
        "correction_counts_random": [],
        "completion_times_random": [],  # Timing for random correction
        "llm_correction_tokens_random": [],  # LLM correction tokens for random
    }
    tokenizer = slm.get_tokenizer()

    for p in problems:
        # UQ-guided
        beams_uq = grouped_results_uq[p]
        completions_uq = [b.current_text for b in beams_uq]
        scores_uq = [b.all_scores for b in beams_uq]
        pred_uq = completions_uq[0] if len(completions_uq) > 0 else ""
        counts_uq = [getattr(b, "llm_corrections", 0) for b in beams_uq]
        times_uq = [getattr(b, "completion_time", 0.0) for b in beams_uq]
        tokens_uq = [getattr(b, "llm_correction_tokens", 0) for b in beams_uq]
        
        # Store UQ-guided results
        results["completions"].append(completions_uq)
        results["pred"].append(pred_uq)
        results["scores"].append(scores_uq)
        results["correction_counts"].append(counts_uq)
        results["completion_times_uq"].append(times_uq)
        results["llm_correction_tokens_uq"].append(tokens_uq)
        
        # SLM-only baseline
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
        
        # Random correction baseline
        if run_random_baseline:
            beams_random = grouped_results_random[p]
            completions_random = [b.current_text for b in beams_random]
            scores_random = [b.all_scores for b in beams_random]
            pred_random = completions_random[0] if len(completions_random) > 0 else ""
            counts_random = [getattr(b, "llm_corrections", 0) for b in beams_random]
            times_random = [getattr(b, "completion_time", 0.0) for b in beams_random]
            tokens_random = [getattr(b, "llm_correction_tokens", 0) for b in beams_random]
            
            results["completions_random"].append(completions_random)
            results["pred_random"].append(pred_random)
            results["scores_random"].append(scores_random)
            results["correction_counts_random"].append(counts_random)
            results["completion_times_random"].append(times_random)
            results["llm_correction_tokens_random"].append(tokens_random)
        else:
            results["completions_random"].append([])
            results["pred_random"].append("")
            results["scores_random"].append([])
            results["correction_counts_random"].append([])
            results["completion_times_random"].append([])
            results["llm_correction_tokens_random"].append([])
    
    # Embedding model is managed globally, no cleanup needed here
    
    return results
