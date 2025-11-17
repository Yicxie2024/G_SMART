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
import time
from collections import defaultdict

from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM
from sal.search.utils import Beam, build_conv

logger = logging.getLogger()


def _beam_search_slm_only_single_pass(
    batch_of_prompts, config: Config, slm: LLM, prm: PRM = None
) -> tuple:
    """SLM-only baseline with single-pass generation: generate everything in one go.
    
    This is faster than iterative generation but may produce slightly different results.
    """
    # Set up sampling parameters for single-pass generation
    # Remove stop token to generate until EOS or max_tokens
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        n=1,
        logprobs=True,
    )
    
    beams: list[Beam] = []
    start_time = time.time()
    
    # Build conversations for all prompts
    convs = []
    for prompt in batch_of_prompts:
        conv = build_conv(prompt, "", config.system_prompt)
        convs.append(conv)
        
        # Initialize beam
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
    
    # Apply chat template
    tokenizer = slm.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template
    templated_convs = tokenizer.apply_chat_template(
        convs,
        add_generation_prompt=True,
        tokenize=False,
    )
    
    # Generate in one pass
    outputs = slm.generate(templated_convs, sampling_params, use_tqdm=False)
    
    completed_beams: list[Beam] = []
    total_tokens = 0
    
    # Process results
    for beam, output in zip(beams, outputs):
        # Get generated text
        generated_text = output.outputs[0].text
        beam.current_text = generated_text
        beam.history = [generated_text]  # Single pass, so history is just the full text
        
        # Get token information
        output_obj = output.outputs[0]
        if hasattr(output_obj, 'token_ids') and output_obj.token_ids:
            token_ids = output_obj.token_ids
            num_tokens = len(token_ids)
        else:
            # Fallback: estimate tokens from text
            token_ids = tokenizer.encode(generated_text, add_special_tokens=False)
            num_tokens = len(token_ids)
        beam.completion_tokens = [num_tokens]
        total_tokens += num_tokens
        
        # Get stop reason
        stop_reason = output_obj.finish_reason
        if stop_reason == "stop":
            beam.stop_reasons = ["EOS"]
        elif stop_reason == "length":
            beam.stop_reasons = ["length"]
        else:
            beam.stop_reasons = [stop_reason] if stop_reason else ["EOS"]
        
        # Check if completed
        beam.completed = True  # Single pass always completes
        
        # Get logprobs for scores (if available)
        # vLLM returns logprobs as a list of dicts, where each dict contains token_id -> LogprobInfo
        # We need to match the actual generated token_ids with their logprobs
        if hasattr(output_obj, 'logprobs') and output_obj.logprobs and token_ids:
            logprobs_list = output_obj.logprobs
            scores = []
            # Match each generated token with its logprob
            for i, token_id in enumerate(token_ids):
                if i < len(logprobs_list) and logprobs_list[i]:
                    token_logprobs = logprobs_list[i]
                    if isinstance(token_logprobs, dict):
                        # Get the logprob for this specific token_id
                        if token_id in token_logprobs:
                            logprob_info = token_logprobs[token_id]
                            if hasattr(logprob_info, 'logprob'):
                                scores.append(logprob_info.logprob)
                            elif isinstance(logprob_info, dict):
                                scores.append(logprob_info.get('logprob', 0.0))
                            else:
                                scores.append(0.0)
                        else:
                            # Token not found in logprobs, use 0.0
                            scores.append(0.0)
                    else:
                        scores.append(0.0)
                else:
                    scores.append(0.0)
            
            if scores:
                beam.all_scores = scores
            else:
                # Fallback: use placeholder scores
                beam.all_scores = [0.0] * max(1, num_tokens)
        else:
            # No logprobs available, use placeholder
            # For single-pass generation, we create one score per token
            beam.all_scores = [0.0] * max(1, num_tokens)
        
        completed_beams.append(beam)
    
    # Record completion time
    end_time = time.time()
    completion_time = end_time - start_time
    for beam in completed_beams:
        beam.smart_step = [-1]
        beam.gen_update = [("-1", "-1")]
        beam.llm_tokens = [-1]
        beam.completion_time = completion_time
        beam.llm_correction_tokens = 0
    
    # Calculate PRM scores if available
    if prm is not None:
        prompts = [b.prompt for b in completed_beams]
        completions = [[b.current_text] for b in completed_beams]
        prm_scores = prm.score(prompts, completions)
    else:
        prm_scores = None
    
    return completed_beams, total_tokens, prm_scores


def smart_beam_search_slm_only(examples, config: Config, slm: LLM, prm: PRM = None):
    """SLM-only baseline: no corrections at all.
    
    This is a standalone function that only runs SLM-only baseline,
    without running UQ-guided method first (unlike smart_beam_search_conf).
    This saves computation resources when you only need SLM-only baseline.
    
    Uses single-pass generation (one-shot) instead of iterative generation
    for better performance, while maintaining output field alignment.
    
    Returns results with aligned fields matching other methods:
    - completions
    - pred
    - scores
    - correction_counts (always 0 for SLM-only)
    - completion_times_uq
    - llm_correction_tokens_uq (always 0 for SLM-only)
    - smart_step (always [-1] for SLM-only)
    - total_tokens
    - correction_token_ratio (always 0.0 for SLM-only)
    """
    problems = examples["problem"]
    
    # Run SLM-only baseline with single-pass generation
    beam_results_slm, total_tokens_slm, prm_scores_slm = _beam_search_slm_only_single_pass(
        problems, config, slm, prm
    )
    grouped_results_slm = defaultdict(list)
    for results in beam_results_slm:
        grouped_results_slm[results.prompt].append(results)
    
    # Collect results with aligned fields
    results = {
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
        beams_slm = grouped_results_slm[p]
        completions_slm = [b.current_text for b in beams_slm]
        scores_slm = [b.all_scores for b in beams_slm]
        pred_slm = completions_slm[0] if len(completions_slm) > 0 else ""
        times_slm = [getattr(b, "completion_time", 0.0) for b in beams_slm]
        
        # SLM-only has no corrections, so correction_counts is always 0
        counts_slm = [0 for _ in beams_slm]
        # SLM-only has no LLM correction tokens
        tokens_slm = [0 for _ in beams_slm]
        # SLM-only has no smart steps (all are -1)
        smart_steps_slm = [getattr(b, "smart_step", [-1]) for b in beams_slm]
        
        # Calculate total_tokens and correction_token_ratio for each beam
        beam_total_tokens = []
        beam_correction_ratios = []
        for beam in beams_slm:
            # Calculate total tokens for this beam: only SLM tokens (no LLM correction)
            slm_tokens = sum(getattr(beam, "completion_tokens", []))
            total_tokens_beam = slm_tokens
            beam_total_tokens.append(total_tokens_beam)
            
            # Correction token ratio is always 0.0 for SLM-only (no corrections)
            beam_correction_ratios.append(0.0)
        
        results["completions"].append(completions_slm)
        results["pred"].append(pred_slm)
        results["scores"].append(scores_slm)
        results["correction_counts"].append(counts_slm)
        results["completion_times_uq"].append(times_slm)
        results["llm_correction_tokens_uq"].append(tokens_slm)
        results["smart_step"].append(smart_steps_slm)
        results["total_tokens"].append(beam_total_tokens)
        results["correction_token_ratio"].append(beam_correction_ratios)
    
    return results

