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
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

from sal.config import Config
from sal.models.reward_models import PRM

from .utils import Beam, build_conv, generate_k_steps, last, generate_k_steps_for_llm

logger = logging.getLogger()
from sal.utils.score import aggregate_scores

# UHead related imports
from typing import List, Tuple
from lm_polygraph.stat_calculators.extract_claims import Claim
from lm_polygraph.model_adapters import WhiteboxModelBasic
from luh.calculator_infer_luh import CalculatorInferLuh
from luh.calculator_apply_uq_head import CalculatorApplyUQHead
from luh.luh_claim_estimator_dummy import LuhClaimEstimatorDummy

class UHeadScorer:
    """Scorer using uncertainty head to score steps, similar to PRM interface."""
    
    def __init__(
        self,
        llm: AutoModelForCausalLM,
        uhead,
        tokenizer: AutoTokenizer,
        config: Config,
        device: str = "cuda",
    ):
        self.llm = llm
        self.uhead = uhead
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Initialize calculators
        self.calc_infer_llm = CalculatorInferLuh(
            self.uhead,
            tokenize=False,
            args_generate={},
            device=device,
            generations_cache_dir="",
            predict_token_uncertainties=False,
        )
        
        self.calc_apply_uhead = CalculatorApplyUQHead(self.uhead)
        self.estimator = LuhClaimEstimatorDummy()
        
        # Create model adapter
        self.model_adapter = WhiteboxModelBasic(
            model=self.llm,
            tokenizer=self.tokenizer,
            tokenizer_args={
                "add_special_tokens": False,
                "return_tensors": "pt",
                "padding": True,
                "truncation": True,
            },
            model_type="CausalLM",
        )
        self.model_adapter.model_path = getattr(config, "model_path", "debug")
    
    def extract_steps(self, text: str) -> List[str]:
        """Extract steps from text, splitting by \n\n."""
        if not text:
            return []
        # Split by double newline
        steps = text.split("\n\n")
        # Filter out empty steps
        steps = [s.strip() for s in steps if s.strip()]
        return steps
    
    def step_to_claim(
        self,
        step_text: str,
        full_tokens: List[int],
        context_length: int,
        tokenizer: AutoTokenizer,
        max_seq_len: int,
    ) -> Claim:
        """
        Convert a step text to a Claim object with token alignment.
        
        Args:
            step_text: The text of the step
            full_tokens: The actual tokenized sequence (from tokenizer)
            context_length: Length of the context (prompt)
            tokenizer: Tokenizer instance
            max_seq_len: Maximum sequence length (to ensure bounds)
        """
        # Tokenize step
        step_tokens = tokenizer.encode(step_text, add_special_tokens=False)
        
        # Find token alignment in the actual tokenized sequence
        aligned_token_ids = []
        search_start = context_length
        search_end = min(len(full_tokens), max_seq_len)
        
        # Try to find step tokens in the full sequence
        for i in range(search_start, search_end - len(step_tokens) + 1):
            window = full_tokens[i : i + len(step_tokens)]
            if window == step_tokens:
                aligned_token_ids = list(range(i, i + len(step_tokens)))
                break
        
        # If not found, use heuristic: try to find a partial match
        if not aligned_token_ids:
            # Try to find at least the first few tokens
            for match_len in range(len(step_tokens), 0, -1):
                for i in range(search_start, search_end - match_len + 1):
                    window = full_tokens[i : i + match_len]
                    step_prefix = step_tokens[:match_len]
                    if window == step_prefix:
                        aligned_token_ids = list(range(i, i + match_len))
                        break
                if aligned_token_ids:
                    break
        
        # aligned_token_ids should be relative to generated tokens (0-indexed)
        # Subtract context_length to make them relative to generated part
        aligned_token_ids_relative = [tid - context_length for tid in aligned_token_ids if tid >= context_length]
        
        # Ensure all indices are valid (non-negative and within bounds)
        # The maximum valid index is min(len(full_tokens), max_seq_len) - context_length - 1
        max_valid_idx = min(search_end - context_length - 1, max_seq_len - context_length - 1)
        if max_valid_idx < 0:
            max_valid_idx = 0
        aligned_token_ids_relative = [tid for tid in aligned_token_ids_relative if 0 <= tid <= max_valid_idx]
        
        # If no valid tokens found, use a fallback (just use the first few generated tokens)
        if not aligned_token_ids_relative:
            # Fallback: use first few tokens of generated part
            fallback_len = min(len(step_tokens), max_valid_idx + 1)
            if fallback_len > 0:
                aligned_token_ids_relative = list(range(fallback_len))
        
        return Claim(
            claim_text=step_text,
            sentence=step_text,
            aligned_token_ids=aligned_token_ids_relative,
        )
    
    def score(
        self, questions: List[str], outputs: List[List[str]], batch_size: int = 8
    ) -> List[List[float]]:
        """
        Score outputs using uncertainty head, similar to PRM.score interface.
        """
        all_scores = []
        
        for question, output_list in zip(questions, outputs):
            question_scores = []
            
            for output in output_list:
                # Prepare full text
                prompt = self.config.system_prompt + "\n" + question + "\n"
                full_text = prompt + output
                
                # Prepare inputs for inference (tokenize first to get actual sequence)
                inputs = self.tokenizer(
                    full_text, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)
                
                # Get actual tokenized sequence
                input_ids = inputs["input_ids"][0].cpu().tolist()
                actual_seq_len = inputs["attention_mask"].shape[1]
                
                # Tokenize prompt separately to get context length
                prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
                context_length = len(prompt_tokens)
                # Adjust context_length if it exceeds actual sequence length
                context_length = min(context_length, actual_seq_len)
                
                # Extract steps from output
                steps = self.extract_steps(output)
                
                if not steps:
                    question_scores.append([])
                    continue
                
                # Convert steps to claims using actual tokenized sequence
                claims = []
                for step in steps:
                    claim = self.step_to_claim(
                        step, input_ids, context_length, self.tokenizer, actual_seq_len
                    )
                    claims.append(claim)
                
                # Get hidden states
                with torch.no_grad():
                    model_outputs = self.llm(
                        **inputs,
                        output_hidden_states=True,
                        output_attentions=self.uhead.output_attentions,
                    )
                
                # Prepare dependencies dict
                deps = {}
                generated_tokens = input_ids[context_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                deps["input_texts"] = [full_text]
                deps["input_tokens"] = [input_ids]
                deps["greedy_texts"] = [generated_text]
                deps["greedy_tokens"] = [generated_tokens]
                
                # Prepare batch
                batch = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "context_lenghts": torch.tensor([context_length]),
                }
                
                # Get uhead features
                with torch.no_grad():
                    # Create an object that supports both dict access and attribute access
                    # feature_extractor uses llm_outputs["logits"] (dict) and llm_outputs.context_lengths (attr)
                    class ModelOutputsWrapper:
                        def __init__(self, logits, context_lengths, hidden_states=None, attentions=None):
                            self.logits = logits
                            self.context_lengths = context_lengths
                            self.hidden_states = hidden_states
                            self.attentions = attentions
                        
                        def __getitem__(self, key):
                            # Support dict-style access
                            if key == "logits":
                                return self.logits
                            elif key == "context_lengths":
                                return self.context_lengths
                            elif key == "hidden_states":
                                return self.hidden_states
                            elif key == "attentions":
                                return self.attentions
                            else:
                                raise KeyError(f"Key {key} not found")
                    
                    model_outputs_wrapper = ModelOutputsWrapper(
                        logits=model_outputs.logits,
                        context_lengths=torch.tensor([context_length]),
                        hidden_states=model_outputs.hidden_states if hasattr(model_outputs, "hidden_states") else None,
                        attentions=model_outputs.attentions if hasattr(model_outputs, "attentions") else None,
                    )
                    
                    uhead_features = self.uhead.feature_extractor(batch, model_outputs_wrapper)
                
                deps["uhead_features"] = uhead_features
                deps["llm_inputs"] = batch
                deps["full_attention_mask"] = inputs["attention_mask"]
                deps["claims"] = [claims]
                
                # Calculate uncertainty scores
                uncertainty_deps = self.calc_apply_uhead(
                    deps, [full_text], self.model_adapter, max_new_tokens=0
                )
                
                # Get final uncertainty scores
                uncertainty_scores = self.estimator(uncertainty_deps)
                
                # Extract scores for this output
                if uncertainty_scores and len(uncertainty_scores) > 0:
                    step_scores = uncertainty_scores[0]
                    if len(step_scores) < len(steps):
                        step_scores = step_scores + [step_scores[-1] if step_scores else 0.0] * (
                            len(steps) - len(step_scores)
                        )
                    elif len(step_scores) > len(steps):
                        step_scores = step_scores[: len(steps)]
                else:
                    step_scores = [0.0] * len(steps)
                
                question_scores.append(step_scores)
            
            all_scores.append(question_scores)
        
        return all_scores


def _beam_search(batch_of_prompts, config: Config, slm: LLM, uhead_scorer: UHeadScorer, llm: None) -> tuple[list[Beam], int]:
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
    )
    
    beams: list[Beam] = []
    for prompt in batch_of_prompts:
        for i in range(config.n):
            beams.append(
                Beam(
                    prompt=prompt,
                    index=i,
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
                    prm_update=[],
                    gen_update=[],
                    llm_tokens=[],
                )
            )

    completed_beams: list[Beam] = []
    total_tokens = 0
    smart_done = False
    
    for iterate_idx in tqdm(range(config.num_iterations), desc="Beam search iterations (UHead)"):
        if iterate_idx == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        # Duplicate active beams to ensure that we have config.n beams per iteration
        if len(active_beams) != config.n:
            repeats = (config.n // len(active_beams)) + 1
            logger.debug(
                f"Extending active_beams with {repeats} repetitions to reach size {config.n}"
            )
            extended_active_beams = [
                copy.deepcopy(b) for b in (active_beams * repeats)[: config.n]
            ]
            active_beams = extended_active_beams
            if len(active_beams) != config.n:
                raise ValueError(
                    f"Expected {config.n} active beams, but got {len(active_beams)}"
                )

        if iterate_idx == config.num_iterations - 1:
            # Last iteration, generate to EOS (remove stop conditions)
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                stop=[],  # Remove stop conditions to ensure generation completes
                n=1,
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
        gen_results = generate_k_steps(
            templated_convs, lookahead, slm, sampling_params, 1
        )
        
        prev_active_beams = copy.deepcopy(active_beams)

        # copy the active beams to regenerate the beams with llm
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
            elif iterate_idx == config.num_iterations - 1:
                # Last iteration: force completion even if stop condition was triggered
                beam.completed = True
                if beam.stop_reasons[0] not in ["EOS", "length"]:
                    # Mark as completed due to reaching max iterations
                    beam.stop_reasons = ["max_iterations"]
                completed_beams.append(beam)
            prompts.append(beam.prompt)
            completions.append([beam.current_text])

        scores = uhead_scorer.score(prompts, completions)

        agg_scores = [
            [aggregate_scores(s, config.agg_strategy) for s in score]
            for score in scores
        ]

        for beam, score in zip(active_beams, scores, strict=True):
            if score and len(score) > 0:
                beam.all_scores = score[0]
            else:
                beam.all_scores = []

        # Now filter active_beams and agg_scores for beams that are completed
        agg_scores = [
            agg_scores[i] for i, b in enumerate(active_beams) if not b.completed
        ]
        
        prev_active_beams = [b for idx, b in enumerate(prev_active_beams) if not active_beams[idx].completed]
        active_beams = [b for b in active_beams if not b.completed]

        # Early stopping if all beams are completed
        if len(active_beams) == 0:
            break
        if not config.sort_completed and len(completed_beams) >= config.n:
            break

        # Filter duplicate active beams
        if config.filter_duplicates:
            # Create a dictionary to filter duplicates and retain order
            unique_beam_dict = {}
            for i, b in enumerate(active_beams):
                if b.current_text not in unique_beam_dict:
                    unique_beam_dict[b.current_text] = (
                        i  # Map the unique text to its index
                    )
            active_beams = [active_beams[i] for i in unique_beam_dict.values()]
            prev_active_beams = [prev_active_beams[i] for i in unique_beam_dict.values()]
            agg_scores = [agg_scores[i] for i in unique_beam_dict.values()]

        # Get indices for top (config.n / config.beam_width) completions
        top_indices = np.argsort(np.array(agg_scores).flatten())[
            -(config.n // config.beam_width) :
        ]

        for idx, beam in enumerate(active_beams):
            if idx not in top_indices:
                beam.pruned = True
                
        # SMART beam search implementation       
        # # filter the pruned beams with low scores
        # active_beams = [b for b in active_beams if not b.pruned]
        # agg_scores = [agg_scores[idx] for idx in top_indices]
        
        re_indices = [top_idx for top_idx in top_indices if agg_scores[top_idx][0] > config.threshold]  # uhead returns uncertainty scores (higher = more uncertain)
        if len(re_indices) == 0:
            continue
        
        smart_done = True
        re_beams = [prev_active_beams[idx] for idx in re_indices]          
        
        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in re_beams
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
        # On last iteration, disable stop criteria to ensure generation completes
        use_stop_criteria = iterate_idx != config.num_iterations - 1
        gen_results = generate_k_steps_for_llm(
            tokenizer, templated_convs, lookahead, llm, config, 1, use_stop_criteria=use_stop_criteria
        )

        reprompts, recompletions = [], []
        for beam, gen_result in zip(re_beams, gen_results, strict=True):
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
            elif iterate_idx == config.num_iterations - 1:
                # Last iteration: force completion even if stop condition was triggered
                beam.completed = True
                if beam.stop_reasons[0] not in ["EOS", "length"]:
                    # Mark as completed due to reaching max iterations
                    beam.stop_reasons = ["max_iterations"]
                completed_beams.append(beam)
            reprompts.append(beam.prompt)
            recompletions.append([beam.current_text])

        re_scores = uhead_scorer.score(reprompts, recompletions)
        reagg_scores = [
            [aggregate_scores(s, config.agg_strategy) for s in score]
            for score in re_scores
        ]
        
        for beam, score in zip(re_beams, re_scores, strict=True):
            if score and len(score) > 0:
                beam.all_scores = score[0]
            else:
                beam.all_scores = []

        for i, (re_idx, beam) in enumerate(zip(re_indices, re_beams)):
            # log correction information
            beam.smart_step.append(iterate_idx)
            beam.gen_update.append((active_beams[re_idx].next_texts[0], beam.next_texts[0]))
            old_agg = agg_scores[re_idx][0] if re_idx < len(agg_scores) else 0.0
            new_agg = reagg_scores[i][0] if i < len(reagg_scores) else 0.0
            beam.prm_update.append((old_agg, new_agg))
            beam.llm_tokens.append(len(tokenizer.encode(beam.next_texts[0])))
            total_tokens += len(tokenizer.encode(beam.next_texts[0]))
            active_beams[re_idx] = beam
    
    # After all iterations, mark any remaining active beams as completed
    # This ensures we always have completed beams, especially for n=1 case
    for beam in active_beams:
        if not beam.completed:
            beam.completed = True
            if beam.stop_reasons is None or len(beam.stop_reasons) == 0:
                beam.stop_reasons = ["max_iterations"]
            elif beam.stop_reasons[0] not in ["EOS", "length"]:
                beam.stop_reasons = ["max_iterations"]
            completed_beams.append(beam)
    
    # Filter completed beams for those with top config.n scores
    if config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: config.n]
    else:
        completed_beams = completed_beams[: config.n]

    # Ensure we have exactly config.n beams (duplicate if needed)
    # Note: completed_beams should never be empty because:
    # 1. In the last iteration, all active_beams are force-completed (line 150-156)
    # 2. After the loop, any remaining active_beams are force-completed (line 287-294)
    if len(completed_beams) < config.n:
        # If we don't have enough completed_beams, duplicate until we reach config.n
        repeats = (config.n // len(completed_beams)) + 1
        logger.debug(
            f"Extending completed_beams from {len(completed_beams)} to {config.n} with {repeats} repetitions"
        )
        extended_completed_beams = [
            copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]
        ]
        completed_beams = extended_completed_beams

    # Print the problem information
    # for problem, info in problem_info.items():
    #     print(f"{{question: {problem}, generate_llm: {info['generate_llm']}, score_changed: {info['score_changed']}, text_changed: {info['text_changed']}}}")

            
    for beam in completed_beams:
        if len(beam.smart_step) == 0:
            beam.smart_step = [-1]
            beam.prm_update = [(-1.0, -1.0)]
            beam.gen_update = [('-1', '-1')]
            beam.llm_tokens = [-1]
    
    return completed_beams, total_tokens


def smart_beam_search(examples, config: Config, slm: LLM, uhead_scorer: UHeadScorer, llm: None):
    problems = examples["problem"]
    beam_results, total_tokens = _beam_search(problems, config, slm, uhead_scorer, llm)

    # Group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "pred": [], "scores": [], "llm_tokens": [], "prm_update": [], "smart_step": [], "total_tokens": []}
    tokenizer = slm.get_tokenizer()

    for p in problems:
        beams = grouped_results[p]
        completions = [b.current_text for b in beams]
        scores = [b.all_scores for b in beams]
        pred = completions[np.argmax([
            aggregate_scores(b.all_scores, config.agg_strategy) for b in beams
        ])]
        llm_tokens = [b.llm_tokens for b in beams]
        prm_updates = [getattr(b, "prm_update", []) for b in beams]
        smart_steps = [getattr(b, "smart_step", []) for b in beams]
        # Calculate total tokens for each beam: sum of completion_tokens (draft model) + sum of llm_tokens (corrections)
        total_tokens_list = [sum(getattr(b, "completion_tokens", [])) + sum(getattr(b, "llm_tokens", [])) for b in beams]
        results["completions"].append(completions)
        results["pred"].append(pred)
        results["scores"].append(scores)
        results["llm_tokens"].append(llm_tokens)
        results["total_tokens"].append(total_tokens_list)
        results["smart_step"].append(smart_steps)
        results["prm_update"].append(prm_updates)
    return results