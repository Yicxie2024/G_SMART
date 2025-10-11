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
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores, calculate_cocoa_uq_scores
from .utils import Beam, build_conv, generate_k_steps_with_responses, generate_k_steps_for_llm

logger = logging.getLogger()


def _cocoa_singletrack_search(
    batch_of_prompts, config: Config, slm: LLM, prm: PRM, llm: None
):
    sampling_params_step = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
        logprobs=True,
    )
    sampling_params_final = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        n=1,
        logprobs=True,
    )

    tokenizer_slm = slm.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer_slm.chat_template = config.custom_chat_template

    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def _detok(token_ids: List[int]) -> str:
        return tokenizer_slm.decode(token_ids, skip_special_tokens=True)

    def _embed_fn(texts: List[str]) -> np.ndarray:
        return sbert.encode(texts, convert_to_numpy=True, normalize_embeddings=False)

    completed_beams: List[Beam] = []
    total_tokens = 0

    for prompt in batch_of_prompts:
        current_text = ""
        step_cocoa_scores: List[float] = []     # (2) 记录每步的 CoCoA
        correction_count: int = 0               # (4) 记录纠错次数

        for iterate_idx in tqdm(range(config.num_iterations), desc="Singletrack iterations"):
            # 最后一轮依然生成一个 step（是否 run-to-EOS 由 sampling_params_final 控制）
            sp = sampling_params_final if iterate_idx == config.num_iterations - 1 else sampling_params_step
            lookahead = 0 if iterate_idx == config.num_iterations - 1 else config.lookahead
            add_generation_prompt = (iterate_idx == 0)
            continue_final_message = (iterate_idx > 0)

            conv = build_conv(prompt, current_text, config.system_prompt)
            templated_conv = tokenizer_slm.apply_chat_template(
                [conv],
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                tokenize=False,
            )

            m = int(config.beam_width)  # 本轮生成：1 个 greedy + (m-1) 个 samples
            gen_results, responses = generate_k_steps_with_responses(
                templated_conv, lookahead, slm, sp, m, config.uq_sampling_temperature
            )
            flat_outputs = [o for r in responses for o in r.outputs]
            assert len(flat_outputs) == m, f"expected {m} outputs, got {len(flat_outputs)}"

            greedy_out = flat_outputs[0]
            sample_outs = flat_outputs[1:]

            # (1)(2) 用你给的 CoCoA：greedy 放第一，其余 samples 在后
            beam_logprobs_list = [greedy_out.logprobs] + [o.logprobs for o in sample_outs]
            cocoa_msp, cocoa_ppl, cocoa_entropy = calculate_cocoa_uq_scores(
                beam_logprobs_list, detok=_detok, embed_fn=_embed_fn
            )
            if config.score_method == "cocoa_msp":
                cocoa_score = float(cocoa_msp)
            elif config.score_method == "cocoa_ppl":
                cocoa_score = float(cocoa_ppl)
            elif config.score_method == "cocoa_entropy":
                cocoa_score = float(cocoa_entropy)
            else:
                raise ValueError(f"Unknown score_method: {config.score_method}")

            # 记录该步的 CoCoA 分数
            step_cocoa_scores.append(cocoa_score)

            # —— 阈值判定：超阈值则纠错，并“替换”本步文本；否则使用 greedy 文本
            step_text = greedy_out.text
            step_tokens_used = len(greedy_out.token_ids)

            if (cocoa_score > config.uq_threshold) and (llm is not None):
                tokenizer_llm = AutoTokenizer.from_pretrained(config.model_path)
                if config.custom_chat_template is not None:
                    tokenizer_llm.chat_template = config.custom_chat_template

                conv_llm = build_conv(prompt, current_text, config.system_prompt)
                templated_conv_llm = tokenizer_llm.apply_chat_template(
                    [conv_llm],
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    tokenize=False,
                )
                gen_results_llm = generate_k_steps_for_llm(
                    tokenizer_llm, templated_conv_llm, lookahead, llm, config, 1
                )
                corrected_text = gen_results_llm[0].next_texts[0]

                step_text = corrected_text
                step_tokens_used = len(tokenizer_llm.encode(corrected_text))
                correction_count += 1  # (4) 纠错次数 +1

            # 累积上下文 & 记账
            current_text += step_text
            total_tokens += step_tokens_used
            # 如需计入 samples 的 token 成本，保留以下循环；否则可注释掉
            for o in sample_outs:
                total_tokens += len(o.token_ids)

            # 迭代到最后一轮就结束（此时 current_text 已包含最后一步的“被采用文本”）
            if iterate_idx == config.num_iterations - 1:
                break

        # (1) 最终只返回 greedy 的完整结果（单个 Beam）
        final_beam = Beam(
            prompt=prompt,
            index=0,
            current_text=current_text,
            next_texts=None,
            lookahead_texts=None,
            completion_tokens=[],
            stop_reasons=None,
            best_scores=[],
            all_scores=step_cocoa_scores,   # (2) 记录全程 CoCoA
            previous_text=None,
            pruned=False,
            history=[],
        )
        # 也把纠错次数存到对象上，便于调试/外部读取
        final_beam.llm_corrections = correction_count

        completed_beams.append(final_beam)

    # PRM：对每个最终（唯一）候选计算
    prompts = [b.prompt for b in completed_beams]
    completions = [[b.current_text] for b in completed_beams]
    prm_scores = prm.score(prompts, completions)

    return completed_beams, total_tokens, prm_scores


def smart_beam_search_cocoa_singletrack(examples, config: Config, slm: LLM, prm: PRM, llm: None):
    """
    维持原有返回结构：
      - completions: List[List[str]]，每个 prompt 里只有 1 个字符串（greedy 最终结果）
      - pred: List[str]，直接等于该唯一结果
      - scores: List[List[float]]，存入“最终 beam 的 all_scores 的均值”，包一层 list
      - correction_counts: List[List[int]]，存入纠错总次数，包一层 list
    """
    problems = examples["problem"]
    beam_results, total_tokens, prm_scores = _cocoa_singletrack_search(
        problems, config, slm, prm, llm
    )

    # 每个 prompt 只有一个 Beam
    grouped = defaultdict(list)
    for b in beam_results:
        grouped[b.prompt].append(b)

    results = {"completions": [], "pred": [], "scores": [], "correction_counts": []}

    for p in problems:
        beam = grouped[p][0]  # 唯一
        completion = beam.current_text

        # (3) 分数：all_scores 的均值；若为空则给 0.0
        mean_score = float(np.mean(beam.all_scores)) if beam.all_scores else 0.0

        # (4) 纠错次数：从对象读取（默认 0）
        corr_cnt = int(getattr(beam, "llm_corrections", 0))

        results["completions"].append([completion])   # 仍保持 list 形状
        results["pred"].append(completion)            # 直接取该唯一结果
        results["scores"].append([mean_score])        # 每个 prompt 一个 list，元素为均值
        results["correction_counts"].append([corr_cnt])

    return results
