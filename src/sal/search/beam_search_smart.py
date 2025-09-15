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

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM

# === 修改：引入带 responses 的前瞻生成 ===
from .utils import (
    Beam,
    build_conv,
    generate_k_steps,
    last,
    generate_k_steps_for_llm,
    generate_k_steps_with_responses,  # 新增导入
)

# === 修改：引入可插拔 UQ scorer ===
from sal.models.uq_scorers import PRMScorer, ConfScorer, SemanticEntropyScorer

logger = logging.getLogger()
from sal.utils.score import aggregate_scores

from transformers import AutoTokenizer


def _beam_search(
    batch_of_prompts, config: Config, slm: LLM, prm: PRM, llm: None
) -> tuple[list["Beam"], int]:

    # ---- 选择打分器 ----
    if config.score_method == "prm":
        scorer = PRMScorer(prm)
    elif config.score_method == "conf":
        # conf_strategy: probs_mean | pmax | logmean | margin | neg_entropy
        scorer = ConfScorer(mode=getattr(config, "conf_strategy", "probs_mean"))
    elif config.score_method == "sse":
        scorer = SemanticEntropyScorer(
            samples=getattr(config, "sse_samples", 6),
            embed_model=getattr(
                config, "sse_embed_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            sim_threshold=getattr(config, "sse_sim_threshold", 0.85),
            max_step_tokens=getattr(config, "sse_max_step_tokens", 128),
            temperature=config.temperature,
            top_p=config.top_p,
        )
    else:
        raise ValueError(f"Unknown score_method: {config.score_method}")

    # ---- 生成参数（会按需打开 logprobs）----
    need_lp = getattr(scorer, "requires_vllm_logprobs", False)
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
        logprobs=need_lp,
        top_logprobs=(2 if need_lp else None),
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
                    prm_update=[],  # 保留字段名以兼容下游
                    gen_update=[],
                    llm_tokens=[],
                )
            )

    completed_beams: list[Beam] = []
    total_tokens = 0
    smart_done = False

    for iterate_idx in tqdm(
        range(config.num_iterations), desc="Beam search iterations"
    ):
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

        # 最后一轮：生成到 EOS（去掉 stop；conf 也可不开 logprobs）
        if iterate_idx == config.num_iterations - 1:
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                n=1,
                logprobs=False,  # 最后一轮不需要 online step score
                top_logprobs=None,
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

        # ---- 生成：conf 需要 responses，其它不需要 ----
        if lookahead == 0:
            # 最后一轮：直接展开到 EOS
            gen_results = generate_k_steps(
                templated_convs, lookahead, slm, sampling_params, 1
            )
            responses = None
        else:
            if need_lp:
                gen_results, responses = generate_k_steps_with_responses(
                    templated_convs, lookahead, slm, sampling_params, 1
                )
            else:
                gen_results = generate_k_steps(
                    templated_convs, lookahead, slm, sampling_params, 1
                )
                responses = None

        prev_active_beams = copy.deepcopy(active_beams)

        # copy the active beams to regenerate the beams with llm
        for beam, gen_result in zip(active_beams, gen_results, strict=True):
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            beam.completion_tokens += gen_result.completion_tokens

            beam.current_text += beam.next_texts[0]
            beam.history.append(beam.next_texts[0])
            total_tokens += sum(gen_result.completion_tokens)

            history_text = " ".join(beam.history)
            # 注意：此处的 2048 是历史逻辑；如果你全程 4k/8k，可改成读取 vLLM 的 max_model_len
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

        # ------------------ 追加本轮“步级分” ------------------
        # 为了形状对齐，我们按 prompt 分组（Q × N）
        prompt_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, b in enumerate(active_beams):
            prompt_to_indices[b.prompt].append(i)
        questions = list(prompt_to_indices.keys())

        # candidates：传给各类 scorer
        # - PRM/Conf：传 current_text（PRM会自己拼 system+question；Conf 不怎么用该字段）
        # - SSE：我们希望基于“完整模板上下文”做短采样，因此传 templated_convs
        candidates_by_q: list[list[str]] = []
        for p in questions:
            idxs = prompt_to_indices[p]
            if config.score_method == "sse":
                candidates_by_q.append([templated_convs[i] for i in idxs])
            else:
                candidates_by_q.append([active_beams[i].current_text for i in idxs])

        # 调 scorer：得到 Q × N × [1] 的新步分
        step_scores_qns = []
        if lookahead > 0:
            step_scores_qns = scorer.score_stepwise(
                questions,
                candidates_by_q,
                tokenizer=tokenizer,
                vllm_responses=responses,
                slm=slm if config.score_method == "sse" else None,
                lookahead=lookahead,
                stop=["\n\n"],
            )
            # 写回每条 beam：append 新分
            for qi, p in enumerate(questions):
                idxs = prompt_to_indices[p]
                for local_j, beam_idx in enumerate(idxs):
                    new_step_score = float(step_scores_qns[qi][local_j][-1])
                    active_beams[beam_idx].all_scores.append(new_step_score)

        # ------------------ 计算聚合分并剪枝 ------------------
        # 先过滤已完成
        agg_scores_by_q: list[list[float]] = []
        keep_mask = []
        for i, b in enumerate(active_beams):
            keep_mask.append(not b.completed)

        # 组回 Q × N
        for p in questions:
            idxs = prompt_to_indices[p]
            vals = []
            for i in idxs:
                if keep_mask[i]:
                    vals.append(
                        aggregate_scores(
                            active_beams[i].all_scores, config.agg_strategy
                        )
                    )
            # 注意：若该 prompt 下全完成，vals 可能为空
            if len(vals) == 0:
                agg_scores_by_q.append([])
            else:
                agg_scores_by_q.append(vals)

        # Now filter active_beams and agg_scores for beams that are completed
        prev_active_beams = [
            b
            for idx, b in enumerate(prev_active_beams)
            if not active_beams[idx].completed
        ]
        active_beams = [b for b in active_beams if not b.completed]

        # Early stopping if all beams are completed
        if len(active_beams) == 0:
            break
        if not config.sort_completed and len(completed_beams) >= config.n:
            break

        # Filter duplicate active beams
        if config.filter_duplicates:
            unique_beam_dict = {}
            for i, b in enumerate(active_beams):
                if b.current_text not in unique_beam_dict:
                    unique_beam_dict[b.current_text] = i
            # 重建映射后的列表（注意同时要同步聚合分列表）
            keep_indices = list(unique_beam_dict.values())
            active_beams = [active_beams[i] for i in keep_indices]
            prev_active_beams = [prev_active_beams[i] for i in keep_indices]

            # 由于 agg_scores_by_q 是按 prompt 分组的，我们在下方直接从 active_beams 现值重算一份扁平聚合分，避免错位
        flat_agg = [
            aggregate_scores(b.all_scores, config.agg_strategy) for b in active_beams
        ]

        # Get indices for top (config.n / config.beam_width) completions
        top_k = max(1, (config.n // config.beam_width))
        top_indices = np.argsort(np.array(flat_agg))[-top_k:]

        for idx, beam in enumerate(active_beams):
            if idx not in top_indices:
                beam.pruned = True

        # SMART beam search implementation
        # 仅对 top 中“聚合分 < threshold”的做纠偏
        re_indices = [i for i in top_indices if flat_agg[i] < config.threshold]
        if len(re_indices) == 0:
            continue

        smart_done = True
        re_beams = [prev_active_beams[idx] for idx in re_indices]

        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt) for b in re_beams
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
        lookahead_corr = (
            0 if iterate_idx == config.num_iterations - 1 else config.lookahead
        )
        gen_results = generate_k_steps_for_llm(
            tokenizer_llm, templated_convs, lookahead_corr, llm, config, 1
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
            reprompts.append(beam.prompt)
            recompletions.append([beam.current_text])

        # 纠偏后的打分更新：
        # - prm：重算并覆盖（保持原逻辑）
        # - sse：用 slm 重算本轮 step 分，替换刚才 append 的那一分（更稳）
        # - conf：无法拿到 llm 的 logprobs，沿用原分（不变）
        if config.score_method == "prm":
            re_scores = prm.score(reprompts, recompletions)
            reagg_scores = [
                [aggregate_scores(s, config.agg_strategy) for s in score]
                for score in re_scores
            ]
            for beam, score in zip(re_beams, re_scores, strict=True):
                beam.all_scores = score[0]  # 覆盖：保持你原来的行为
        elif config.score_method == "sse":
            # 用 slm 对纠偏后的上下文重算一次 SSE 分（只 1 step）
            # 先把 re_beams 的上下文套模板（用 vLLM tokenizer）
            templ_convs_vllm = slm.get_tokenizer().apply_chat_template(
                [
                    build_conv(b.prompt, b.current_text, config.system_prompt)
                    for b in re_beams
                ],
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                tokenize=False,
            )
            # Q=1 的简化：questions 取对应 prompt，cands 放模板字符串
            q_list = [b.prompt for b in re_beams]
            cand_list = [[tc] for tc in templ_convs_vllm]
            sse_scores_qns = SemanticEntropyScorer(
                samples=getattr(config, "sse_samples", 6),
                embed_model=getattr(
                    config, "sse_embed_model", "sentence-transformers/all-MiniLM-L6-v2"
                ),
                sim_threshold=getattr(config, "sse_sim_threshold", 0.85),
                max_step_tokens=getattr(config, "sse_max_step_tokens", 128),
                temperature=config.temperature,
                top_p=config.top_p,
            ).score_stepwise(
                q_list,
                cand_list,
                tokenizer=slm.get_tokenizer(),
                slm=slm,
                lookahead=lookahead_corr,
                stop=["\n\n"],
            )
            # 用新的分替换当前步最后一个分
            for beam, s in zip(re_beams, sse_scores_qns, strict=True):
                if len(beam.all_scores) == 0:
                    beam.all_scores = [float(s[0][-1])]
                else:
                    beam.all_scores[-1] = float(s[0][-1])

            # 方便记录：计算纠偏前聚合与纠偏后聚合
            reagg_scores = [
                [aggregate_scores(b.all_scores, config.agg_strategy)] for b in re_beams
            ]
        else:
            # conf：不更新分，只记录“未变化”
            reagg_scores = [[flat_agg[r]] for r in re_indices]

        for i, (re_idx, beam) in enumerate(zip(re_indices, re_beams)):
            # log correction information
            beam.smart_step.append(iterate_idx)
            beam.gen_update.append(
                (active_beams[re_idx].next_texts[0], beam.next_texts[0])
            )
            beam.prm_update.append(
                (flat_agg[re_idx], reagg_scores[i][0])
            )  # 仍用原字段名
            beam.llm_tokens.append(len(tokenizer_llm.encode(beam.next_texts[0])))
            total_tokens += len(tokenizer_llm.encode(beam.next_texts[0]))
            active_beams[re_idx] = beam

    # Filter completed beams for those with top config.n scores
    if config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: config.n]
    else:
        completed_beams = completed_beams[: config.n]

    if len(completed_beams) != config.n:
        # If we don't have enough completed_beams, duplicate until we reach config.n
        repeats = (config.n // len(completed_beams)) + 1
        logger.debug(
            f"Extending completed_beams with {repeats} repetitions to reach size {config.n}"
        )
        extended_completed_beams = [
            copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]
        ]
        completed_beams = extended_completed_beams

    for beam in completed_beams:
        if len(beam.smart_step) == 0:
            beam.smart_step = [-1]
            beam.prm_update = [(-1.0, -1.0)]
            beam.gen_update = [("-1", "-1")]
            beam.llm_tokens = [-1]

    return completed_beams, total_tokens


def smart_beam_search(examples, config: Config, slm: LLM, prm: PRM, llm: None):
    problems = examples["problem"]
    beam_results, total_tokens = _beam_search(problems, config, slm, prm, llm)

    # Group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "scores": [], "pred": []}
    tokenizer = slm.get_tokenizer()

    for p in problems:
        beams = grouped_results[p]
        completions = [b.current_text for b in beams]
        scores = [b.all_scores for b in beams]  # 保存原始逐步分
        pred = completions[
            np.argmax(
                [aggregate_scores(b.all_scores, config.agg_strategy) for b in beams]
            )
        ]
        results["completions"].append(completions)
        results["scores"].append(scores)
        results["pred"].append(pred)
    return results
