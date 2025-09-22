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

from .utils import Beam, build_conv, generate_k_steps, generate_k_steps_for_llm

logger = logging.getLogger()
from sal.utils.score import aggregate_scores

from transformers import AutoTokenizer


def _beam_search(
    batch_of_prompts,
    config: Config,
    slm: LLM,
    prm: PRM,
    llm: None,
    random_quotas_by_prompt: list[list[int]] | None = None,
    random_seed: int | None = None,
) -> tuple[list[Beam], int]:
    """
    当 random_quotas_by_prompt 为 None 时，执行 SMART/PRM 流程；
    否则执行“随机基线”（随机 prune + 随机调度 LLM 纠错，PRM 仅用于记录/监控，不影响决策）。
    """
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
    )

    beams: list[Beam] = []
    # 建立所有 beam，并打上 prompt_idx（用于随机调度的槽位键）
    for p_idx, prompt in enumerate(batch_of_prompts):
        for i in range(config.n):
            b = Beam(
                prompt=prompt,
                index=i,
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
            )
            # 兜底：若 utils.Beam 没有该字段，动态赋 0
            if not hasattr(b, "llm_corrections"):
                setattr(b, "llm_corrections", 0)
            # 标注样本槽位索引
            setattr(b, "prompt_idx", p_idx)
            beams.append(b)

    # 若给定“随机配额”（每个槽位纠错几次），预先抽样每个槽位在哪些迭代用 LLM
    schedule_by_slot = {}
    if random_quotas_by_prompt is not None:
        rng = np.random.default_rng(0 if random_seed is None else random_seed)
        assert len(random_quotas_by_prompt) == len(batch_of_prompts)
        for p_idx, quotas in enumerate(random_quotas_by_prompt):
            assert len(quotas) == config.n
            for i, q in enumerate(quotas):
                k = max(0, int(q))
                if k > 0:
                    iters = set(
                        rng.choice(
                            config.num_iterations, size=k, replace=False
                        ).tolist()
                    )
                else:
                    iters = set()
                schedule_by_slot[(p_idx, i)] = iters

    completed_beams: list[Beam] = []
    total_tokens = 0

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

        # 最后一轮：不再用 stop 字符，直接生成到 EOS/max_tokens
        if iterate_idx == config.num_iterations - 1:
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
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

        # PRM 打分（SMART/PRM 流程用于决策；随机基线仅记录、不可用于决策）
        scores = prm.score(prompts, completions)
        agg_scores = [
            [aggregate_scores(s, config.agg_strategy) for s in score]
            for score in scores
        ]
        for beam, score in zip(active_beams, scores, strict=True):
            beam.all_scores = score[0]

        # 过滤已完成的 beam
        agg_scores = [
            agg_scores[i] for i, b in enumerate(active_beams) if not b.completed
        ]
        prev_active_beams = [
            b
            for idx, b in enumerate(prev_active_beams)
            if not active_beams[idx].completed
        ]
        active_beams = [b for b in active_beams if not b.completed]

        # 所有都结束就跳出
        if len(active_beams) == 0:
            break
        # 不排序且已集满 config.n 个完成，就提前停止
        if not config.sort_completed and len(completed_beams) >= config.n:
            break

        # 去重（按 current_text）
        if config.filter_duplicates:
            uniq = {}
            for i, b in enumerate(active_beams):
                if b.current_text not in uniq:
                    uniq[b.current_text] = i
            active_beams = [active_beams[i] for i in uniq.values()]
            prev_active_beams = [prev_active_beams[i] for i in uniq.values()]
            agg_scores = [agg_scores[i] for i in uniq.values()]

        # —— 关键修改 1：本轮保留的槽位 top_indices —— #
        k = max(1, config.n // config.beam_width)
        if random_quotas_by_prompt is None:
            # SMART：按 PRM 选 top-k
            top_indices = np.argsort(np.array(agg_scores).flatten())[-k:]
        else:
            # 随机基线：top-k 槽位随机抽
            rng_top = np.random.default_rng(0 if random_seed is None else random_seed)
            total = len(agg_scores)  # 与 active_beams 同长
            k = min(k, total)
            if k == 0:
                top_indices = np.array([], dtype=int)
            else:
                top_indices = rng_top.choice(total, size=k, replace=False)

        # 非 top_indices 的 beam 本轮 prune
        for idx, beam in enumerate(active_beams):
            if idx not in top_indices:
                beam.pruned = True

        # —— 关键修改 2：从 top_indices 里选择需要“用 LLM 纠错”的槽位 —— #
        if random_quotas_by_prompt is None:
            # 正常 SMART：PRM 分数低于阈值的槽位走 LLM（限定在 top_indices 内）
            re_indices = [i for i in top_indices if agg_scores[i][0] < config.threshold]
        else:
            # 随机基线：只在本轮保留的随机 top_indices 中，根据预先抽好的迭代号决定是否用 LLM
            re_indices = []
            for i in top_indices:
                b = prev_active_beams[i]
                if b.completed:
                    continue
                key = (getattr(b, "prompt_idx", 0), b.index)
                iters = schedule_by_slot.get(key)
                if iters is not None and iterate_idx in iters:
                    re_indices.append(i)
                    iters.discard(iterate_idx)

        if len(re_indices) == 0:
            continue

        # 用大模型纠错
        re_beams = [prev_active_beams[i] for i in re_indices]
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
        lookahead = 0 if iterate_idx == config.num_iterations - 1 else config.lookahead
        gen_results = generate_k_steps_for_llm(
            tokenizer_llm, templated_convs, lookahead, llm, config, 1
        )

        reprompts, recompletions = [], []
        for beam, gen_result in zip(re_beams, gen_results, strict=True):
            # 更新 beam
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

        # 纠错后 PRM 分仅记录，不影响随机基线的决策
        re_scores = prm.score(reprompts, recompletions)
        for beam, score in zip(re_beams, re_scores, strict=True):
            beam.all_scores = score[0]

        for i, beam in zip(re_indices, re_beams):
            beam.smart_step.append(iterate_idx)
            beam.gen_update.append((active_beams[i].next_texts[0], beam.next_texts[0]))
            beam.llm_tokens.append(len(tokenizer_llm.encode(beam.next_texts[0])))
            total_tokens += len(tokenizer_llm.encode(beam.next_texts[0]))
            # 纠错次数 +1
            if not hasattr(beam, "llm_corrections"):
                setattr(beam, "llm_corrections", 0)
            beam.llm_corrections += 1
            active_beams[i] = beam

    # 收尾：只保留前 n 个完成 beam（可选排序）
    if config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: config.n]
    else:
        completed_beams = completed_beams[: config.n]

    # 数量不足则深拷贝补齐
    if len(completed_beams) != config.n:
        repeats = (config.n // len(completed_beams)) + 1
        extended_completed_beams = [
            copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]
        ]
        completed_beams = extended_completed_beams

    # 没走过 SMART 的 beam 填入默认占位
    for beam in completed_beams:
        if len(beam.smart_step) == 0:
            beam.smart_step = [-1]
            beam.prm_update = [(-1.0, -1.0)]
            beam.gen_update = [("-1", "-1")]
            beam.llm_tokens = [-1]
        if not hasattr(beam, "llm_corrections"):
            setattr(beam, "llm_corrections", 0)

    return completed_beams, total_tokens


def smart_beam_search(examples, config: Config, slm: LLM, prm: PRM, llm: None):
    problems = examples["problem"]

    # 1) 先跑正常 SMART/PRM 流程
    beam_results_prm, total_tokens_prm = _beam_search(problems, config, slm, prm, llm)

    # 按 prompt 分组
    grouped_prm = defaultdict(list)
    for b in beam_results_prm:
        grouped_prm[b.prompt].append(b)

    # 为随机基线准备“纠错配额”（每个 prompt 的 16 个槽位各自用了几次 LLM）
    quotas_by_prompt = []
    for p in problems:
        beams = grouped_prm[p]
        quotas_by_prompt.append([getattr(b, "llm_corrections", 0) for b in beams])

    # 2) 跑“随机基线”（随机 prune + 随机调度 LLM 纠错，PRM 仅记录不决策）
    beam_results_rand, total_tokens_rand = _beam_search(
        problems,
        config,
        slm,
        prm,
        llm,
        random_quotas_by_prompt=quotas_by_prompt,
        random_seed=getattr(config, "seed", None),
    )

    grouped_rand = defaultdict(list)
    for b in beam_results_rand:
        grouped_rand[b.prompt].append(b)

    # 3) 组装输出（只保留你需要的 random_uniform）
    results = {
        "completions": [],  # PRM流程产出的16条
        "pred": [],  # PRM选优的最终答案
        "scores": [],  # PRM流程里每条的打分（记录）
        "llm_corrections": [],  # PRM流程里每条纠错次数
        "completions_random": [],  # 随机流程产出的16条
        "pred_random_uniform": [],  # 随机流程里从16条中均匀随机挑1条
        "llm_corrections_random": [],  # 随机流程里每条纠错次数
    }

    rng = np.random.default_rng(getattr(config, "seed", None))

    for p in problems:
        # --- PRM 流程 ---
        beams_p = grouped_prm[p]
        completions = [b.current_text for b in beams_p]
        scores = [b.all_scores for b in beams_p]
        corrections = [getattr(b, "llm_corrections", 0) for b in beams_p]
        # 用“aggregate_scores + argmax”逻辑选 pred
        pred = completions[
            np.argmax(
                [aggregate_scores(b.all_scores, config.agg_strategy) for b in beams_p]
            )
        ]
        results["completions"].append(completions)
        results["pred"].append(pred)
        results["scores"].append(scores)
        results["llm_corrections"].append(corrections)

        # --- 随机基线 ---
        beams_r = grouped_rand[p]
        completions_r = [b.current_text for b in beams_r]
        corrections_r = [getattr(b, "llm_corrections", 0) for b in beams_r]
        # 均匀随机挑一个
        rand_idx = rng.integers(low=0, high=len(completions_r))
        pred_r_uniform = completions_r[rand_idx]

        results["completions_random"].append(completions_r)
        results["pred_random_uniform"].append(pred_r_uniform)
        results["llm_corrections_random"].append(corrections_r)

    return results
