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

from .utils import Beam, build_conv, generate_k_steps, last, generate_k_steps_for_llm

logger = logging.getLogger()
from sal.utils.score import aggregate_scores

from transformers import AutoTokenizer

# ==== UQHead: 新增依赖 ====
from transformers import AutoModelForCausalLM, AutoTokenizer as HFAutoTokenizer
from lm_polygraph.model_adapters import WhiteboxModelBasic
from lm_polygraph.stat_calculators.extract_claims import Claim
from luh.auto_uncertainty_head import AutoUncertaintyHead
from luh.calculator_infer_luh import CalculatorInferLuh
from luh.calculator_apply_uq_head import CalculatorApplyUQHead

# ==== UQHead: 惰性上下文（避免顶层用 config 变量）====
_UQ_CTX = {
    "wb": None,
    "uhead": None,
    "calc_infer": None,
    "calc_apply": None,
    "chat_template": None,
}

def _ensure_uq_ctx(config: Config):
    """按需初始化 UQ 运行环境，并与 config.custom_chat_template 保持一致。"""
    if _UQ_CTX["wb"] is not None:
        if _UQ_CTX["chat_template"] != getattr(config, "custom_chat_template", None):
            if config.custom_chat_template is not None:
                _UQ_CTX["wb"].tokenizer.chat_template = config.custom_chat_template
            _UQ_CTX["chat_template"] = getattr(config, "custom_chat_template", None)
        return

    uq_model_path = getattr(config, "uq_model_path", config.model_path)
    hf_tok = HFAutoTokenizer.from_pretrained(uq_model_path, use_fast=True)
    if hf_tok.pad_token is None:
        hf_tok.pad_token = hf_tok.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(
        uq_model_path, device_map="auto", torch_dtype="auto",
    )
    wb = WhiteboxModelBasic(
        model=hf_model,
        tokenizer=hf_tok,
        tokenizer_args=dict(
            add_special_tokens=False, return_tensors="pt", padding=True, truncation=True
        ),
        model_type="CausalLM",
    )
    uq_head_path = getattr(
        config, "uq_head_path", "llm-uncertainty-head/uhead_claim_Mistral-7B-Instruct-v0.2"
    )
    uhead = AutoUncertaintyHead.from_pretrained(uq_head_path, base_model=hf_model)

    calc_infer = CalculatorInferLuh(
        uhead, tokenize=True,
        args_generate={"max_new_tokens": 0},  # 只前向，不生成
        device="cuda",
        generations_cache_dir="",
        predict_token_uncertainties=True,     # 要 token 级 UE
    )
    calc_apply = CalculatorApplyUQHead(uhead)

    if getattr(config, "custom_chat_template", None) is not None:
        wb.tokenizer.chat_template = config.custom_chat_template

    _UQ_CTX.update(
        dict(
            wb=wb, uhead=uhead, calc_infer=calc_infer, calc_apply=calc_apply,
            chat_template=getattr(config, "custom_chat_template", None),
        )
    )

def _compute_uq_for_texts(convs: list[str], add_generation_prompt: bool,
                          continue_final_message: bool, beams: list[Beam], config: Config):
    """
    返回：
      - overall_conf: 每个 beam 的整体"置信分" = -mean(token_UE[0:cur_len])，越大越好
      - step_uq: 每个 beam 本轮"新增 tokens 的 UE 均值"（用于纠错触发）
    
    注意：将每一步生成的文本段（step）视为一个 claim
    """
    _ensure_uq_ctx(config)
    wb = _UQ_CTX["wb"]
    calc_infer = _UQ_CTX["calc_infer"]
    calc_apply = _UQ_CTX["calc_apply"]

    templated = wb.tokenizer.apply_chat_template(
        convs,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        tokenize=False,
    )
    
    # 为每个 beam 创建 claims：每一步生成的文本段就是一个 claim
    all_claims = []
    for beam in beams:
        beam_claims = []
        if len(beam.history) > 0:
            # 获取完整文本的 token ids（用于计算 token 位置）
            full_text = beam.current_text
            full_token_ids = wb.tokenizer.encode(full_text, add_special_tokens=False)
            
            # 为每个 step（history 中的每一项）创建一个 claim
            current_pos = 0
            for step_text in beam.history:
                if not step_text.strip():
                    continue
                # 找到 step_text 在 full_text 中的位置
                step_start = full_text.find(step_text, current_pos)
                if step_start == -1:
                    # 如果找不到，尝试从 current_pos 开始匹配
                    step_start = full_text.find(step_text)
                if step_start == -1:
                    continue
                
                # 计算 step 对应的 token 位置
                text_before_step = full_text[:step_start]
                tokens_before = wb.tokenizer.encode(text_before_step, add_special_tokens=False)
                step_tokens = wb.tokenizer.encode(step_text, add_special_tokens=False)
                
                # aligned_token_ids 是相对于完整文本的 token 位置（不包括 context）
                # 在 CalculatorApplyUQHead 中会加上 context_length
                aligned_token_ids = list(range(len(tokens_before), len(tokens_before) + len(step_tokens)))
                
                claim = Claim(
                    claim_text=step_text,
                    sentence=full_text,
                    aligned_token_ids=aligned_token_ids
                )
                beam_claims.append(claim)
                current_pos = step_start + len(step_text)
        all_claims.append(beam_claims)
    
    deps = {}
    deps["claims"] = all_claims  # 添加 claims 到 deps
    deps.update(calc_infer(deps, texts=templated, model=wb))
    deps.update(calc_apply(deps, texts=templated, model=wb))

    token_ues = deps.get("token_uncertainties", None)  # List[np.ndarray], 每条长度 = 当前 token 长度
    input_ids = deps.get("input_ids", None)

    if token_ues is None or input_ids is None:
        # 回退：不给分
        return [0.0 for _ in beams], [0.0 for _ in beams]

    overall_conf, step_uq = [], []
    for beam, ue_vec, ids in zip(beams, token_ues, input_ids, strict=True):
        cur_len = len(ids)
        # 整体置信 = -mean(UE[0:cur_len]) —— 更低不确定 => 更高得分
        if cur_len > 0:
            conf = float(-np.mean(ue_vec[:cur_len]))
        else:
            conf = 0.0
        overall_conf.append(conf)

        # 本轮新增段（用于纠错触发）
        prev = getattr(beam, "measured_len", 0)
        prev = max(0, min(prev, cur_len))
        seg = ue_vec[prev:cur_len]
        step_score = float(np.mean(seg)) if len(seg) > 0 else 0.0
        step_uq.append(step_score)
        # 标记已测长度
        beam.measured_len = cur_len

        # 记录
        if not hasattr(beam, "uq_scores"):
            beam.uq_scores = []
        beam.uq_scores.append(step_score)

    return overall_conf, step_uq


# ==== UQHead: 以上为新增 ====


def _beam_search(
    batch_of_prompts, config: Config, slm: LLM, prm: PRM, llm: None,
) -> tuple[list[Beam], int]:
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
    )

    beams: list[Beam] = []
    for p_idx, prompt in enumerate(batch_of_prompts):
        for i in range(config.n):
            b = Beam(
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
            beams.append(b)

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

        if iterate_idx == config.num_iterations - 1:
            # Last iteration, generate to EOS
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

        # 更新 beams 的文本与状态
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

        # ==== UQHead: 用 UHead 完全替代 PRM 进行“打分/排序” ====
        # 1) 计算整体置信（-mean UE）与本轮新增段 UE
        overall_conf, step_uq = _compute_uq_for_texts(
            convs=[build_conv(b.prompt, b.current_text, config.system_prompt) for b in active_beams],
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            beams=active_beams,
            config=config,
        )
        # 2) 将每个 step 的 uq 累积到 beam.all_scores（对齐 PRM 版本：累积每个 step 的分数）
        #    注意：step_uq 是当前 step 的不确定性，越小越好
        for beam, uq in zip(active_beams, step_uq, strict=True):
            beam.all_scores.append(uq)
        # 使用 aggregate_scores 聚合所有步骤的分数（对齐 PRM 版本）
        agg_scores = [
            [aggregate_scores(b.all_scores, config.agg_strategy)]
            for b in active_beams if not b.completed
        ]
        prev_active_beams = [b for idx, b in enumerate(prev_active_beams) if not active_beams[idx].completed]
        active_beams = [b for b in active_beams if not b.completed]

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
            # 重新计算 step_uq 用于纠错触发（因为 active_beams 被过滤了）
            _, step_uq = _compute_uq_for_texts(
                convs=[build_conv(b.prompt, b.current_text, config.system_prompt) for b in active_beams],
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                beams=active_beams,
                config=config,
            )

        # 先按整体置信取 Top-K（越大越好）
        top_indices = np.argsort(np.array(agg_scores).flatten())[
            -(config.n // config.beam_width) :
        ]

        for idx, beam in enumerate(active_beams):
            if idx not in top_indices:
                beam.pruned = True

        # 纠错触发：基于 UHead 的"新增段 UE"阈值
        # 阈值：绝对值或分位数
        uq_th = getattr(config, "uq_threshold", 0.5)
        use_q = getattr(config, "uq_use_quantile", False)
        q = float(getattr(config, "uq_quantile", 0.8))
        if use_q and len(top_indices) > 0:
            th = float(np.quantile([step_uq[i] for i in top_indices], q))
        else:
            th = float(uq_th)
        re_indices = [i for i in top_indices if step_uq[i] > th]
        if len(re_indices) == 0:
            continue

        re_beams = [prev_active_beams[idx] for idx in re_indices]

        convs = [build_conv(b.prompt, b.current_text, config.system_prompt) for b in re_beams]
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
        gen_results = generate_k_steps_for_llm(tokenizer, templated_convs, lookahead, llm, config, 1)

        reprompts, recompletions = [], []
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
            reprompts.append(beam.prompt)
            recompletions.append([beam.current_text])

        # ==== UQHead: 纠错后，重算 step_uq，更新 all_scores 最后一个元素 ====
        _, step_uq_after = _compute_uq_for_texts(
            convs=[build_conv(b.prompt, b.current_text, config.system_prompt) for b in re_beams],
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            beams=re_beams,
            config=config,
        )
        # 更新 all_scores 的最后一个元素为纠错后的 step_uq（对齐 PRM：更新最后一个 step 的分数）
        # 注意：需要在更新前获取 before_uq，因为 re_beams 和 prev_active_beams 共享引用
        for i, (re_idx, beam) in enumerate(zip(re_indices, re_beams)):
            # 先获取纠错前的 step_uq（在更新前）
            before_uq = -1.0
            if re_idx < len(prev_active_beams):
                prev_beam = prev_active_beams[re_idx]
                before_uq = float(prev_beam.all_scores[-1]) if len(prev_beam.all_scores) > 0 else -1.0
            
            # 更新 all_scores 的最后一个元素为纠错后的 step_uq
            uq_after = float(step_uq_after[i]) if i < len(step_uq_after) else before_uq
            if len(beam.all_scores) > 0:
                beam.all_scores[-1] = uq_after
            else:
                beam.all_scores.append(uq_after)
            
            # 记录信息
            beam.smart_step.append(iterate_idx)
            beam.gen_update.append((active_beams[re_idx].next_texts[0], beam.next_texts[0]))
            beam.prm_update.append((before_uq, uq_after))
            beam.llm_tokens.append(len(tokenizer.encode(beam.next_texts[0])))
            total_tokens += len(tokenizer.encode(beam.next_texts[0]))
            active_beams[re_idx] = beam
        
    # 完成/收尾：用整体置信排序（越大越好）
    if config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: config.n]
    else:
        completed_beams = completed_beams[: config.n]
    if len(completed_beams) != config.n:
        repeats = (config.n // len(completed_beams)) + 1
        logger.debug(f"Extending completed_beams with {repeats} repetitions to reach size {config.n}")
        extended_completed_beams = [copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]]
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

    # UHead-驱动的 SMART
    beam_results_main, _ = _beam_search(problems, config, slm, prm, llm)
    grouped_main = defaultdict(list)
    for b in beam_results_main:
        grouped_main[b.prompt].append(b)


    results = {
        "completions": [], "pred": [], "scores": [],
        "llm_tokens": [],
        "correction_counts": [],
    }

    for p in problems:
        beams = grouped_main[p]
        completions = [b.current_text for b in beams]
        scores = [b.all_scores for b in beams]  # all_scores = [-mean UE] 形式
        # 选整体置信最高（= 不确定最低）
        pred = completions[np.argmax([aggregate_scores(b.all_scores, config.agg_strategy) for b in beams])]
        results["completions"].append(completions)
        results["pred"].append(pred)
        results["scores"].append(scores)
        results["llm_tokens"].append([getattr(b, "llm_tokens", []) for b in beams])
        results["correction_counts"].append([len(getattr(b, "llm_tokens", [])) for b in beams])

    return results
