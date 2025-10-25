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


import math

from datasets import Dataset
from tqdm import tqdm
import numpy as np 

from sal.config import Config
from sal.utils.math import (
    compute_maj_pred,
    compute_naive_pred,
    compute_weighted_pred,
    extract_completion_answers,
    subsample_completions,
)

def calculate_confidence_score(answer_tokens_logprobs_list):
    """
    answer_tokens_logprobs_list에서 logprob 값을 합산하여 log-likelihood 및 likelihood를 계산하는 함수.

    Args:
        answer_tokens_logprobs_list (list of dict): [{token_id: Logprob(logprob=value, ...)}, {...}, ...]

    Returns:
        tuple: (likelihood, likelihood_mean, probs_mean)
    * mean : likelihood(norm)
    * sum할때 -> answer_tokens_logprobs_list의 갯수를 뽑을 수 있는데 = T, 
    각 generation hyperparameter=e 
    e^T
    """
    log_likelihood_of_completion = sum(next(iter(logprob.values())).logprob for logprob in answer_tokens_logprobs_list)
    
    likelihood_score = np.exp(log_likelihood_of_completion)
    
    T = len(answer_tokens_logprobs_list) if len(answer_tokens_logprobs_list) > 0 else 1
    likelihood_mean_score = np.exp(log_likelihood_of_completion / T)
    
    probs_mean_score = np.mean([np.exp(next(iter(logprob.values())).logprob) for logprob in answer_tokens_logprobs_list])
    
    return likelihood_score, likelihood_mean_score, probs_mean_score


def calculate_perplexity_score(answer_tokens_logprobs_list):
    """
    answer_tokens_logprobs_list에서 perplexity를 계산하는 함수.
    Perplexity = exp(-mean(log_prob)) = exp(-log_likelihood / T)

    Args:
        answer_tokens_logprobs_list (list of dict): [{token_id: Logprob(logprob=value, ...)}, {...}, ...]

    Returns:
        tuple: (perplexity, normalized_perplexity, token_perplexity)
        * perplexity: exp(-log_likelihood) = 1/likelihood
        * normalized_perplexity: exp(-log_likelihood / T) = perplexity^(1/T)
        * token_perplexity: exp(-mean(log_prob_per_token))
    """
    
    log_likelihood_of_completion = sum(next(iter(logprob.values())).logprob for logprob in answer_tokens_logprobs_list)
    
    # Standard perplexity: exp(-log_likelihood)
    perplexity_score = np.exp(-log_likelihood_of_completion)
    
    T = len(answer_tokens_logprobs_list)
    # Normalized perplexity: exp(-log_likelihood / T)
    normalized_perplexity_score = np.exp(-log_likelihood_of_completion / T)
    
    # Token-level perplexity: exp(-mean(log_prob_per_token))
    token_logprobs = [next(iter(logprob.values())).logprob for logprob in answer_tokens_logprobs_list]
    token_perplexity_score = np.exp(-np.mean(token_logprobs))
    
    return perplexity_score, normalized_perplexity_score, token_perplexity_score


def aggregate_scores(
    scores: list[float], agg_strategy: str
) -> float:
    print(f"[DEBUG] aggregate_scores called with scores: {scores}, type: {type(scores)}, agg_strategy: {agg_strategy}")
    
    # Handle case where scores is already a single float (already aggregated)
    if isinstance(scores, (int, float)):
        print(f"[DEBUG] scores is already a single value: {scores}")
        return float(scores)
    
    # Handle empty list
    if not scores:
        print(f"[DEBUG] scores is empty, returning 0.0")
        return 0.0
    
    # Check for None values
    if any(s is None for s in scores):
        print(f"[DEBUG] scores contains None values: {scores}")
        # Filter out None values
        scores = [s for s in scores if s is not None]
        if not scores:
            print(f"[DEBUG] After filtering None values, scores is empty, returning 0.0")
            return 0.0
    
    print(f"[DEBUG] Processing scores: {scores}")
    
    if agg_strategy == "min":
        result = min(scores)
    elif agg_strategy == "prod":
        result = math.prod(scores)
    elif agg_strategy == "last":
        result = scores[-1]
    else:
        raise ValueError(f"Invalid aggregation strategy: {agg_strategy}")
    
    print(f"[DEBUG] aggregate_scores result: {result}, type: {type(result)}")
    return result


def score(dataset: Dataset, config: Config) -> Dataset:
    print(f"[DEBUG] score function called with dataset size: {len(dataset)}")
    print(f"[DEBUG] config.agg_strategy: {config.agg_strategy}")
    
    # Debug the first few samples
    print(f"[DEBUG] First sample scores: {dataset[0]['scores'] if len(dataset) > 0 else 'No samples'}")
    
    dataset = dataset.map(
        lambda x: {
            "agg_scores": [aggregate_scores(s, "last") for s in x["scores"]],
            "debug_scores_info": [f"type: {type(s)}, len: {len(s) if hasattr(s, '__len__') else 'N/A'}, content: {s}" for s in x["scores"]]
        }
    )
    subsets = [2**i for i in range(config.n) if 2**i <= config.n]
    for n in tqdm(subsets, desc="Computing majority & weighted predictions"):
        dataset = dataset.map(
            subsample_completions,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Subsample {n}",
        )
        dataset = dataset.map(
            extract_completion_answers,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Extract answers {n}",
        )
        dataset = dataset.map(
            compute_weighted_pred,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Compute weighted pred {n}",
        )
        dataset = dataset.map(
            compute_maj_pred,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Compute majority pred {n}",
        )
        dataset = dataset.map(
            compute_naive_pred,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Compute naive pred {n}",
        )
        # Nuke unused columns to keep dataset lean
        dataset = dataset.remove_columns(
            [f"completions@{n}", f"agg_scores@{n}", f"preds@{n}"]
        )
    return dataset

from typing import Callable, List, Dict, Any, Tuple

def calculate_cocoa_uq_scores(
    beam_logprobs_list: List[List[Dict[int, Any]]],
    *,
    detok: Callable[[List[int]], str],           # List[int] -> str
    embed_fn: Callable[[List[str]], np.ndarray], # List[str] -> np.ndarray, shape (B, D)
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    """
    CoCoA-style scores for the first beam (y*):
      - MSP (paper):  u_msp     = 1 - exp(sum_t log p_t) = 1 - p(y*|x)
      - PPL:          u_ppl     = -mean_t log p_t
      - Entropy:      u_entropy = mean_t H_t,  H_t = -sum(lp * exp(lp)) per step

    Each base u is multiplied by semantic inconsistency:
      mean_i (1 - cosine_sim(embed(y*), embed(y^i))) for i >= 1 (exclude self).
    """

    def _seq_token_ids(step_dict_list: List[Dict[int, Any]]) -> List[int]:
        return [next(iter(step.keys())) for step in step_dict_list]

    def _cosine_sim(a: np.ndarray, b: np.ndarray, eps_inner: float = 1e-12) -> float:
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na < eps_inner or nb < eps_inner:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # ----- first beam (y*) -----
    y_star_steps = beam_logprobs_list[0]
    logps = [next(iter(step.values())).logprob for step in y_star_steps] if y_star_steps else []

    # MSP base (paper): u = 1 - p(y*|x)
    if logps:
        logp_star = float(np.sum(logps))             # log p(y*|x)
        p_star = float(np.exp(logp_star))            # may underflow to 0 for long seqs (that's fine)
        p_star = float(np.clip(p_star, 0.0, 1.0))
        base_msp = 1.0 - p_star
    else:
        base_msp = 0.0

    # PPL base: -mean log p_t
    base_ppl = -float(np.mean(logps)) if logps else 0.0

    # Entropy base: mean_t H_t = -sum(lp * exp(lp)) per step
    if y_star_steps:
        step_H = []
        for step_dict in y_star_steps:
            lp = np.array([v.logprob for v in step_dict.values()], dtype=float)
            mask = ~np.isinf(lp)
            H_t = -float(np.sum(lp[mask] * np.exp(lp[mask])))
            step_H.append(H_t)
        base_entropy = float(np.mean(step_H)) if step_H else 0.0
    else:
        base_entropy = 0.0

    # ----- semantic inconsistency term (shared) -----
    token_seqs = [_seq_token_ids(b) for b in beam_logprobs_list]
    texts = [detok(ids) for ids in token_seqs]

    embs = embed_fn(texts)
    if not isinstance(embs, np.ndarray):
        embs = np.asarray(embs, dtype=float)

    e0 = embs[0]
    sims01 = []
    for k in range(1, len(beam_logprobs_list)):  # exclude self
        s = _cosine_sim(e0, embs[k])
        s01 = float(np.clip((s + 1.0) * 0.5, 0.0, 1.0))  # [-1,1] -> [0,1]
        sims01.append(s01)
    mean_dissim = 0.0 if not sims01 else float(np.mean([1.0 - s for s in sims01]))

    # ----- final CoCoA-enriched scores -----
    cocoa_msp     = float(base_msp     * mean_dissim)
    cocoa_ppl     = float(base_ppl     * mean_dissim)
    cocoa_entropy = float(base_entropy * mean_dissim)

    return cocoa_msp, cocoa_ppl, cocoa_entropy


from typing import List, Dict, Any, Tuple
import numpy as np

# -----------------------------
# 2) Top-2 margin（步级前两大概率差）
# -----------------------------
def calculate_top2_margin_scores(
    beam_logprobs_list: List[List[Dict[int, Any]]]
) -> Tuple[float, float, List[float]]:
    """
    基于第一条 beam（y*）计算每步的 top-2 margin: margin_t = p1 - p2
    返回:
      (min_margin, mean_margin, margins_per_step)
    说明:
      - 若某步只有一个候选概率，则视为 p2=0，margin = p1
      - 若该步不存在有限 logprob，跳过
    """

    y_star_steps = beam_logprobs_list[0]
    margins = []
    for step_dict in y_star_steps:
        try:
            lps = np.array([float(v.logprob) for v in step_dict.values()], dtype=float)
            lps = lps[np.isfinite(lps)]
            if lps.size == 0:
                continue
            if lps.size == 1:
                p1 = float(np.exp(lps[0]))
                margins.append(max(p1 - 0.0, 0.0))
            else:
                # 取前两大 logprob -> 概率差
                idx = np.argpartition(-lps, 1)[:2]
                top2 = np.sort(lps[idx])[::-1]  # [lp1, lp2] 降序排列
                p1, p2 = float(np.exp(top2[0])), float(np.exp(top2[1]))
                margins.append(max(p1 - p2, 0.0))
        except Exception:
            continue

    if not margins:
        return 0.0, 0.0, []
    return float(np.min(margins)), float(np.mean(margins)), margins



# -----------------------------
# 4) MSP (Maximum Softmax Probability) 不确定性评分
# -----------------------------
def calculate_msp_scores(
    beam_logprobs_list: List[List[Dict[int, Any]]]
) -> Tuple[float, float, float]:

    y_star_steps = beam_logprobs_list[0]
    if not y_star_steps:
        return 1.0, 0.0, 0.0

    # 计算序列的对数似然
    logps = []
    for step_dict in y_star_steps:
        try:
            lp = float(next(iter(step_dict.values())).logprob)
            if np.isfinite(lp):
                logps.append(lp)
        except Exception:
            continue

    if not logps:
        return 1.0, 0.0, 0.0

    # 计算总的对数似然
    log_likelihood = float(np.sum(logps))
    
    # 计算序列概率 p(y*|x) = exp(log_likelihood)
    # 使用数值稳定的方法避免下溢
    probability = float(np.exp(log_likelihood))
    probability = float(np.clip(probability, 0.0, 1.0))
    
    
    return log_likelihood, probability



def calculate_token_entropy_scores(
    beam_logprobs_list: List[List[Dict[int, Any]]],
    *,
    eps: float = 1e-12,
) -> Tuple[float, float, List[float]]:
    """
    基于第一条 beam（y*）计算每一步的 token-level 熵:
        H_t = -sum_j p_j * log p_j
    其中 p_j 为该步 token 分布（对提供的候选做 logsumexp 归一化）。
    
    返回:
        (max_entropy_over_steps, mean_entropy_over_steps, entropies_per_step)
    说明:
        - 若某步只包含一个有限的 logprob, 则该步熵为 0.0
        - 若某步没有有限值，跳过该步
        - 若整条序列没有可用步，则返回 (0.0, 0.0, [])
    """
    # 验证输入

    y_star_steps = beam_logprobs_list[0]
    if not y_star_steps:
        return 0.0, 0.0, []

    entropies: List[float] = []

    for step_dict in y_star_steps:
        # 收集该步的 logprob 值
        try:
            lps = np.array([float(v.logprob) for v in step_dict.values()], dtype=float)
        except Exception:
            # 结构异常直接跳过该步
            continue

        # 过滤非有限值
        mask = np.isfinite(lps)
        if not np.any(mask):
            continue

        lps = lps[mask]

        # 若只有一个候选，熵为 0
        if lps.size == 1:
            entropies.append(0.0)
            continue

        # 使用 logsumexp 做数值稳定的归一化
        # p = softmax(lps)；H = -sum p * log p
        lse = np.log(np.sum(np.exp(lps - np.max(lps)))) + np.max(lps)   # logsumexp
        probs = np.exp(lps - lse)                                       # 归一化的概率
        probs = np.clip(probs, 0.0, 1.0)
        probs_sum = np.sum(probs)
        if probs_sum <= eps:
            # 极端情况下保护
            continue
        probs = probs / probs_sum

        # H = -sum p * log p
        # 用 log(probs + eps) 防止 log(0)
        H = -float(np.sum(probs * np.log(probs + eps)))
        entropies.append(H)

    if not entropies:
        return 0.0, 0.0, []

    entropies = list(map(float, entropies))
    max_entropy = float(np.max(entropies))
    mean_entropy = float(np.mean(entropies))

    return max_entropy, mean_entropy, entropies