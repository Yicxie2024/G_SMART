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


import itertools
import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
    
    # Handle empty list case
    if not answer_tokens_logprobs_list or len(answer_tokens_logprobs_list) == 0:
        # Return default values for empty list
        return 1.0, 1.0, 1.0
    
    log_likelihood_of_completion = sum(next(iter(logprob.values())).logprob for logprob in answer_tokens_logprobs_list)
    
    # Standard perplexity: exp(-log_likelihood)
    perplexity_score = np.exp(-log_likelihood_of_completion)
    
    T = len(answer_tokens_logprobs_list)
    # Normalized perplexity: exp(-log_likelihood / T)
    # T should never be 0 here due to the check above, but add safeguard
    if T == 0:
        normalized_perplexity_score = 1.0
    else:
        normalized_perplexity_score = np.exp(-log_likelihood_of_completion / T)
    
    # Token-level perplexity: exp(-mean(log_prob_per_token))
    token_logprobs = [next(iter(logprob.values())).logprob for logprob in answer_tokens_logprobs_list]
    if token_logprobs:
        token_perplexity_score = np.exp(-np.mean(token_logprobs))
    else:
        token_perplexity_score = 1.0
    
    return perplexity_score, normalized_perplexity_score, token_perplexity_score


def aggregate_scores(
    scores: list[float], agg_strategy: str
) -> float:
    # Handle case where scores is already a single float (already aggregated)
    if isinstance(scores, (int, float)):
        return float(scores)
    
    # Handle empty list
    if not scores:
        return 0.0
    
    if agg_strategy == "min":
        return min(scores)
    elif agg_strategy == "prod":
        return math.prod(scores)
    elif agg_strategy == "last":
        return scores[-1]
    else:
        raise ValueError(f"Invalid aggregation strategy: {agg_strategy}")


def score(dataset: Dataset, config: Config) -> Dataset:
    dataset = dataset.map(
        lambda x: {"agg_scores": [aggregate_scores(s, "last") for s in x["scores"]]}
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

def calculate_cocoa_uq_scores(
    beam_logprobs_list: List[List[Dict[int, Any]]],
    *,
    detok: Callable[[List[int]], str],           # List[int] -> str
    embed_fn: Callable[[List[str]], np.ndarray], # List[str] -> np.ndarray, shape (B, D)
    eps: float = 1e-12,
) -> Tuple[float, float, float, float]:
    """
    CoCoA-style scores for the first beam (y*):
      - MSP (paper):  u_msp     = 1 - exp(sum_t log p_t) = 1 - p(y*|x)
      - PPL:          u_ppl     = -mean_t log p_t
      - Entropy:      u_entropy = mean_t H_t,  H_t = -sum(lp * exp(lp)) per step
      - Confidence:   c_conf    = 1 - p(y*|x) (same base as msp, treated as uncertainty)

    Each base u is multiplied by semantic inconsistency:
      mean_i (1 - cosine_sim(embed(y*), embed(y^i))) for i >= 1 (exclude self).
    
    Note: cocoa_confidence uses the same calculation as cocoa_msp:
      cocoa_confidence = (1 - p(y*|x)) * mean_dissim = cocoa_msp
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
        p_star = 0.0

    # Confidence base: use 1 - p(y*|x) same as msp (so base_confidence = base_msp)
    # This gives us uncertainty (not confidence), which we multiply by semantic inconsistency
    base_confidence = float(base_msp)  # 1 - p(y*|x) = base_msp

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
    
    # Confidence: same as msp - use 1 - p(y*|x) multiplied by semantic inconsistency
    # cocoa_confidence = (1 - p(y*|x)) * mean_dissim = base_confidence * mean_dissim
    cocoa_confidence = float(base_confidence * mean_dissim)

    return cocoa_msp, cocoa_ppl, cocoa_entropy, cocoa_confidence


# -----------------------------
# TokenSAR: Token Similarity and Relevance
# -----------------------------
def calculate_token_similarity(
    token_ids: List[int],
    input_text: str,
    tokenizer,
    crossencoder,
    special_tokens: List[int] = None
) -> np.ndarray:
    """
    计算每个 token 的相似度（留一法 leave-one-out）
    
    Args:
        token_ids: 生成的 token IDs
        input_text: 输入文本（prompt）
        tokenizer: tokenizer
        crossencoder: CrossEncoder 模型
        special_tokens: 特殊 token IDs 列表
        
    Returns:
        np.ndarray: 每个 token 的相似度分数 [0, 1]
    """
    if len(token_ids) <= 1:
        return np.array([0.5] * len(token_ids))
    
    # 处理特殊 tokens
    if special_tokens is None:
        special_tokens = []
    is_special_tokens = np.isin(token_ids, special_tokens)
    
    # 生成留一法序列（每次去掉一个 token）
    cropped_tokens = list(itertools.combinations(token_ids, len(token_ids) - 1))[::-1]
    
    # 完整文本
    raw_text = input_text + " " + tokenizer.decode(token_ids, skip_special_tokens=True)
    
    # 构建相似度计算对：(完整文本, 去掉第i个token的文本)
    batches = [
        (
            raw_text,
            input_text + " " + tokenizer.decode(list(t), skip_special_tokens=True)
        )
        for t in cropped_tokens
    ]
    
    # 使用 CrossEncoder 计算相似度
    token_scores = crossencoder.predict(batches, batch_size=10)
    token_scores = np.asarray(token_scores, dtype=float)

    if token_scores.ndim == 2:
        # 多类别输出（例如 NLI 模型），转换为概率并选取“蕴含”分数作为相似度
        logits = token_scores
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-12)
        entailment_idx = 2 if probs.shape[1] >= 3 else probs.shape[1] - 1
        token_scores = probs[:, entailment_idx]

    token_scores = np.clip(token_scores, 0.0, 1.0)
    
    # 特殊 tokens 设为高相似度（不重要）
    token_scores[is_special_tokens] = 1.0
    
    return token_scores


def calculate_token_sar_score(
    logprobs_list: List[Dict[int, Any]],
    token_similarity: np.ndarray
) -> float:
    """
    计算 TokenSAR (Token Semantic Alignment and Relevance) 分数
    
    基于论文: https://arxiv.org/abs/2307.01379
    
    Args:
        logprobs_list: token logprobs (from vLLM output)
        token_similarity: 每个 token 的相似度分数
        
    Returns:
        float: TokenSAR 不确定性分数（越高越不确定）
    """
    if not logprobs_list:
        return 0.0
    
    # 提取 log likelihoods
    log_likelihoods = []
    for lp in logprobs_list:
        try:
            ll = float(next(iter(lp.values())).logprob)
            if np.isfinite(ll):
                log_likelihoods.append(ll)
        except Exception:
            continue
    
    if not log_likelihoods:
        return 0.0
    
    log_likelihoods = np.array(log_likelihoods)
    
    # 确保 token_similarity 长度匹配
    if len(token_similarity) != len(log_likelihoods):
        # 如果长度不匹配，截断或填充
        min_len = min(len(token_similarity), len(log_likelihoods))
        token_similarity = token_similarity[:min_len]
        log_likelihoods = log_likelihoods[:min_len]
    
    # TokenSAR 计算
    R_t = 1 - token_similarity  # 相关性权重（低相似度 = 高权重）
    R_t_norm = R_t / (R_t.sum() + 1e-12)  # 归一化
    E_t = -log_likelihoods * R_t_norm  # 加权不确定性
    
    return float(E_t.sum())


# -----------------------------
# Token-SAR + Confidence + Margin（融合版）
# -----------------------------
def combine_token_sar_conf_margin(
    beam_logprobs_list: List[List[Dict[int, Any]]],
    token_similarity: np.ndarray,
    *,
    # —— 通用 ——
    skip_token_ids: Optional[Set[int]] = None,
    tail_ratio: float = 0.2,
    # —— confidence 配置 ——
    conf_mode: str = "one_minus_p",  # 'one_minus_p' | 'nll'
    # —— margin 配置 ——
    margin_space: str = "logit",  # 'logit' | 'prob'
    margin_robust_q: Optional[float] = 0.10,
    min_gap_fallback: float = 1e-3,
    tie_metric: str = "logit",  # 'logit' | 'prob'
    near_tie_delta: float = 0.5,
    # —— SAR 配置 ——
    use_entropy: bool = True,
    sim_norm: str = "z",  # 'z' | 'minmax' | 'none'
    sar_temp: float = 1.0,
    # —— 短句回退 ——
    short_token_threshold: int = 8,
    # —— 融合权重 ——
    w_sar: float = 1.0,
    w_conf: float = 1.0,
    w_margin: float = 0.4,
    # —— 归一化 ——
    normalize: str = "z",  # 'z' | 'minmax' | 'none'
) -> dict:
    """融合 Token-SAR、confidence 与 top-2 margin 的不确定性分数。

    返回的 ``final_score`` 值越大表示越不确定。
    """

    token_similarity_arr = (
        np.asarray(token_similarity, dtype=float)
        if token_similarity is not None
        else np.empty(0, dtype=float)
    )

    if not beam_logprobs_list or not beam_logprobs_list[0]:
        return {
            "final_score": 0.0,
            "mode": "empty",
            "sar": {},
            "margin": {},
            "conf": {},
        }

    def _tail_slice(arr: np.ndarray) -> Tuple[np.ndarray, int]:
        if arr.size == 0:
            return arr, 0
        if 0.0 < tail_ratio < 1.0 and arr.size >= 2:
            k = max(1, int(round(arr.size * tail_ratio)))
            start = arr.size - k
            return arr[start:], start
        return arr, 0

    def _safe_z(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        mu = float(np.mean(x))
        sd = float(np.std(x))
        if not np.isfinite(sd) or sd < 1e-12:
            return np.zeros_like(x)
        return (x - mu) / (sd + 1e-12)

    def _safe_minmax(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        lo = float(np.min(x))
        hi = float(np.max(x))
        if not (np.isfinite(lo) and np.isfinite(hi)) or (hi - lo) < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo + 1e-12)

    def _norm(arr: np.ndarray) -> np.ndarray:
        if normalize == "z":
            return _safe_z(arr)
        if normalize == "minmax":
            return _safe_minmax(arr)
        return arr

    def _norm_scalar(val: float, *, fallback: bool = True) -> float:
        """Apply the same normalization as ``_norm`` to a scalar value.

        When there is only a single sample, both z-score and min-max
        normalization would degenerate to zero because the variance or range
        collapses. This in turn flattened every component score to ``0.0`` and
        made the final uncertainty score uninformative. To avoid that, we fall
        back to the raw value whenever the normalized result is (close to) all
        zeros.
        """

        arr = np.asarray([val], dtype=float)
        normed = _norm(arr)

        if fallback and normed.size and np.allclose(normed, 0.0, atol=1e-9):
            return float(val)

        return float(normed[0]) if normed.size else float(val)

    flat_cands: List[Dict[int, Any]] = list(beam_logprobs_list[0])
    T_total = len(flat_cands)

    ps_list: List[np.ndarray] = []
    top1_list: List[float] = []
    logit_gap_list: List[float] = []
    prob_gap_list: List[float] = []

    for cand in flat_cands:
        filtered = {
            tid: obj
            for tid, obj in cand.items()
            if not (skip_token_ids and tid in skip_token_ids)
        }
        if not filtered:
            filtered = cand  # fallback: 保留原候选，避免整个位被丢弃

        lps = [float(getattr(v, "logprob", np.nan)) for v in filtered.values()]
        lps = np.asarray([x for x in lps if np.isfinite(x)], dtype=float)
        if lps.size == 0:
            continue

        lps_norm = lps - np.max(lps)
        ps = np.exp(lps_norm)
        ps /= np.sum(ps) + 1e-12

        ps_list.append(ps)
        top1 = float(ps.max())
        top1_list.append(top1)

        if ps.size >= 2:
            idx_prob = np.argpartition(-ps, 1)[:2]
            p1, p2 = float(ps[idx_prob[0]]), float(ps[idx_prob[1]])
            prob_gap_list.append(max(p1 - p2, 0.0))

            idx_lp = np.argpartition(-lps_norm, 1)[:2]
            lp1, lp2 = float(lps_norm[idx_lp[0]]), float(lps_norm[idx_lp[1]])
            logit_gap_list.append(max(lp1 - lp2, min_gap_fallback))
        else:
            prob_gap_list.append(1.0)
            logit_gap_list.append(max(min_gap_fallback, 0.0))

    if not ps_list:
        return {
            "final_score": 0.0,
            "mode": "conf_fallback",
            "sar": {},
            "margin": {},
            "conf": {},
        }

    p_top1 = np.asarray(top1_list, dtype=float)
    logit_gap = np.asarray(logit_gap_list, dtype=float)
    prob_gap = np.asarray(prob_gap_list, dtype=float)

    conf_tokens = 1.0 - p_top1 if conf_mode == "one_minus_p" else -np.log(np.clip(p_top1, 1e-12, 1.0))
    conf_tail_vals, _ = _tail_slice(conf_tokens)
    conf_tail = float(np.mean(conf_tail_vals)) if conf_tail_vals.size else 0.0

    margin_arr = logit_gap if margin_space == "logit" else prob_gap
    margin_tail_vals, _ = _tail_slice(margin_arr)
    if margin_tail_vals.size:
        m_mean = float(np.mean(margin_tail_vals))
        if margin_robust_q is not None:
            m_robust = float(np.quantile(margin_tail_vals, margin_robust_q))
        else:
            m_robust = m_mean
    else:
        m_mean = 0.0
        m_robust = 0.0

    tie_arr = logit_gap if tie_metric == "logit" else prob_gap
    tie_tail_vals, _ = _tail_slice(tie_arr)
    tie_metric_val = float(np.mean(tie_tail_vals)) if tie_tail_vals.size else 0.0
    near_tie = tie_metric_val <= near_tie_delta

    if T_total <= short_token_threshold:
        final_score = w_conf * _norm_scalar(conf_tail)
        return {
            "final_score": float(final_score),
            "mode": "conf",
            "sar": {},
            "margin": {
                "per_step": margin_arr.tolist(),
                "robust": m_robust,
                "mean": m_mean,
                "space": margin_space,
            },
            "conf": {
                "tail_mean": conf_tail,
                "mode": conf_mode,
            },
        }

    U = []
    for ps in ps_list:
        if use_entropy and ps.size >= 2:
            U_t = -np.sum(ps * np.log(ps + 1e-12))
        else:
            U_t = 1.0 - float(ps.max())
        U.append(float(U_t))
    U = np.asarray(U, dtype=float)

    if token_similarity_arr.size == 0:
        sim = np.zeros_like(U)
    else:
        sim = token_similarity_arr[: U.size]
        if sim.size < U.size:
            fill_value = float(np.mean(sim)) if sim.size else 0.0
            sim = np.pad(sim, (0, U.size - sim.size), constant_values=fill_value)

    if sim_norm == "z":
        sim_scaled = _safe_z(sim)
        rel = 1.0 - 1.0 / (1.0 + np.exp(-sim_scaled))
    elif sim_norm == "minmax":
        sim_scaled = _safe_minmax(sim)
        rel = 1.0 - sim_scaled
    else:
        rel = 1.0 - sim

    weights = np.exp(rel / max(sar_temp, 1e-6))
    weights /= np.sum(weights) + 1e-12

    U_tail, tail_start = _tail_slice(U)
    if U_tail.size == 0:
        U_tail = U
        tail_start = 0

    w_tail = weights[tail_start:]
    if w_tail.size == 0:
        w_tail = weights
    w_tail = w_tail / (np.sum(w_tail) + 1e-12)

    sar_step = float(np.sum(w_tail * U_tail)) if U_tail.size else 0.0

    sar_s = _norm_scalar(sar_step)
    conf_s = _norm_scalar(conf_tail)
    margin_s = _norm_scalar(m_robust) if margin_tail_vals.size else 0.0

    if near_tie:
        final_score = w_sar * sar_s + w_conf * conf_s - w_margin * margin_s
        mode = "sar+conf+margin"
    else:
        final_score = w_sar * sar_s + w_conf * conf_s
        mode = "sar+conf"

    return {
        "final_score": float(final_score),
        "mode": mode,
        "sar": {
            "per_token_U": U_tail.tolist(),
            "per_token_w": w_tail.tolist(),
            "step_score": sar_step,
        },
        "margin": {
            "per_step": margin_arr.tolist(),
            "robust": m_robust,
            "mean": m_mean,
            "space": margin_space,
        },
        "conf": {
            "tail_mean": conf_tail,
            "mode": conf_mode,
        },
    }


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