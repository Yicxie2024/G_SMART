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
from typing import Literal

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
        tuple: (log_likelihood, likelihood)
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
    
    return [likelihood_score, likelihood_mean_score, probs_mean_score]


def aggregate_scores(
    scores: list[float], agg_strategy: Literal["min", "prod", "last"]
) -> float:
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
    if not beam_logprobs_list or not isinstance(beam_logprobs_list[0], list):
        return 0.0, 0.0, 0.0

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
