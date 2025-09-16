from __future__ import annotations
from typing import List, Optional, Any
import math
import numpy as np

# 仅用于 conf 策略（logprob 置信），沿用你项目里现成的统计函数
try:
    from sal.utils.score import calculate_confidence_score
except Exception:
    calculate_confidence_score = None


class BaseScorer:
    """
    统一接口：返回 Q × N × Steps 的“步级”分数（越大越好）。
    Q: 问题数量；N: 每题候选数；Steps: 逐轮 append 的步数。
    """
    name: str = "base"
    requires_vllm_logprobs: bool = False

    def score_stepwise(
        self,
        questions: List[str],
        candidates: List[List[str]],
        *,
        tokenizer=None,
        vllm_responses: Optional[List[Any]] = None,
        slm=None,
        lookahead: int = 1,
        stop: Optional[List[str]] = None,
    ) -> List[List[List[float]]]:
        raise NotImplementedError


# ---------- 1) PRM 适配器：直接复用你已有 PRM ----------
class PRMScorer(BaseScorer):
    name = "prm"
    def __init__(self, prm):
        self.prm = prm

    def score_stepwise(self, questions, candidates, **_):
        # 你的 PRM.score(questions, outputs) 返回的就是 Q × N × Steps
        return self.prm.score(questions, candidates)


# ---------- 2) Logprob 置信度（轻量在线） ----------
class ConfScorer(BaseScorer):
    """
    使用 vLLM 的 logprobs 对“本轮新增 step”打一个分（越大越好）。
    支持 conf_strategy: probs_mean / log_sum / log_mean
    （保持与你现有项目里的策略名兼容）
    """
    name = "conf"
    requires_vllm_logprobs = True

    def __init__(self, mode: str = "probs_mean"):
        self.mode = mode

    def _score_from_vllm_output(self, out_any) -> float:
        """
        与你之前代码兼容：调用 sal.utils.score.calculate_confidence_score(logprobs)
        该函数通常返回 [log_sum, log_mean, probs_mean]，这里按 mode 取用。
        """
        if calculate_confidence_score is None:
            # 兜底：没有该函数时，用简化的 token 概率均值（如果结构允许）
            # 期望 out.outputs[0].logprobs 是一个 token 列表，元素含 "logprob"
            try:
                toks = out_any.outputs[0].logprobs
                probs = np.exp([t["logprob"] for t in toks])
                return float(np.mean(probs))
            except Exception:
                return 0.0

        # 标准路径：直接用你项目里的打分函数
        out0 = out_any.outputs[0]
        stats = calculate_confidence_score(out0.logprobs)  # -> [log_sum, log_mean, probs_mean]
        # 兼容不同实现：长度不足时做兜底
        if not isinstance(stats, (list, tuple)) or len(stats) == 0:
            return 0.0

        if self.mode in ("log_sum", "logsum"):
            return float(stats[0])
        if self.mode in ("log_mean", "logmean"):
            # 有些实现下标 1 才是 log_mean
            idx = 1 if len(stats) > 1 else 0
            return float(stats[idx])
        # 默认 probs_mean（通常在最后一位）
        return float(stats[-1])

    def score_stepwise(self, questions, candidates, *, vllm_responses=None, **_):
        assert vllm_responses is not None, "ConfScorer 需要 vllm_responses（logprobs=True）"
        # vllm_responses 的展平顺序应与 candidates 展平一致（你的 generate_* 已保证）
        scores_qns: List[List[List[float]]] = []
        it = iter(vllm_responses)
        for cand_list in candidates:
            per_q: List[List[float]] = []
            for _ in range(len(cand_list)):
                resp = next(it)
                s = self._score_from_vllm_output(resp)
                per_q.append([float(s)])   # 本轮仅追加 1 个分
            scores_qns.append(per_q)
        return scores_qns


# ---------- 3) 语义熵 / 采样一致性 SSE（中成本） ----------
class SemanticEntropyScorer(BaseScorer):
    """
    对“当前步”做 S 次短采样（只生成到 stop），取样本文本做句向量聚类，计算归一化熵 H_norm；
    分数 = 1 - H_norm （越大越一致/越可靠）。
    依赖 sentence-transformers（延迟加载）。
    """
    name = "sse"
    requires_vllm_logprobs = False

    def __init__(
        self,
        samples: int = 6,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sim_threshold: float = 0.85,
        max_step_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        self.samples = samples
        self.embed_model = embed_model
        self.sim_threshold = sim_threshold
        self.max_step_tokens = max_step_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._embedder = None

    def _lazy_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embed_model)
        return self._embedder
    
    def _cluster_entropy(self, embs: np.ndarray) -> float:
        """基于相似度阈值的连通分量聚类，返回归一化熵 H/ln(K)。"""
        if len(embs) == 1:
            return 0.0
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        sims = (embs @ embs.T)

        n = sims.shape[0]
        visited = [False] * n
        clusters = []
        for i in range(n):
            if visited[i]:
                continue
            cluster = [i]
            visited[i] = True
            changed = True
            while changed:
                changed = False
                for j in range(n):
                    if visited[j]:
                        continue
                    if np.any(sims[j, cluster] >= self.sim_threshold):
                        visited[j] = True
                        cluster.append(j)
                        changed = True
            clusters.append(cluster)

        freqs = np.array([len(c) for c in clusters], dtype=np.float64)
        p = freqs / np.sum(freqs)
        H = -(p * np.log(p + 1e-12)).sum()
        H_norm = H / (math.log(len(freqs)) + 1e-12)
        # ---- 新增：数值钳制，保证在 [0,1]
        H_norm = float(min(max(H_norm, 0.0), 1.0))
        return H_norm

    def score_stepwise(
        self,
        questions: List[str],
        candidates: List[List[str]],
        *,
        tokenizer=None,
        vllm_responses=None,
        slm=None,
        lookahead: int = 1,
        stop: Optional[List[str]] = None,
    ) -> List[List[List[float]]]:
        assert slm is not None, "SemanticEntropyScorer 需要 slm 做短采样"
        from vllm import SamplingParams

        # candidates 里放的是“模板后的完整上下文字符串”（建议在上游按此传入）
        flat_ctx: List[str] = []
        for cand_list in candidates:
            flat_ctx.extend(cand_list)

        sp = SamplingParams(
            n=self.samples,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_step_tokens,
            stop=stop or ["\n\n"],
            include_stop_str_in_output=True,
        )
        outs = slm.generate(flat_ctx, sp, use_tqdm=False)

        # 收集每个候选的样本文本
        step_texts_all: List[List[str]] = []
        for out in outs:
            texts = [o.text for o in out.outputs]
            step_texts_all.append(texts)

        # 嵌入并计算一致性分
        embedder = self._lazy_embedder()
        step_scores = []
        for texts in step_texts_all:
            embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
            Hn = self._cluster_entropy(embs)
            score = 1.0 - Hn  # 越大越好
            # ---- 新增：再次钳制，确保写入 all_scores 的值在 [0,1]
            score = float(min(max(score, 0.0), 1.0))
            step_scores.append(score)

        # 还原为 Q × N × [1]
        scores_qns: List[List[List[float]]] = []
        idx = 0
        for cand_list in candidates:
            per_q = []
            for _ in range(len(cand_list)):
                per_q.append([step_scores[idx]])
                idx += 1
            scores_qns.append(per_q)
        return scores_qns
