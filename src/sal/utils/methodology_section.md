# Methodology

## 3.1 Overview

This section presents our comprehensive framework for quantifying uncertainty in large language model (LLM) generations. Our approach integrates multiple uncertainty estimation methods, including confidence-based metrics, perplexity measures, semantic similarity analysis, and entropy-based indicators. The framework is designed to provide robust uncertainty quantification for guiding routing decisions in multi-model inference systems.

## 3.2 Uncertainty Quantification Framework

### 3.2.1 Confidence-Based Metrics

We begin by computing confidence scores based on the likelihood of generated sequences. Given a sequence of tokens $y = \{y_1, y_2, \ldots, y_T\}$ with corresponding log-probabilities $\{\log p_1, \log p_2, \ldots, \log p_T\}$, we compute three variants of confidence scores:

**Total Likelihood:**
$$L_{\text{total}} = \exp\left(\sum_{t=1}^{T} \log p_t\right) = \prod_{t=1}^{T} p_t$$

**Normalized Mean Likelihood:**
$$L_{\text{mean}} = \exp\left(\frac{1}{T} \sum_{t=1}^{T} \log p_t\right) = \left(\prod_{t=1}^{T} p_t\right)^{1/T}$$

**Mean Probability Score:**
$$P_{\text{mean}} = \frac{1}{T} \sum_{t=1}^{T} p_t$$

The normalized mean likelihood provides a length-invariant measure, while the mean probability score offers a more intuitive interpretation as the average token probability.

### 3.2.2 Perplexity-Based Uncertainty

Perplexity serves as a fundamental measure of model uncertainty. We compute three variants:

**Standard Perplexity:**
$$\text{PP} = \exp\left(-\sum_{t=1}^{T} \log p_t\right) = \frac{1}{L_{\text{total}}}$$

**Normalized Perplexity:**
$$\text{PP}_{\text{norm}} = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log p_t\right) = \text{PP}^{1/T}$$

**Token-Level Perplexity:**
$$\text{PP}_{\text{token}} = \exp\left(-\mathbb{E}[\log p_t]\right)$$

Normalized perplexity accounts for sequence length, making it comparable across different generation lengths.

### 3.2.3 Maximum Softmax Probability (MSP)

The MSP metric quantifies uncertainty as the complement of the sequence probability:

$$u_{\text{msp}} = 1 - p(y^*|x) = 1 - \exp\left(\sum_{t=1}^{T} \log p_t(y^*)\right)$$

where $y^*$ denotes the best sequence (first beam) and $x$ is the input prompt. Higher MSP values indicate greater uncertainty.

## 3.3 Semantic-Aware Uncertainty Quantification

### 3.3.1 CoCoA-Style Uncertainty Scores

To incorporate semantic consistency across multiple generations, we extend base uncertainty metrics with semantic inconsistency measures. Given $B$ beam search candidates $\{y^*, y^2, \ldots, y^B\}$, we compute:

**Semantic Dissimilarity:**
For each beam $y^i$ ($i \geq 2$), we compute cosine similarity with the best sequence $y^*$:

$$\text{sim}(y^*, y^i) = \frac{\text{embed}(y^*) \cdot \text{embed}(y^i)}{\|\text{embed}(y^*)\| \cdot \|\text{embed}(y^i)\|}$$

The normalized similarity is mapped to $[0,1]$:
$$\text{sim}_{01}(y^*, y^i) = \frac{\text{sim}(y^*, y^i) + 1}{2}$$

The mean dissimilarity across all beams is:
$$\text{mean\_dissim} = \frac{1}{B-1} \sum_{i=2}^{B} (1 - \text{sim}_{01}(y^*, y^i))$$

**CoCoA-Enriched Scores:**
We multiply base uncertainty scores by semantic dissimilarity:

$$\text{cocoa}_{\text{msp}} = (1 - p(y^*|x)) \times \text{mean\_dissim}$$

$$\text{cocoa}_{\text{ppl}} = \left(-\frac{1}{T} \sum_{t=1}^{T} \log p_t(y^*)\right) \times \text{mean\_dissim}$$

$$\text{cocoa}_{\text{entropy}} = \left(\frac{1}{T} \sum_{t=1}^{T} H_t\right) \times \text{mean\_dissim}$$

where $H_t$ is the entropy at step $t$:
$$H_t = -\sum_{j} \log p_{t,j} \cdot \exp(\log p_{t,j})$$

### 3.3.2 Token-Level Semantic Similarity

To assess the importance of individual tokens, we employ a leave-one-out (LOO) approach. For a sequence of length $T$, we generate $T$ cropped sequences:

$$y_{-t} = \{y_1, y_2, \ldots, y_{t-1}, y_{t+1}, \ldots, y_T\}$$

Using a CrossEncoder model, we compute the semantic similarity between the full sequence and each cropped variant:

$$\text{sim}_t = \text{CrossEncoder}(x \oplus y, x \oplus y_{-t})$$

If the CrossEncoder outputs multi-class logits (e.g., NLI), we extract the entailment probability:

$$p_{\text{entail},t} = \frac{\exp(\text{logit}_{\text{entail},t})}{\sum_{c} \exp(\text{logit}_{c,t})}$$

This yields a token-level similarity vector $\mathbf{sim} \in [0,1]^T$, where higher values indicate that removing the token has less impact on semantic meaning.

### 3.3.3 TokenSAR: Token Semantic Alignment and Relevance

Building on token similarity, we compute the TokenSAR (Token Semantic Alignment and Relevance) score, which weights uncertainty by semantic relevance:

**Relevance Weight:**
$$R_t = 1 - \text{sim}_t$$

Lower similarity corresponds to higher relevance weight, as semantically important tokens should have lower similarity when removed.

**Normalized Weight:**
$$R_{t,\text{norm}} = \frac{R_t}{\sum_{t=1}^{T} R_t + \epsilon}$$

**TokenSAR Score:**
$$\text{TokenSAR} = -\sum_{t=1}^{T} \log p_t \cdot R_{t,\text{norm}}$$

This formulation emphasizes uncertainty in semantically critical tokens, providing a more nuanced measure than uniform weighting.

## 3.4 Integrated Uncertainty Scoring

### 3.4.1 Multi-Component Fusion

We propose a unified uncertainty score that combines Token-SAR, confidence, and top-2 margin metrics. The fusion strategy adapts based on sequence characteristics.

**Confidence Component:**
We compute per-token confidence using either:
- **Mode 1 (one_minus_p):** $\text{conf}_t = 1 - p_{\text{top1},t}$
- **Mode 2 (nll):** $\text{conf}_t = -\log(p_{\text{top1},t})$

The tail average (last $k$ tokens, where $k = \lfloor T \times \text{tail\_ratio} \rfloor$) is:
$$\text{conf}_{\text{tail}} = \frac{1}{k} \sum_{t=T-k+1}^{T} \text{conf}_t$$

**Margin Component:**
The top-2 margin quantifies decision confidence at each step:

$$\text{margin}_t = \max(p_{1,t} - p_{2,t}, 0)$$

where $p_{1,t}$ and $p_{2,t}$ are the top-2 probabilities at step $t$. We compute a robust statistic (e.g., 10th percentile) over the tail:
$$\text{margin}_{\text{robust}} = Q_{0.10}(\{\text{margin}_t\}_{t=T-k+1}^{T})$$

**Token-SAR Component:**
For each step $t$, we compute uncertainty $U_t$:

$$U_t = \begin{cases}
-\sum_{j} p_{j,t} \log(p_{j,t} + \epsilon) & \text{if entropy mode} \\
1 - p_{\text{top1},t} & \text{otherwise}
\end{cases}$$

We then compute relevance weights from token similarity. The similarity vector is normalized using one of three methods:

- **Z-score normalization:**
  $$\text{sim}_{\text{scaled}} = \frac{\text{sim} - \mu_{\text{sim}}}{\sigma_{\text{sim}}}$$
  $$\text{rel} = 1 - \frac{1}{1 + \exp(-\text{sim}_{\text{scaled}})}$$

- **Min-max normalization:**
  $$\text{sim}_{\text{scaled}} = \frac{\text{sim} - \min(\text{sim})}{\max(\text{sim}) - \min(\text{sim})}$$
  $$\text{rel} = 1 - \text{sim}_{\text{scaled}}$$

- **No normalization:**
  $$\text{rel} = 1 - \text{sim}$$

The weights are computed using a temperature-scaled softmax:
$$w_t = \frac{\exp(\text{rel}_t / \tau)}{\sum_{t=1}^{T} \exp(\text{rel}_t / \tau)}$$

where $\tau$ is the temperature parameter. The SAR score is:
$$\text{sar} = \sum_{t=T-k+1}^{T} w_t \cdot U_t$$

### 3.4.2 Final Score Computation

After normalizing each component (using z-score or min-max normalization), we compute the final uncertainty score:

**Near-Tie Detection:**
We detect when the model is uncertain between top candidates:
$$\text{near\_tie} = \begin{cases}
\text{True} & \text{if } \text{mean}(\text{margin}_{\text{tail}}) \leq \delta \\
\text{False} & \text{otherwise}
\end{cases}$$

**Final Score:**
$$\text{final\_score} = \begin{cases}
w_{\text{sar}} \cdot \text{sar}_s + w_{\text{conf}} \cdot \text{conf}_s - w_{\text{margin}} \cdot \text{margin}_s & \text{if near\_tie} \\
w_{\text{sar}} \cdot \text{sar}_s + w_{\text{conf}} \cdot \text{conf}_s & \text{otherwise}
\end{cases}$$

where $w_{\text{sar}}$, $w_{\text{conf}}$, and $w_{\text{margin}}$ are component weights, and the subscript $s$ denotes normalized scores. The margin term is subtracted (with negative weight) when near-tie is detected, as smaller margins indicate higher uncertainty.

**Short Sequence Fallback:**
For sequences shorter than a threshold (e.g., 8 tokens), we use only the confidence component:
$$\text{final\_score} = w_{\text{conf}} \cdot \text{conf}_s$$

## 3.5 Token-Level Entropy Analysis

To provide fine-grained uncertainty analysis, we compute per-step entropy:

**Probability Normalization:**
Using log-sum-exp for numerical stability:
$$\text{LSE}_t = \log\left(\sum_{j} \exp(\log p_{j,t} - \max_j \log p_{j,t})\right) + \max_j \log p_{j,t}$$

$$p_{j,t} = \frac{\exp(\log p_{j,t} - \text{LSE}_t)}{\sum_{j} \exp(\log p_{j,t} - \text{LSE}_t)}$$

**Entropy Computation:**
$$H_t = -\sum_{j} p_{j,t} \log(p_{j,t} + \epsilon)$$

We aggregate entropy across steps:
$$\text{max\_entropy} = \max_{t} H_t$$
$$\text{mean\_entropy} = \frac{1}{T} \sum_{t=1}^{T} H_t$$

## 3.6 Implementation Details

### 3.6.1 Numerical Stability

All computations employ numerical stability techniques:
- Log-sum-exp trick for probability normalization
- Clipping operations to prevent overflow/underflow
- Epsilon terms ($\epsilon = 10^{-12}$) to avoid division by zero

### 3.6.2 Aggregation Strategies

For multi-step sequences, we support three aggregation strategies:
- **Min:** $s_{\text{agg}} = \min\{s_1, s_2, \ldots, s_n\}$
- **Product:** $s_{\text{agg}} = \prod_{i=1}^{n} s_i$
- **Last:** $s_{\text{agg}} = s_n$

### 3.6.3 Beam Search Integration

Our framework integrates with beam search decoding, utilizing multiple candidate sequences to compute semantic inconsistency and provide more robust uncertainty estimates.

## 3.7 Summary

Our uncertainty quantification framework provides a comprehensive suite of metrics that capture different aspects of model uncertainty:
- **Confidence metrics** measure sequence likelihood
- **Perplexity metrics** quantify prediction difficulty
- **Semantic metrics** assess consistency across generations
- **Entropy metrics** measure distributional uncertainty
- **Integrated scores** combine multiple signals for robust uncertainty estimation

This multi-faceted approach enables more informed routing decisions in multi-model inference systems, where uncertainty estimates guide the selection of appropriate models or strategies for different inputs.

