# score.py 方法定义与数学公式文档

本文档详细介绍了 `score.py` 文件中所有方法的定义、参数、返回值和数学公式。

---

## 1. `calculate_confidence_score`

### 功能描述
计算答案序列的置信度分数，基于 token 的对数概率计算似然度。

### 参数
- `answer_tokens_logprobs_list` (list of dict): 每个 token 的对数概率列表，格式为 `[{token_id: Logprob(logprob=value, ...)}, {...}, ...]`

### 返回值
- `tuple`: `(likelihood_score, likelihood_mean_score, probs_mean_score)`
  - `likelihood_score`: 序列总似然度
  - `likelihood_mean_score`: 归一化平均似然度
  - `probs_mean_score`: 平均概率分数

### 数学公式

设 $T$ 为序列长度，$\log p_t$ 为第 $t$ 个 token 的对数概率：

1. **对数似然度**:
   $$\log p(y|x) = \sum_{t=1}^{T} \log p_t$$

2. **总似然度**:
   $$L_{\text{total}} = \exp\left(\sum_{t=1}^{T} \log p_t\right) = \prod_{t=1}^{T} p_t$$

3. **归一化平均似然度**:
   $$L_{\text{mean}} = \exp\left(\frac{1}{T} \sum_{t=1}^{T} \log p_t\right) = \left(\prod_{t=1}^{T} p_t\right)^{1/T}$$

4. **平均概率分数**:
   $$P_{\text{mean}} = \frac{1}{T} \sum_{t=1}^{T} \exp(\log p_t) = \frac{1}{T} \sum_{t=1}^{T} p_t$$

---

## 2. `calculate_perplexity_score`

### 功能描述
计算序列的困惑度（Perplexity）分数，用于衡量模型对序列的预测不确定性。

### 参数
- `answer_tokens_logprobs_list` (list of dict): 每个 token 的对数概率列表

### 返回值
- `tuple`: `(perplexity_score, normalized_perplexity_score, token_perplexity_score)`
  - `perplexity_score`: 标准困惑度
  - `normalized_perplexity_score`: 归一化困惑度
  - `token_perplexity_score`: token 级困惑度

### 数学公式

1. **对数似然度**:
   $$\log p(y|x) = \sum_{t=1}^{T} \log p_t$$

2. **标准困惑度**:
   $$\text{PP} = \exp(-\log p(y|x)) = \exp\left(-\sum_{t=1}^{T} \log p_t\right) = \frac{1}{L_{\text{total}}}$$

3. **归一化困惑度**:
   $$\text{PP}_{\text{norm}} = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log p_t\right) = \text{PP}^{1/T}$$

4. **Token 级困惑度**:
   $$\text{PP}_{\text{token}} = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log p_t\right) = \exp\left(-\mathbb{E}[\log p_t]\right)$$

---

## 3. `aggregate_scores`

### 功能描述
对分数列表进行聚合操作。

### 参数
- `scores` (list[float]): 分数列表
- `agg_strategy` (str): 聚合策略，可选值：`"min"`, `"prod"`, `"last"`

### 返回值
- `float`: 聚合后的分数

### 数学公式

1. **最小值聚合** (`"min"`):
   $$s_{\text{agg}} = \min\{s_1, s_2, \ldots, s_n\}$$

2. **乘积聚合** (`"prod"`):
   $$s_{\text{agg}} = \prod_{i=1}^{n} s_i$$

3. **最后值聚合** (`"last"`):
   $$s_{\text{agg}} = s_n$$

---

## 4. `score`

### 功能描述
对数据集进行评分处理，计算多数投票、加权预测和朴素预测。

### 参数
- `dataset` (Dataset): HuggingFace 数据集
- `config` (Config): 配置对象

### 返回值
- `Dataset`: 处理后的数据集

### 处理流程
1. 聚合分数（使用 `"last"` 策略）
2. 对子集大小 $n \in \{2^0, 2^1, \ldots, 2^k \leq N\}$ 进行迭代：
   - 子采样完成序列
   - 提取答案
   - 计算加权预测
   - 计算多数投票预测
   - 计算朴素预测

---

## 5. `calculate_cocoa_uq_scores`

### 功能描述
计算 CoCoA 风格的不确定性分数，结合序列概率和语义不一致性。

### 参数
- `beam_logprobs_list` (List[List[Dict[int, Any]]]): Beam search 的对数概率列表
- `detok` (Callable): Token ID 到文本的解码函数
- `embed_fn` (Callable): 文本到嵌入向量的编码函数
- `eps` (float): 数值稳定性参数，默认 `1e-12`

### 返回值
- `Tuple[float, float, float, float]`: `(cocoa_msp, cocoa_ppl, cocoa_entropy, cocoa_confidence)`

### 数学公式

设 $y^*$ 为第一条 beam（最佳序列），$y^i$ 为第 $i$ 条 beam，$T$ 为序列长度。

#### 基础不确定性分数

1. **MSP 基础分数**:
   $$\text{base}_{\text{msp}} = 1 - p(y^*|x) = 1 - \exp\left(\sum_{t=1}^{T} \log p_t(y^*)\right)$$

2. **PPL 基础分数**:
   $$\text{base}_{\text{ppl}} = -\frac{1}{T} \sum_{t=1}^{T} \log p_t(y^*)$$

3. **熵基础分数**:
   $$\text{base}_{\text{entropy}} = \frac{1}{T} \sum_{t=1}^{T} H_t$$
   
   其中每步熵为：
   $$H_t = -\sum_{j} \log p_{t,j} \cdot \exp(\log p_{t,j})$$

4. **置信度基础分数**:
   $$\text{base}_{\text{confidence}} = 1 - p(y^*|x) = \text{base}_{\text{msp}}$$

#### 语义不一致性

5. **余弦相似度**:
   $$\text{sim}(y^*, y^i) = \frac{\text{embed}(y^*) \cdot \text{embed}(y^i)}{\|\text{embed}(y^*)\| \cdot \|\text{embed}(y^i)\|}$$

6. **归一化相似度** (映射到 [0,1]):
   $$\text{sim}_{01}(y^*, y^i) = \frac{\text{sim}(y^*, y^i) + 1}{2}$$

7. **平均不一致性**:
   $$\text{mean\_dissim} = \frac{1}{B-1} \sum_{i=2}^{B} (1 - \text{sim}_{01}(y^*, y^i))$$

   其中 $B$ 为 beam 数量。

#### 最终 CoCoA 分数

8. **CoCoA MSP**:
   $$\text{cocoa}_{\text{msp}} = \text{base}_{\text{msp}} \times \text{mean\_dissim}$$

9. **CoCoA PPL**:
   $$\text{cocoa}_{\text{ppl}} = \text{base}_{\text{ppl}} \times \text{mean\_dissim}$$

10. **CoCoA Entropy**:
    $$\text{cocoa}_{\text{entropy}} = \text{base}_{\text{entropy}} \times \text{mean\_dissim}$$

11. **CoCoA Confidence**:
    $$\text{cocoa}_{\text{confidence}} = \text{base}_{\text{confidence}} \times \text{mean\_dissim} = \text{cocoa}_{\text{msp}}$$

---

## 6. `calculate_token_similarity`

### 功能描述
使用留一法（leave-one-out）计算每个 token 的语义相似度。

### 参数
- `token_ids` (List[int]): 生成的 token ID 序列
- `input_text` (str): 输入文本（prompt）
- `tokenizer`: Tokenizer 对象
- `crossencoder`: CrossEncoder 模型
- `special_tokens` (List[int]): 特殊 token ID 列表

### 返回值
- `np.ndarray`: 每个 token 的相似度分数数组，范围 [0, 1]

### 计算流程

1. **留一法序列生成**:
   对于长度为 $T$ 的序列，生成 $T$ 个裁剪序列，每个序列去掉一个 token：
   $$y_{-t} = \{y_1, y_2, \ldots, y_{t-1}, y_{t+1}, \ldots, y_T\}$$

2. **相似度计算**:
   使用 CrossEncoder 计算完整序列与裁剪序列的相似度：
   $$\text{sim}_t = \text{CrossEncoder}(x \oplus y, x \oplus y_{-t})$$

3. **概率提取**:
   如果 CrossEncoder 输出多类别 logits，使用 softmax 并提取蕴含（entailment）概率：
   $$p_{\text{entail}} = \frac{\exp(\text{logit}_{\text{entail}})}{\sum_{c} \exp(\text{logit}_c)}$$

4. **特殊 token 处理**:
   特殊 token 的相似度设为 1.0（表示不重要）。

---

## 7. `calculate_token_sar_score`

### 功能描述
计算 TokenSAR (Token Semantic Alignment and Relevance) 不确定性分数。

### 参数
- `logprobs_list` (List[Dict[int, Any]]): Token 对数概率列表
- `token_similarity` (np.ndarray): 每个 token 的相似度分数

### 返回值
- `float`: TokenSAR 不确定性分数（值越大越不确定）

### 数学公式

基于论文: https://arxiv.org/abs/2307.01379

1. **相关性权重**:
   $$R_t = 1 - \text{sim}_t$$
   
   其中 $\text{sim}_t$ 为第 $t$ 个 token 的相似度。

2. **归一化权重**:
   $$R_{t,\text{norm}} = \frac{R_t}{\sum_{t=1}^{T} R_t + \epsilon}$$

3. **加权不确定性**:
   $$E_t = -\log p_t \times R_{t,\text{norm}}$$

4. **TokenSAR 总分**:
   $$\text{TokenSAR} = \sum_{t=1}^{T} E_t = -\sum_{t=1}^{T} \log p_t \cdot R_{t,\text{norm}}$$

---

## 8. `combine_token_sar_conf_margin`

### 功能描述
融合 Token-SAR、Confidence 和 Top-2 Margin 的不确定性分数。

### 参数
- `beam_logprobs_list`: Beam search 对数概率列表
- `token_similarity`: Token 相似度数组
- `tail_ratio` (float): 尾部比例，默认 0.2
- `conf_mode` (str): 置信度模式，`"one_minus_p"` 或 `"nll"`
- `margin_space` (str): Margin 空间，`"logit"` 或 `"prob"`
- `margin_robust_q` (float): Margin 鲁棒分位数，默认 0.10
- `use_entropy` (bool): 是否使用熵，默认 True
- `sim_norm` (str): 相似度归一化方法，`"z"`, `"minmax"` 或 `"none"`
- `sar_temp` (float): SAR 温度参数，默认 1.0
- `w_sar`, `w_conf`, `w_margin` (float): 各组件权重
- `normalize` (str): 最终归一化方法

### 返回值
- `dict`: 包含 `final_score`, `mode`, `sar`, `margin`, `conf` 的字典

### 数学公式

#### 1. Confidence 计算

**模式 1: one_minus_p**
$$\text{conf}_t = 1 - p_{\text{top1},t}$$

**模式 2: nll**
$$\text{conf}_t = -\log(p_{\text{top1},t})$$

**尾部平均**:
$$\text{conf}_{\text{tail}} = \frac{1}{k} \sum_{t=T-k+1}^{T} \text{conf}_t$$

其中 $k = \lfloor T \times \text{tail\_ratio} \rfloor$。

#### 2. Margin 计算

**概率空间**:
$$\text{margin}_t = p_{1,t} - p_{2,t}$$

**Logit 空间**:
$$\text{margin}_t = \log p_{1,t} - \log p_{2,t} = \text{logit}_{1,t} - \text{logit}_{2,t}$$

**鲁棒统计**:
$$\text{margin}_{\text{robust}} = Q_{\text{robust\_q}}(\{\text{margin}_t\}_{t=T-k+1}^{T})$$

其中 $Q_q$ 表示分位数函数。

#### 3. Token-SAR 计算

**每步不确定性**:

如果 `use_entropy=True`:
$$U_t = -\sum_{j} p_{j,t} \log(p_{j,t} + \epsilon)$$

否则:
$$U_t = 1 - p_{\text{top1},t}$$

**相关性权重**:
$$R_t = 1 - \text{sim}_t$$

**相似度归一化**:

- `"z"`: 
  $$\text{sim}_{\text{scaled}} = \frac{\text{sim} - \mu_{\text{sim}}}{\sigma_{\text{sim}}}$$
  $$\text{rel} = 1 - \frac{1}{1 + \exp(-\text{sim}_{\text{scaled}})}$$

- `"minmax"`:
  $$\text{sim}_{\text{scaled}} = \frac{\text{sim} - \min(\text{sim})}{\max(\text{sim}) - \min(\text{sim})}$$
  $$\text{rel} = 1 - \text{sim}_{\text{scaled}}$$

- `"none"`:
  $$\text{rel} = 1 - \text{sim}$$

**权重计算**:
$$w_t = \frac{\exp(\text{rel}_t / \tau)}{\sum_{t=1}^{T} \exp(\text{rel}_t / \tau)}$$

其中 $\tau = \text{sar\_temp}$。

**SAR 分数**:
$$\text{sar} = \sum_{t=T-k+1}^{T} w_t \cdot U_t$$

#### 4. 最终融合

**归一化** (Z-score 或 Min-Max):
$$\text{sar}_s = \text{norm}(\text{sar})$$
$$\text{conf}_s = \text{norm}(\text{conf}_{\text{tail}})$$
$$\text{margin}_s = \text{norm}(\text{margin}_{\text{robust}})$$

**最终分数**:

如果 `near_tie=True`:
$$\text{final\_score} = w_{\text{sar}} \cdot \text{sar}_s + w_{\text{conf}} \cdot \text{conf}_s - w_{\text{margin}} \cdot \text{margin}_s$$

否则:
$$\text{final\_score} = w_{\text{sar}} \cdot \text{sar}_s + w_{\text{conf}} \cdot \text{conf}_s$$

---

## 9. `calculate_top2_margin_scores`

### 功能描述
计算第一条 beam 每步的 top-2 margin（前两大概率差）。

### 参数
- `beam_logprobs_list` (List[List[Dict[int, Any]]]): Beam search 对数概率列表

### 返回值
- `Tuple[float, float, List[float]]`: `(min_margin, mean_margin, margins_per_step)`

### 数学公式

对于每一步 $t$：

1. **提取概率**:
   $$p_{j,t} = \exp(\log p_{j,t})$$

2. **Top-2 Margin**:
   $$\text{margin}_t = \max(p_{1,t} - p_{2,t}, 0)$$
   
   其中 $p_{1,t}$ 和 $p_{2,t}$ 分别为该步的第一和第二大概率。

3. **统计量**:
   $$\text{min\_margin} = \min_{t} \text{margin}_t$$
   $$\text{mean\_margin} = \frac{1}{T} \sum_{t=1}^{T} \text{margin}_t$$

---

## 10. `calculate_msp_scores`

### 功能描述
计算 MSP (Maximum Softmax Probability) 不确定性评分。

### 参数
- `beam_logprobs_list` (List[List[Dict[int, Any]]]): Beam search 对数概率列表

### 返回值
- `Tuple[float, float]`: `(log_likelihood, probability)`

### 数学公式

1. **对数似然度**:
   $$\log p(y^*|x) = \sum_{t=1}^{T} \log p_t(y^*)$$

2. **序列概率**:
   $$p(y^*|x) = \exp\left(\sum_{t=1}^{T} \log p_t(y^*)\right) = \prod_{t=1}^{T} p_t(y^*)$$

3. **MSP 不确定性** (在 CoCoA 中使用):
   $$u_{\text{msp}} = 1 - p(y^*|x)$$

---

## 11. `calculate_token_entropy_scores`

### 功能描述
计算第一条 beam 每一步的 token-level 熵。

### 参数
- `beam_logprobs_list` (List[List[Dict[int, Any]]]): Beam search 对数概率列表
- `eps` (float): 数值稳定性参数，默认 `1e-12`

### 返回值
- `Tuple[float, float, List[float]]`: `(max_entropy, mean_entropy, entropies_per_step)`

### 数学公式

对于每一步 $t$：

1. **Log-Sum-Exp 归一化**:
   $$\text{LSE}_t = \log\left(\sum_{j} \exp(\log p_{j,t} - \max_j \log p_{j,t})\right) + \max_j \log p_{j,t}$$

2. **归一化概率**:
   $$p_{j,t} = \frac{\exp(\log p_{j,t} - \text{LSE}_t)}{\sum_{j} \exp(\log p_{j,t} - \text{LSE}_t)}$$

3. **熵计算**:
   $$H_t = -\sum_{j} p_{j,t} \log(p_{j,t} + \epsilon)$$

4. **统计量**:
   $$\text{max\_entropy} = \max_{t} H_t$$
   $$\text{mean\_entropy} = \frac{1}{T} \sum_{t=1}^{T} H_t$$

---

## 符号说明

- $T$: 序列长度（token 数量）
- $B$: Beam search 的 beam 数量
- $x$: 输入文本（prompt）
- $y$: 输出序列
- $y^*$: 第一条 beam（最佳序列）
- $y^i$: 第 $i$ 条 beam
- $p_t$: 第 $t$ 个 token 的概率
- $\log p_t$: 第 $t$ 个 token 的对数概率
- $p(y|x)$: 给定输入 $x$ 时序列 $y$ 的条件概率
- $\epsilon$: 数值稳定性参数（通常为 $10^{-12}$）
- $\tau$: 温度参数
- $Q_q$: 分位数函数（$q$-th quantile）

---

## 参考文献

- TokenSAR: https://arxiv.org/abs/2307.01379
- CoCoA: 相关论文（未在代码中明确引用）

