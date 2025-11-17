# Notebook 中 UHead 不确定性评估流程详解

## 整体架构

Notebook 使用 `CausalLMWithUncertainty` 包装器，它协调三个 `StatCalculator`：
1. **CalculatorInferLuh**: 生成文本并提取特征
2. **StepsExtractor**: 从生成文本中提取步骤（claims）
3. **CalculatorApplyUQHead**: 对每个 claim 计算不确定性分数

---

## 详细流程（一步一步）

### 阶段 0: 初始化（Cell 2-3）

```python
# 1. 加载基础 LLM 模型
llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# 2. 加载 UHead（基于基础模型）
uhead = AutoUncertaintyHead.from_pretrained(
    "rediska0123/uhead_Qwen2.5-1.5B-Instruct_6epochs", 
    base_model=llm
)

# 3. 初始化三个 StatCalculator
calc_infer_llm = CalculatorInferLuh(
    uhead,
    tokenize=True,
    args_generate={"generation_config": generation_config, "max_new_tokens": 200},
    device="cuda",
    generations_cache_dir="",
    predict_token_uncertainties=False,  # 关键：只提取特征，不计算 token-level 不确定性
)

calc_steps_extractor = StepsExtractor()  # 提取步骤

calc_apply_uhead = CalculatorApplyUQHead(uhead)  # 应用 UHead 计算不确定性

# 4. 创建包装器
llm_adapter = CausalLMWithUncertainty(
    llm,
    tokenizer=tokenizer,
    stat_calculators=[calc_infer_llm, calc_steps_extractor, calc_apply_uhead],
    estimator=LuhClaimEstimatorDummy(),
)
```

---

### 阶段 1: 生成和特征提取（CalculatorInferLuh）

**调用**: `llm_adapter.generate(inputs["input_ids"], max_new_tokens=200)`

#### 步骤 1.1: 文本生成

```python
# CalculatorInferLuh.__call__() 内部执行：

# 1. Tokenize 输入
batch = model.tokenize(texts)  # texts = ["<|im_start|>system\n...user\n..."]

# 2. 调用 model.generate() 生成文本
out = model.generate(
    **batch,
    output_scores=True,
    return_dict_in_generate=True,
    max_new_tokens=200,
    output_attentions=True,  # 需要 attention 用于特征提取
    output_hidden_states=True,  # 需要 hidden states 用于特征提取
    do_sample=False,  # Greedy decoding
)
```

**输出 `out` 包含**:
- `out.sequences`: 生成的完整序列（input + generated）
- `out.scores`: 每个生成步骤的 logits
- `out.attentions`: 注意力权重（如果 `output_attentions=True`）
- `out.hidden_states`: 隐藏状态（如果 `output_hidden_states=True`）

#### 步骤 1.2: 后处理预测结果

```python
# CalculatorInferLuh.postprocess_predictions()

# 1. 提取生成的 tokens
idx = batch["input_ids"].shape[1]  # Context 长度
seq = out.sequences[i, idx:]  # 生成的 tokens
greedy_tokens = seq[:length].tolist()  # 去除 EOS 后的 tokens

# 2. 提取 logits
logits = torch.stack(out.scores, dim=1)  # [batch_size, seq_len, vocab_size]
cut_logits = logits[i, :length, :].cpu().numpy()

# 3. 计算 log probabilities
log_probs = [cut_logits[j, greedy_tokens[j]] for j in range(length)]

# 4. 解码为文本
greedy_texts = tokenizer.decode(seq[:text_length])
```

**结果字典**:
```python
result_dict = {
    "greedy_tokens": [[123, 456, 789, ...]],  # 生成的 token IDs
    "greedy_texts": ["Step 1: Align the numbers..."],  # 生成的文本
    "greedy_log_probs": [[-0.5, -0.3, -0.8, ...]],  # 每个 token 的 log prob
    "input_tokens": [[...]],  # 输入的 token IDs
}
```

#### 步骤 1.3: 构建 full_attention_mask

```python
# 创建与 sequences 相同形状的 mask
full_attn_mask = torch.zeros_like(out.sequences).bool()

for i in range(batch_size):
    idx = batch["input_ids"].shape[1]  # Context 长度
    
    # Context 部分：使用原始 attention_mask
    full_attn_mask[i, :idx] = batch["attention_mask"][i]
    
    # Generated 部分：全部设为 1
    length = len(result_dict["greedy_tokens"][i])
    full_attn_mask[i][idx : idx + length] = 1

out["full_attention_mask"] = full_attn_mask
out["context_lengths"] = torch.tensor([len(it) for it in batch["input_ids"]])
batch["context_lenghts"] = out["context_lengths"]  # 注意拼写错误是故意的
```

#### 步骤 1.4: 提取特征（关键步骤）

```python
# 因为 predict_token_uncertainties=False，所以执行：

# 调用 UHead 的特征提取器
uhead_features = uhead.feature_extractor(batch, out)
```

**特征提取器做了什么**:
1. **basic_attention**: 从 `out.attentions` 中提取注意力特征
   - 选择特定的 attention heads（根据 config.yaml）
   - 聚合为固定维度的特征向量
   
2. **token_probabilities**: 从 `out.scores` 中提取 token 概率特征
   - 计算每个位置的 token 概率分布
   - 提取统计特征（entropy, max prob, etc.）

3. **组合**: `torch.cat([basic_attention_features, token_probabilities_features], dim=-1)`

**输出**:
```python
result_dict = {
    "uhead_features": uhead_features,  # [batch_size, seq_len, feature_dim]
    "llm_inputs": batch,  # 包含 input_ids, attention_mask, context_lenghts
    "full_attention_mask": full_attn_mask,  # [batch_size, full_seq_len]
    "greedy_tokens": [[...]],
    "greedy_texts": ["..."],
    ...
}
```

---

### 阶段 2: 提取步骤（StepsExtractor）

**依赖**: `greedy_texts`, `greedy_tokens`

#### 步骤 2.1: 查找步骤标记

```python
# StepsExtractor.__call__()

for input_text, greedy_text, greedy_tokens in zip(...):
    # 使用正则表达式查找步骤标记
    spans = self._find_spans(greedy_text)
    # 例如: [(0, 50), (50, 120), (120, 200)]  # Step 1, Step 2, Step 3 的位置
```

**正则表达式**:
- `STEP_RE = r'(?mi)(?:^|\n)(?P<marker>\s*-?\s*Step\s+\d+\s*:\s*)'`
- `ANSWER_RE = r'(?mi)(?:^|\n)(?P<marker>\s*(?:<\s*Answer\s*>|Answer)\s*:\s*)'`

#### 步骤 2.2: 字符位置 → Token 位置

```python
# 将字符边界转换为 token 边界
char_boundaries = [0, 50, 50, 120, 120, 200]  # start, end, start, end, ...
token_boundaries = self._char_to_token_index_boundaries(
    greedy_text, greedy_tokens, tokenizer, char_boundaries
)
# 结果: [0, 10, 10, 25, 25, 40]  # token 索引
```

#### 步骤 2.3: 创建 Claim 对象

```python
claims = []
for i, (start, end) in enumerate(spans):
    seg = text[start:end]  # "Step 1: Align the numbers..."
    
    if not self.filter_claim_texts(seg):  # 过滤掉 "Reasoning Steps:" 等
        continue
    
    tok_start = token_boundaries[2 * i]      # 0
    tok_end = token_boundaries[2 * i + 1]    # 10
    
    # aligned_token_ids 是相对于生成开始位置的 token 索引
    aligned_ids = list(range(tok_start, min(tok_end, len(greedy_tokens))))
    
    claims.append(Claim(
        claim_text=seg.strip(),
        sentence=seg,
        aligned_token_ids=aligned_ids,  # [0, 1, 2, ..., 9]
    ))
```

**输出**:
```python
{
    "claims": [
        [
            Claim(claim_text="Step 1: Align...", aligned_token_ids=[0,1,2,...,9]),
            Claim(claim_text="Step 2: Add...", aligned_token_ids=[10,11,12,...,24]),
            Claim(claim_text="Step 3: Write...", aligned_token_ids=[25,26,27,...,39]),
        ]
    ]
}
```

---

### 阶段 3: 计算不确定性（CalculatorApplyUQHead）

**依赖**: `uhead_features`, `claims`, `llm_inputs`, `full_attention_mask`

#### 步骤 3.1: 准备 Claim Tensors

```python
# CalculatorApplyUQHead.prepare_claims()

batch_size = len(batch["input_ids"])
context_lenghts = batch["context_lenghts"]  # [context_len]

for i in range(batch_size):
    instance_claims = []
    for claim in claims[i]:
        # 创建全零 mask
        mask = torch.zeros(full_len, dtype=int)
        
        # 在 claim 对应的 token 位置设为 1
        # 注意：aligned_token_ids 是相对于生成开始的，需要加上 context_len
        mask[context_lenghts[i] + torch.as_tensor(claim.aligned_token_ids)] = 1
        
        # 忽略第一个 token（通常是 <s>）
        instance_claims.append(mask[1:])
    
    all_claim_tensors.append(torch.stack(instance_claims))
```

**示例**:
```python
# 假设 context_len=50, claim.aligned_token_ids=[0,1,2,...,9]
# 那么 mask[50:60] = 1，其他位置 = 0
# mask[1:] 表示忽略第一个 token
```

#### 步骤 3.2: 调用 UHead 计算不确定性

```python
# CalculatorApplyUQHead.__call__()

batch["claims"] = all_claim_tensors  # [batch_size, num_claims, seq_len-1]

with torch.no_grad():
    uncertainty_logits = uhead._compute_tensors(
        batch,  # 包含 input_ids, attention_mask, claims, context_lenghts
        uhead_features.to(device),  # [batch_size, seq_len, feature_dim]
        full_attention_mask[:, :-1].to(device),  # 忽略最后一个 token
    )
```

**UHead._compute_tensors() 内部**:
1. **特征聚合**: 对每个 claim 的 token 位置，聚合 `uhead_features`
   ```python
   # 对于每个 claim，提取对应位置的 features
   claim_features = uhead_features[mask == 1]  # [num_claim_tokens, feature_dim]
   aggregated = claim_features.mean(dim=0)  # [feature_dim]
   ```

2. **通过 UHead 网络**: 
   ```python
   logits = uhead.proj(aggregated)  # [1]  # 单个 logit
   ```

3. **输出**: `[batch_size, num_claims]` 的 logits

#### 步骤 3.3: 后处理

```python
# 过滤掉 -100（padding 值）
final_uncertainty_claims = [
    np.asarray([e.item() for e in claim if e != -100]) 
    for claim in uncertainty_logits.cpu().numpy()
]

# 结果: [[logit1, logit2, logit3]]  # 每个 claim 一个 logit
```

---

### 阶段 4: 转换为概率分数

**在 notebook 外部（或通过 estimator）**:

```python
# 对每个 claim 的 logit 应用 sigmoid
uncertainty_scores = [
    1.0 / (1.0 + np.exp(-logit)) 
    for logit in uncertainty_logits[0]
]

# 结果: [0.0387, 0.4127, 0.7929]  # 每个步骤的不确定性分数
```

---

## 数据流图

```
输入文本
  ↓
[CalculatorInferLuh]
  ├─→ model.generate() → out (sequences, scores, attentions, hidden_states)
  ├─→ postprocess_predictions() → greedy_tokens, greedy_texts
  ├─→ 构建 full_attention_mask
  └─→ uhead.feature_extractor() → uhead_features [batch, seq_len, feature_dim]
        ↓
[StepsExtractor]
  ├─→ _find_spans() → 找到 "Step 1:", "Step 2:" 等位置
  ├─→ _char_to_token_index_boundaries() → 转换为 token 索引
  └─→ 创建 Claim 对象 → claims [batch, num_claims]
        ↓
[CalculatorApplyUQHead]
  ├─→ prepare_claims() → claim_tensors [batch, num_claims, seq_len-1]
  ├─→ uhead._compute_tensors() → uncertainty_logits [batch, num_claims]
  └─→ 后处理 → final_uncertainty_claims
        ↓
[Estimator (可选)]
  └─→ 转换为概率 → uncertainty_scores [num_claims]
```

---

## 关键点总结

1. **特征提取时机**: 在生成完成后，使用完整的 `out` 对象（包含 attentions 和 hidden_states）

2. **Claim 提取**: 基于正则表达式匹配 "Step N:" 模式，将生成文本分割为多个 claims

3. **特征聚合**: 对每个 claim 内的所有 token 的特征进行聚合（通常是 mean）

4. **不确定性计算**: UHead 是一个简单的 MLP，输入是聚合后的特征，输出是单个 logit

5. **概率转换**: 通过 sigmoid 将 logit 转换为 [0, 1] 的概率分数

6. **依赖关系**: 
   - `CalculatorInferLuh` 必须最先执行（提供 `greedy_texts`, `greedy_tokens`, `uhead_features`）
   - `StepsExtractor` 依赖 `greedy_texts` 和 `greedy_tokens`
   - `CalculatorApplyUQHead` 依赖 `uhead_features` 和 `claims`

---

## 与 SMART 实现的差异

1. **生成方式**: Notebook 使用 `model.generate()`，SMART 使用 vLLM 生成
2. **特征提取**: Notebook 在生成时同时提取特征，SMART 在生成后单独提取
3. **步骤提取**: Notebook 使用 `StepsExtractor`，SMART 当前使用整个 response 作为单个 claim
4. **不确定性聚合**: Notebook 对每个 claim 单独计算，SMART 当前对整个 response 计算

