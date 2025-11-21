# UHead 集成到 SMART Beam Search 的可行性分析

## 当前架构对比

### PRM 流程（beam_search_smart_prm_only.py）
```
1. 使用 vLLM 生成 step (generate_k_steps)
   └─> 返回: beam.next_texts[0] (生成的 step 文本)

2. 调用 PRM 打分
   prm.score(prompts, completions)
   └─> 输入: questions (list[str]), outputs (list[list[str]])
   └─> 输出: list[list[float]] (每个 completion 的 step 分数列表)

3. 存储分数
   beam.all_scores = score[0]  # 每个 step 的分数列表
```

### UHead 流程（从 basic_usage_reasoning.ipynb）
```
1. 使用 transformers 模型生成（需要 hidden states）
   llm_adapter.generate(inputs["input_ids"], max_new_tokens=200)
   └─> 返回: 包含 hidden_states, greedy_texts, greedy_tokens 等

2. 提取 Claims/Steps
   StepsExtractor 或类似工具
   └─> 输入: greedy_text, greedy_tokens, tokenizer
   └─> 输出: List[Claim] (每个 Claim 包含 claim_text, aligned_token_ids)

3. 计算不确定性分数
   CalculatorApplyUQHead + LuhClaimEstimatorDummy
   └─> 输入: claims, uhead_features, hidden_states
   └─> 输出: uncertainty_score (每个 claim 的不确定性分数)
```

## 关键差异

### 1. 生成方式
- **PRM**: 使用 vLLM，轻量级，不需要 hidden states
- **UHead**: 需要 hidden states，必须使用 transformers 模型（或修改 vLLM 以返回 hidden states）

### 2. 打分接口
- **PRM**: 
  ```python
  scores = prm.score(questions, outputs)  # 简单函数调用
  ```
- **UHead**: 
  ```python
  # 需要完整的 pipeline
  deps = calc_infer_llm(texts, model)  # 生成 + 获取 hidden states
  claims = steps_extractor(deps)  # 提取 claims
  uncertainty = calc_apply_uhead(claims, deps)  # 计算不确定性
  ```

### 3. 数据格式
- **PRM**: 直接返回分数列表 `list[list[float]]`
- **UHead**: 需要 Claim 对象，包含 token alignment 信息

## 可行性评估

### ✅ 可行的部分
1. **Step 作为 Claim**: SMART 中的 step（由 `\n\n` 分隔）可以很好地映射为 Claim
2. **分数存储**: 可以将 uhead 的不确定性分数直接存入 `beam.all_scores`
3. **接口封装**: 可以创建一个类似 PRM 的接口，封装 uhead 的完整流程

### ⚠️ 需要解决的问题

#### 问题 1: 生成方式
**当前**: 使用 vLLM 的 `generate_k_steps`，不返回 hidden states

**解决方案**:
- **方案 A**: 修改生成逻辑，使用 transformers 模型生成（需要修改 `generate_k_steps`）
- **方案 B**: 在生成后，重新 forward pass 获取 hidden states（增加计算开销）
- **方案 C**: 使用 vLLM 的底层 API 获取 hidden states（如果支持）

#### 问题 2: Step 到 Claim 的转换
**需求**: 将生成的 step 文本转换为 Claim 对象，需要：
- `claim_text`: step 的文本内容
- `sentence`: 完整的 step 文本
- `aligned_token_ids`: step 在整个序列中的 token 位置

**解决方案**:
```python
def step_to_claim(step_text: str, full_text: str, tokenizer, context_length: int) -> Claim:
    """
    将 step 转换为 Claim 对象
    
    Args:
        step_text: 当前 step 的文本
        full_text: 完整的生成文本（prompt + 所有 steps）
        tokenizer: tokenizer 对象
        context_length: prompt 的 token 长度
    
    Returns:
        Claim 对象
    """
    # 1. 找到 step 在 full_text 中的位置
    step_start = full_text.find(step_text)
    step_end = step_start + len(step_text)
    
    # 2. 将字符位置转换为 token 位置
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    step_tokens = tokenizer.encode(step_text, add_special_tokens=False)
    
    # 3. 找到 step tokens 在 full_tokens 中的位置
    # （需要处理 tokenization 的边界对齐问题）
    aligned_token_ids = find_token_alignment(full_tokens, step_tokens, context_length)
    
    return Claim(
        claim_text=step_text,
        sentence=step_text,
        aligned_token_ids=aligned_token_ids
    )
```

#### 问题 3: 接口设计
**目标**: 创建一个类似 PRM 的接口，但使用 uhead

**设计**:
```python
class UHeadScorer:
    def __init__(self, llm, uhead, tokenizer, steps_extractor):
        self.llm = llm  # transformers 模型
        self.uhead = uhead
        self.tokenizer = tokenizer
        self.calc_infer_llm = CalculatorInferLuh(uhead, ...)
        self.calc_apply_uhead = CalculatorApplyUQHead(uhead)
        self.steps_extractor = steps_extractor
        self.estimator = LuhClaimEstimatorDummy()
    
    def score(self, questions: list[str], outputs: list[list[str]]) -> list[list[float]]:
        """
        类似 PRM.score 的接口
        
        Args:
            questions: 问题列表
            outputs: 每个问题的输出列表（每个输出是一个完整的文本，包含多个 steps）
        
        Returns:
            list[list[float]]: 每个输出的 step 分数列表
        """
        all_scores = []
        for question, output_list in zip(questions, outputs):
            question_scores = []
            for output in output_list:
                # 1. 准备输入
                full_text = question + " " + output
                inputs = self.tokenizer(full_text, return_tensors="pt").to(self.llm.device)
                
                # 2. 生成并获取 hidden states
                deps = self.calc_infer_llm(
                    dependencies={},
                    texts=[full_text],
                    model=WhiteboxModelBasic(self.llm, self.tokenizer),
                    max_new_tokens=0  # 已经生成，只需要 forward pass
                )
                
                # 3. 提取 steps/claims
                steps = self.extract_steps(output)  # 按 \n\n 分割
                claims = [self.step_to_claim(step, full_text, deps) for step in steps]
                
                # 4. 计算不确定性
                deps["claims"] = [claims]
                uncertainty_deps = self.calc_apply_uhead(deps, [full_text], ...)
                uncertainty_scores = self.estimator(uncertainty_deps)
                
                question_scores.append(uncertainty_scores[0])
            all_scores.append(question_scores)
        
        return all_scores
```

## 实现建议

### 阶段 1: 最小可行实现
1. 创建 `UHeadScorer` 类，实现类似 PRM 的接口
2. 实现 step 到 Claim 的转换逻辑
3. 在 beam search 中替换 PRM 调用为 UHeadScorer

### 阶段 2: 优化
1. 批量处理以提高效率
2. 缓存 hidden states（如果可能）
3. 优化 token alignment 算法

### 阶段 3: 集成测试
1. 确保分数格式与 PRM 兼容
2. 验证 SMART 逻辑（threshold, pruning 等）正常工作
3. 性能对比

## 注意事项

1. **性能**: uhead 需要额外的 forward pass，可能比 PRM 慢
2. **内存**: 需要存储 hidden states，内存占用可能增加
3. **兼容性**: 需要确保 uhead 的分数范围与 PRM 兼容（可能需要归一化）
4. **生成模型**: 如果继续使用 vLLM 生成，需要额外的 forward pass 获取 hidden states

## 结论

**可行性**: ✅ **高度可行**

主要工作：
1. 实现 step 到 Claim 的转换
2. 创建 UHeadScorer 接口
3. 修改生成逻辑以获取 hidden states（或添加额外的 forward pass）
4. 在 beam search 中集成

关键挑战是 token alignment 和性能优化，但这些都是可以解决的工程问题。

