# uHead Deployment and Usage Guide

本指南介绍如何从零开始配置 uHead（LLM Uncertainty Head）环境、加载模型，并获得生成阶段的不确定性输出。

---

## 1. 环境准备

### 1.1 系统要求
- macOS / Linux / Windows（示例以 macOS + Conda 为主）
- Python 3.10 或 3.11
- Conda / Miniconda（推荐）
- 网络访问 Hugging Face 以及 GitHub 的权限

### 1.2 创建 Conda 环境
```bash
conda create -n smart python=3.10 -y
conda activate smart
```

### 1.3 安装基础依赖
为了与 `luh` 项目的版本兼容，建议安装以下版本：
```bash
pip install --upgrade pip
pip install "torch==2.3.1" --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.46.2 accelerate==0.34.2 huggingface-hub==0.26.2 safetensors sentencepiece datasets
```
> 注：Apple Silicon 可安装相同的 CPU 版 torch，若需要 GPU 加速，可切换为 nightlies 或 MPS 版本；在 MPS 上运行 uHead 存在部分 feature extractor 输出 `nan` 的已知问题，建议先用 CPU 调试。

### 1.4 安装 uHead 及依赖
1. 克隆仓库（假设路径为 `~/Desktop/smart_new_all_results`）：
   ```bash
   cd ~/Desktop/smart_new_all_results
   git clone https://github.com/IINemo/llm-uncertainty-head.git llm-uncertainty-layer
   ```
2. 安装 uHead 与 lm-polygraph（dev 分支）：
   ```bash
   cd llm-uncertainty-layer
   pip install -e . --no-deps
   pip install git+https://github.com/iinemo/lm-polygraph.git@dev
   ```
   - `--no-deps` 防止覆盖已装的 transformers/torch 版本。
   - `lm-polygraph` 是 uHead 统计器所需的工具包。

---

## 2. 快速示例：运行 `run_uhead_quantized.py`
项目根目录提供了脚本 `run_uhead_quantized.py`，会：
1. 加载 base LLM（默认 `Qwen/Qwen2.5-1.5B-Instruct`）。
2. 生成回答并打印聊天格式、逐 token logprob。
3. （可选）加载 uHead 权重，输出 claim 级别不确定性。

### 2.1 基础调用
```bash
conda activate smart
cd ~/Desktop/smart_new_all_results
python run_uhead_quantized.py \
    --device cpu \
    --max-new-tokens 64 \
    --no-sample
```
输出包含：
- `RAW DECODED SEQUENCE`：完整系统 + 用户 + 助手对话。
- `ASSISTANT ANSWER`：剥离特殊 token 后的助手回复。
- `[Token xx]` 行：逐 token 的 logprob / 概率。

### 2.2 启用 uHead
要查看 claim 级别的不确定性，需指定 `--enable-uhead`：
```bash
python run_uhead_quantized.py \
    --device cpu \
    --max-new-tokens 32 \
    --no-sample \
    --enable-uhead
```
脚本会：
- 自动下载 `weights.pth` 与 `config.yaml`（默认 repo `rediska0123/uhead_Qwen2.5-1.5B-Instruct_6epochs`）。
- 构造 `full_attention_mask`、模拟 claim mask（覆盖全部生成 token）。
- 调用 `AutoUncertaintyHead` 并打印：
  ```
  === CLAIM-LEVEL LOGITS (naive span) ===
  Claim 00: logit=+1.4238 prob=0.8059
  [uHead] NOTE: claims are approximated by a single span covering all generated tokens.
  ```
  其中 `logit` 为原始输出，`prob` 为 `sigmoid(logit)`。

### 2.3 保存结构化结果
```bash
python run_uhead_quantized.py \
    --device cpu \
    --max-new-tokens 32 \
    --no-sample \
    --enable-uhead \
    --save-json outputs/demo.json
```
JSON 中的 `uhead_summary` 字段包含：
```json
{
  "claim_logits": [...],
  "claim_probabilities": [...],
  "claim_mask": [...],
  "span_mode": "generated_tokens_full"
}
```
方便后续离线分析或可视化。

---

## 3. 更深入的用法

### 3.1 参考官方示例笔记本
仓库 `llm-uncertainty-layer/examples` 内包含三个示例：
- `generate_with_uncertainty_claim_level.ipynb`：claim 级别不确定性流程。
- `generate_with_uncertainty_token_level.ipynb`：token 级别不确定性流程。
- `generate_with_uncertainty_claim_level_saplma.ipynb`：SAPLMA 变体。

主要步骤包括：
1. `AutoUncertaintyHead.from_pretrained(...)` 加载指定权重。
2. 使用 `WhiteboxModelBasic` 包装 LLM 与 tokenizer。
3. 通过 `CalculatorInferLuh` 或其他统计器生成文本并返回：
   - `greedy_texts` / `greedy_tokens`
   - `greedy_log_probs`
   - `uncertainty_logits`
   - 需要 claim 时，还会读取 `stats["claims"]` 与 `aligned_token_ids`。

### 3.2 替换 claim mask / 接入真实注释
示例脚本使用“全部生成 token”作为单一 claim demo。若要获得真实 claim 级别得分：
1. 利用仓库中的 `generate_dataset/run_extract_verify_claims.py`、`train_luh/run_train_luh.py` 生成/加载包含 claim 对齐信息的数据。
2. 构造 `llm_inputs["claims"]`（二维 0/1 mask 或 `Claim` 对象），再调用 uHead。

### 3.3 MPS / GPU 注意事项
- Apple M 系列上的 MPS 后端目前在 `basic_attention` 特征提取时容易出现 `nan`；推荐切到 CPU 或 CUDA。
- 如果必须使用 GPU，请先确认 `output_attentions=True` 时的返回张量正常无 `nan`。

---

## 4. 常见问题

| 问题 | 解决方案 |
|------|-----------|
| `ImportError: No module named 'luh'` | 确认已在 `smart` 环境内执行 `pip install -e llm-uncertainty-layer --no-deps` |
| 加载权重时报权限 / 找不到文件 | 检查网络是否可访问 Hugging Face，或提前 `huggingface-cli login` |
| 生成全是 `!` 导致 logprob 为 `nan` | 通常是注意力掩码未正确传递；本脚本版本已通过 `tokenizer(..., return_tensors="pt")` 获取 `attention_mask` 解决 |
| uHead 输出为空 | 检查 `llm_inputs["claims"]` 是否为空；脚本默认会构造嘈杂的占位 mask，并告警 |
| 想获取 token 级不确定性 | 使用 `generate_with_uncertainty_token_level.ipynb` 或加载对应的 token 版 uHead repo |

---

## 5. 后续扩展
- 将 `run_uhead_quantized.py` 融入自己的评测框架，批量生成与记录 `token_scores`、`claim_probabilities`。
- 根据 `config.yaml` 自定义 feature extractor / head 结构并重新训练：参考 `train_luh/`、`configs/` 下的训练脚本与配置。
- 若要部署在线服务，可参考 README 中的 vLLM 或 `lm-polygraph` 的服务化接口。

---

完成以上步骤后，你就可以在本地环境中加载 uHead，观测 LLM 在生成过程中的不确定性指标，并结合真实 claim 注释进行精细化分析。祝调试顺利！
