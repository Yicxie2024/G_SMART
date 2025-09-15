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

from dataclasses import dataclass
from typing import Literal

from huggingface_hub import get_full_repo_name
from sal.utils.hub import get_dataset_revisions


@dataclass
class Config:
    # === Search / Model ===
    approach: Literal["best_of_n", "beam_search", "dvts"] = "best_of_n"
    smart_search: bool = True

    # 打分方式：'prm'（原始 PRM）、'conf'（logprob 置信度）、'sse'（采样一致性/语义熵）
    score_method: Literal["prm", "conf", "sse"] = "prm"

    # PRM
    prm_path: str = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
    prm_batch_size: int = 8  # 合并：去掉了重复定义

    # 主/草稿模型
    model_path: str = "meta-llama/Llama-3.2-1B-Instruct"
    draft_model_path: str = None

    # vLLM 资源
    gpu_memory_utilization: float = 0.4  # vLLM 占用比例

    # SMART 触发阈值（分数越大越好；低于该阈值会纠偏）
    threshold: float = 0.9

    # === IO / Runner ===
    output_dir: str = (
        "/mnt/beegfs/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/outputs/smart"
    )
    num_proc: int = None
    push_to_hub: bool = False
    hub_dataset_id: str = None
    hub_dataset_private: bool = False
    overwrite_hub_revision: bool = False
    apply_voting: bool = True
    measure_flops: bool = False

    # === Dataset ===
    dataset_name: str = "HuggingFaceH4/MATH-500"
    dataset_config: str = None
    dataset_split: str = "test"
    dataset_start: int = None
    dataset_end: int = 2
    num_samples: int = None

    # Chat template related options
    system_prompt: str = (
        "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."
    )
    custom_chat_template: str = (
        '{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'
    )
    # Search Related Options
    n: int = None
    temperature: float = 0.8
    top_p: float = 1.0
    search_batch_size: int = 1
    seed: int = 42
    max_tokens: int = 2048

    # 聚合策略（步级分 -> 候选总分）
    agg_strategy: Literal["last", "min", "prod"] = "last"

    # logprob 置信打分的策略
    conf_strategy: Literal["probs_mean", "pmax", "logmean", "margin", "neg_entropy"] = (
        "probs_mean"
    )

    # SSE 语义熵参数（仅在 score_method='sse' 时使用）
    sse_samples: int = 6
    sse_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sse_sim_threshold: float = 0.85
    sse_max_step_tokens: int = 128

    # DVTS / Beam Search options
    beam_width: int = 4
    num_iterations: int = 20
    lookahead: int = 1  # 前瞻“段/step”数（由 stop=["\\n\\n"] 定义）

    # Beam search options:
    filter_duplicates: bool = True
    sort_completed: bool = False

    def __post_init__(self):
        if self.approach == "dvts":
            if self.n % self.beam_width != 0:
                raise ValueError("n should be a multiple of beam_width")
            self.n_beams = self.n // self.beam_width

        if self.approach == "beam_search":
            # TODO: implemented a batched version
            if self.search_batch_size != 1:
                raise ValueError("search_batch_size should be 1 for beam_search")

        # Setting up push to hub dataset
        if self.push_to_hub:
            model_name = self.model_path.split("/")[-1]
            if self.hub_dataset_id is None:
                self.hub_dataset_id = get_full_repo_name(
                    f"{model_name}-{self.approach}-prm-completions"
                )
            revisions = get_dataset_revisions(self.hub_dataset_id)

            if self.approach == "beam_search" or self.approach == "dvts":
                self.revision = (
                    f"{self.dataset_name.replace('/', '_')}--T-{self.temperature}"
                    f"--top_p-{self.top_p}--n-{self.n}--m-{self.beam_width}"
                    f"--iters-{self.num_iterations}--look-{self.lookahead}"
                    f"--seed-{self.seed}--agg_strategy--{self.agg_strategy}"
                )
            elif self.approach == "best_of_n":
                self.revision = (
                    f"{self.dataset_name.replace('/', '_')}--T-{self.temperature}"
                    f"--top_p-{self.top_p}--n-{self.n}--seed-{self.seed}"
                    f"--agg_strategy-{self.agg_strategy}"
                )
            else:
                raise ValueError(f"Unknown approach {self.approach}")

            if self.dataset_start is not None and self.dataset_end is not None:
                self.revision = (
                    f"{self.revision}--chunk-{self.dataset_start}_{self.dataset_end}"
                )

            if not self.overwrite_hub_revision and self.revision in revisions:
                exit()
