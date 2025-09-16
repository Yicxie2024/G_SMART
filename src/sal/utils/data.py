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

import logging
import time
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import (
    create_branch,
    list_repo_commits,
    repo_exists,
)

from sal.config import Config

logger = logging.getLogger()


def get_dataset(config: Config) -> Dataset:
    if config.dataset_name == "sampled_math500":
        dataset = pd.read_csv(config.dataset_name + ".csv")
        dataset = Dataset.from_pandas(dataset)
    elif config.dataset_name == "prm_math500":
        # load from jsonl file
        dataset = pd.read_json("train.jsonl", lines=True)
        dataset = Dataset.from_pandas(dataset)
    else:
        dataset = load_dataset(config.dataset_name, split=config.dataset_split)

    if config.dataset_start is not None and config.dataset_end is not None:
        dataset = dataset.select(range(config.dataset_start, config.dataset_end))
    if config.num_samples is not None:
        dataset = dataset.select(range(min(len(dataset), config.num_samples)))
    return dataset


def save_dataset(dataset, config):
    if config.push_to_hub:
        # Since concurrent pushes can get rejected by the Hub, we make several attempts to push the dataset with try/except
        for _ in range(20):
            try:
                # Create branch from the repo's initial commit.
                # This is needed to avoid branching from a commit on main that already has data
                if repo_exists(config.hub_dataset_id, repo_type="dataset"):
                    initial_commit = list_repo_commits(
                        config.hub_dataset_id, repo_type="dataset"
                    )[-1]
                    create_branch(
                        repo_id=config.hub_dataset_id,
                        branch=config.revision,
                        revision=initial_commit.commit_id,
                        exist_ok=True,
                        repo_type="dataset",
                    )
                url = dataset.push_to_hub(
                    config.hub_dataset_id,
                    revision=config.revision,
                    split="train",
                    private=config.hub_dataset_private,
                    commit_message=f"Add {config.revision}",
                )
                break
            except Exception as e:
                logger.error(f"Error pushing dataset to the Hub: {e}")
                time.sleep(5)
        logger.info(f"Pushed dataset to {url}")
        return

    # ------------ 本地保存路径与命名规则（健壮版） ------------
    # 根目录
    out_root = Path(config.output_dir or f"data/{config.model_path}")
    out_root.mkdir(parents=True, exist_ok=True)

    # 文件夹名：base/smart + score_method（自动兼容 prm/conf/sse/...）
    is_smart = (getattr(config, "draft_model_path", None) is not None) or \
               (getattr(config, "smart_search", False) is True)
    base_tag = "smart" if is_smart else "base"
    score_tag = getattr(config, "score_method", "prm")
    folder_name = f"{base_tag}_{score_tag}"
    out_dir = out_root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # approach 名：beam_width=1 就当作 best_of_n；否则用 config.approach
    #（可选也可加上 _smart 后缀，但为了与你原来一致，这里不加）
    approach_fn = "best_of_n" if getattr(config, "beam_width", 1) == 1 else config.approach

    # 兜底/规范化字段，避免 None 出现在文件名里
    T = config.temperature
    top_p = config.top_p
    n = getattr(config, "n", 1) or 1
    m = getattr(config, "beam_width", 1) or 1
    iters = getattr(config, "num_iterations", 1) or 1
    look = getattr(config, "lookahead", 0) or 0
    seed = getattr(config, "seed", 42)
    agg = getattr(config, "agg_strategy", "last")
    threshold = getattr(config, "threshold", None)

    num_samples = getattr(config, "num_samples", None)
    num_samples_tag = str(num_samples) if num_samples is not None else "all"

    ds_start = getattr(config, "dataset_start", None)
    ds_end = getattr(config, "dataset_end", None)
    split_tag = f"{ds_start}-{ds_end}" if (ds_start is not None and ds_end is not None) else "full"

    # 组装文件名
    # 与你原先一致的核心超参都保留；当有 threshold 时才追加，避免无意义字段
    base_name = (
        f"{approach_fn}_completions_"
        f"T-{T}--top_p-{top_p}"
        f"--n-{n}--m-{m}--iters-{iters}--look-{look}"
        f"--seed-{seed}--agg_strategy--{agg}"
    )
    if threshold is not None:
        base_name += f"_threshold-{threshold}"

    filename = f"{base_name}_{num_samples_tag}_datasplit_{split_tag}.jsonl"
    out_path = out_dir / filename

    # 保存为 JSONL
    dataset.to_json(str(out_path), lines=True)
    logger.info(f"Saved completions to {out_path}")
