import logging

import os
import sys
import json

from sal.config import Config
from sal.utils.data import save_dataset
from sal.utils.parser import H4ArgumentParser

from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_jsonl(path: str):
    """
    Load a JSONL file into a list[dict].
    Skips empty/invalid lines but logs a warning.
    """
    samples = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSONL not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # ensure it's a dict
                if isinstance(obj, dict):
                    samples.append(obj)
                else:
                    logger.warning(f"[load_jsonl] Line {i} is not a JSON object; skipping.")
            except Exception as e:
                logger.warning(f"[load_jsonl] Failed to parse line {i}: {e}")
    if not samples:
        logger.warning("[load_jsonl] No valid samples loaded.")
    else:
        logger.info(f"[load_jsonl] Loaded {len(samples)} samples from {path}")
    return samples


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    # so that evaluation can be imported
    sys.path.append("src/evaluation")
    from evaluation.evaluate import evaluate

    # === 1) Load dataset from your JSONL ===
    # replace with your actual path
    jsonl_path = "/storage/ukp/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/outputs/smart/conf/smart_conf/beam_search_completions_T-0.8--top_p-1.0--n-16--m-4--iters-40--look-0--seed-0--agg_strategy--last_threshold-0.9_None_cocoa.jsonl"
    samples = load_jsonl(jsonl_path)
    print('[INFO] config: ', config)

    if config.approach == "best_of_n" or config.approach == "beam_search":
        subsets = [2**i for i in range(16) if 2**i <16]
        keys = []
        for n in subsets:
            keys.extend([f"pred_weighted@{n}", f"pred_maj@{n}", f"pred_naive@{n}"])
    else:
        keys = ["pred", "pred_random_uniform"]

    dataset, result = evaluate(
        data_name="math", prompt_type=None, samples=samples, pred_keys=keys
    )
    dataset = Dataset.from_list(
        [
            {k: v for k, v in dict(sample).items() if k != "pred_completions"}
            for sample in dataset
        ]
    )

    save_dataset(dataset, config)

    logger.info(result)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
