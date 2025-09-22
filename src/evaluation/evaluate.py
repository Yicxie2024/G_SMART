import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from grader import math_equal_process

from parser import parse_ground_truth, extract_answer
from utils import load_jsonl
from python_executor import PythonExecutor


def get_result(samples: list=None, file_path: str=None):
    if not samples:
        samples = list(load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]
        
    pred_keys = samples[0]['metrics']
    score_mat = np.array([s['correct'] for s in samples])
    col_means = score_mat.mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    result_json = {
        pred_key: score for pred_key, score in zip(pred_keys, mean_score)
    }

    return result_json



def evaluate(
    data_name,
    prompt_type,
    samples: list = None,
    file_path: str = None,
    max_num_samples: int = None,
    execute: bool = False,
    pred_keys: list = None
):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx'])
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]

    # 自动扩展 pred_keys：如果样本里有 pred_random_uniform，也纳入评估
    if pred_keys is None:
        candidate_keys = ['pred']
        if 'pred_random_uniform' in samples[0]:
            candidate_keys.append('pred_random_uniform')
        pred_keys = candidate_keys
    else:
        if 'pred_random_uniform' in samples[0] and 'pred_random_uniform' not in pred_keys:
            pred_keys = list(pred_keys) + ['pred_random_uniform']

    # 解析 GT 与抽取“最终预测”的答案
    for sample in samples:
        _, sample['gt'] = parse_ground_truth(sample, data_name)
        sample['metrics'] = pred_keys
        sample['preds'] = [
            extract_answer(sample.get(pred_key, ""), data_name) for pred_key in pred_keys
        ]

    # 计算每个样本、每个最终预测的正确性（与 pred_keys 对齐）
    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['preds']]

    scores = []
    timeout_cnt = 0

    progress_bar = tqdm(total=len(params), desc="Evaluate final preds")
    for idx, pred, gt in params:
        try:
            result = math_equal_process((idx, pred, gt))
            scores.append(result)
        except TimeoutError as error:
            print(error)
            scores.append(False)
            timeout_cnt += 1
        except Exception as error:
            print(error)
            exit()
        progress_bar.update(1)
    progress_bar.close()

    # —— 逐 completion 的正确性（主 completions）——
    for sample in samples:
        sample['pred_completions'] = [
            extract_answer(completion, data_name) for completion in sample.get('completions', [])
        ]
    params = [
        (idx, pred, sample['gt'])
        for idx, sample in enumerate(samples)
        for pred in sample['pred_completions']
    ]
    completion_scores = []
    timeout_cnt = 0

    progress_bar = tqdm(total=len(params), desc="Evaluate per-completion (baseline)")
    for idx, pred, gt in params:
        try:
            result = math_equal_process((idx, pred, gt))
            completion_scores.append(result)
        except TimeoutError as error:
            print(error)
            completion_scores.append(False)
            timeout_cnt += 1
        except Exception as error:
            print(error)
            exit()
        progress_bar.update(1)
    progress_bar.close()

    # —— 若存在随机 completions_random，也评估逐条正确性 —— 
    have_random_completions = 'completions_random' in samples[0]
    if have_random_completions:
        for sample in samples:
            sample['pred_completions_random'] = [
                extract_answer(c, data_name) for c in sample.get('completions_random', [])
            ]
        params_rand = [
            (idx, pred, sample['gt'])
            for idx, sample in enumerate(samples)
            for pred in sample['pred_completions_random']
        ]
        completion_scores_random = []
        progress_bar = tqdm(total=len(params_rand), desc="Evaluate per-completion (random)")
        for idx, pred, gt in params_rand:
            try:
                result = math_equal_process((idx, pred, gt))
                completion_scores_random.append(result)
            except TimeoutError as error:
                print(error)
                completion_scores_random.append(False)
                timeout_cnt += 1
            except Exception as error:
                print(error)
                exit()
            progress_bar.update(1)
        progress_bar.close()
    else:
        completion_scores_random = None

    # —— 回填到 samples —— 
    # pred_keys 的正确性矩阵
    idx = 0
    score_mat = []
    for sample in samples:
        sample['correct'] = scores[idx: idx + len(sample['preds'])]
        assert len(sample['correct']) == len(sample['preds'])
        score_mat.append(sample['correct'])
        idx += len(sample['preds'])

    # 主 completions 的正确性
    idx = 0
    for sample in samples:
        n = len(sample['pred_completions'])
        sample['correct_completions'] = completion_scores[idx: idx + n]
        assert len(sample['correct_completions']) == n
        idx += n

    # 随机 completions 的正确性
    if completion_scores_random is not None:
        idx = 0
        for sample in samples:
            n = len(sample.get('pred_completions_random', []))
            sample['correct_completions_random'] = completion_scores_random[idx: idx + n]
            assert len(sample['correct_completions_random']) == n
            idx += n

    # —— 汇总列均值（对 pred_keys）——
    max_len = max([len(s) for s in score_mat])
    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s))  # pad

    col_means = np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['preds'][-1]]),
        "acc": {pred_key: score for pred_key, score in zip(pred_keys, mean_score)},
    }

    return samples, result_json
