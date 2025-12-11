import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from grader import math_equal_process
from mbpp_grader import extract_python_code, mbpp_equal_process

from parser import parse_ground_truth, extract_answer, extract_bbh_answer
from utils import load_jsonl
from python_executor import PythonExecutor


def extract_answer_with_context(pred_str, data_name, sample=None):
    """
    Extract answer with additional context (e.g., bbh_subset for BBH dataset, MBPP code).
    """
    if data_name == "bbh" and sample and 'bbh_subset' in sample:
        return extract_bbh_answer(pred_str, bbh_subset=sample['bbh_subset'])
    elif data_name == "mbpp":
        # For MBPP, extract Python code instead of math answer
        return extract_python_code(pred_str)
    else:
        return extract_answer(pred_str, data_name)


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



def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False, pred_keys: list=None):
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

    # automatically extend pred_keys: if there are pred_random_uniform, pred_random, pred_slm, pred_llm, also evaluate them
    if pred_keys is None:
        candidate_keys = ['pred']
        if 'pred_random_uniform' in samples[0]:
            candidate_keys.append('pred_random_uniform')
        if 'pred_slm' in samples[0]:
            candidate_keys.append('pred_slm')
        if 'pred_llm' in samples[0]:
            candidate_keys.append('pred_llm')
        if 'pred_random' in samples[0]:
            candidate_keys.append('pred_random')
        pred_keys = candidate_keys
    else:
        if 'pred_random_uniform' in samples[0] and 'pred_random_uniform' not in pred_keys:
            pred_keys = list(pred_keys) + ['pred_random_uniform']
        if 'pred_slm' in samples[0] and 'pred_slm' not in pred_keys:
            pred_keys = list(pred_keys) + ['pred_slm']
        if 'pred_llm' in samples[0] and 'pred_llm' not in pred_keys:
            pred_keys = list(pred_keys) + ['pred_llm']
        if 'pred_random' in samples[0] and 'pred_random' not in pred_keys:
            pred_keys = list(pred_keys) + ['pred_random']

    # parse GT and extract final prediction
    for sample in samples:
        _, sample['gt'] = parse_ground_truth(sample, data_name)
        sample['metrics'] = pred_keys
        sample['preds'] = [extract_answer_with_context(sample[pred_key], data_name, sample) for pred_key in pred_keys]
        

    # calculate scores for final prediction
    if data_name == "mbpp":
        # For MBPP: evaluate extracted code using test cases
        # pred is already extracted code, we need to execute it against test cases
        params = [
            (idx, sample[pred_key], sample)  # Use original pred (code), not extracted
            for idx, sample in enumerate(samples)
            for pred_key in pred_keys
        ]
        
        scores = []
        timeout_cnt = 0 

        progress_bar = tqdm(total=len(params), desc="Evaluate final predictions (MBPP)")
        for idx, pred, sample in params:
            try:
                result = mbpp_equal_process((idx, pred, sample))
                scores.append(result)
            except TimeoutError as error:
                print(error)
                scores.append(False)
                timeout_cnt += 1
            except Exception as error:
                print(error)
                print(f"Error in final prediction eval for sample {idx}: {error}")
                scores.append(False)
            progress_bar.update(1)
        progress_bar.close()
    else:
        # For math/BBH: compare extracted answer with ground truth
        params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['preds']]

        scores = []
        timeout_cnt = 0 

        progress_bar = tqdm(total=len(params), desc="Extract preds for each completion")
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
    
    # calculate scores for each completions
    # For MBPP, we evaluate using test cases instead of comparing predictions with ground truth
    if data_name == "mbpp":
        # For MBPP: use original completions and test cases
        params = [
            (idx, completion, sample)
            for idx, sample in enumerate(samples)
            for completion in sample.get('completions', [])
        ]
        completion_scores = []
        timeout_cnt = 0

        progress_bar = tqdm(total=len(params), desc="Evaluate per-completion (MBPP with test cases)")
        for idx, completion, sample in params:
            try:
                result = mbpp_equal_process((idx, completion, sample))
                completion_scores.append(result)
            except TimeoutError as error:
                print(error)
                completion_scores.append(False)
                timeout_cnt += 1
            except Exception as error:
                print(error)
                print(f"Error evaluating sample {idx}: {error}")
                completion_scores.append(False)
            progress_bar.update(1)
        progress_bar.close()
        
        # Also extract code for display purposes
        for sample in samples:
            sample['pred_completions'] = [
                extract_answer_with_context(completion, data_name, sample) for completion in sample.get('completions', [])
            ]
    else:
        # For math/BBH: extract answer and compare with ground truth
        for sample in samples:
            sample['pred_completions'] = [
                extract_answer_with_context(completion, data_name, sample) for completion in sample.get('completions', [])
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

    # if there are random completions_random, also evaluate the correctness of each completion
    have_random_completions = 'completions_random' in samples[0]
    if have_random_completions:
        if data_name == "mbpp":
            # For MBPP: evaluate using test cases
            params_rand = [
                (idx, completion, sample)
                for idx, sample in enumerate(samples)
                for completion in sample.get('completions_random', [])
            ]
            completion_scores_random = []
            progress_bar = tqdm(total=len(params_rand), desc="Evaluate per-completion (random, MBPP)")
            for idx, completion, sample in params_rand:
                try:
                    result = mbpp_equal_process((idx, completion, sample))
                    completion_scores_random.append(result)
                except TimeoutError as error:
                    print(error)
                    completion_scores_random.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error)
                    print(f"Error evaluating random sample {idx}: {error}")
                    completion_scores_random.append(False)
                progress_bar.update(1)
            progress_bar.close()
            
            # Extract code for display
            for sample in samples:
                sample['pred_completions_random'] = [
                    extract_answer_with_context(c, data_name, sample) for c in sample.get('completions_random', [])
                ]
        else:
            # For math/BBH: extract answer and compare
            for sample in samples:
                sample['pred_completions_random'] = [
                    extract_answer_with_context(c, data_name, sample) for c in sample.get('completions_random', [])
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

    # if there are completions_slm, also evaluate the correctness of each completion
    have_slm_completions = 'completions_slm' in samples[0]
    if have_slm_completions:
        if data_name == "mbpp":
            # For MBPP: evaluate using test cases
            params_slm = [
                (idx, completion, sample)
                for idx, sample in enumerate(samples)
                for completion in sample.get('completions_slm', [])
            ]
            completion_scores_slm = []
            progress_bar = tqdm(total=len(params_slm), desc="Evaluate per-completion (slm-only, MBPP)")
            for idx, completion, sample in params_slm:
                try:
                    result = mbpp_equal_process((idx, completion, sample))
                    completion_scores_slm.append(result)
                except TimeoutError as error:
                    print(error)
                    completion_scores_slm.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error)
                    print(f"Error evaluating slm sample {idx}: {error}")
                    completion_scores_slm.append(False)
                progress_bar.update(1)
            progress_bar.close()
            
            # Extract code for display
            for sample in samples:
                sample['pred_completions_slm'] = [
                    extract_answer_with_context(c, data_name, sample) for c in sample.get('completions_slm', [])
                ]
        else:
            # For math/BBH: extract answer and compare
            for sample in samples:
                sample['pred_completions_slm'] = [
                    extract_answer_with_context(c, data_name, sample) for c in sample.get('completions_slm', [])
                ]
            params_slm = [
                (idx, pred, sample['gt'])
                for idx, sample in enumerate(samples)
                for pred in sample['pred_completions_slm']
            ]
            completion_scores_slm = []
            progress_bar = tqdm(total=len(params_slm), desc="Evaluate per-completion (slm-only)")
            for idx, pred, gt in params_slm:
                try:
                    result = math_equal_process((idx, pred, gt))
                    completion_scores_slm.append(result)
                except TimeoutError as error:
                    print(error)
                    completion_scores_slm.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error)
                    exit()
                progress_bar.update(1)
            progress_bar.close()
    else:
        completion_scores_slm = None

    # if there are completions_llm, also evaluate the correctness of each completion
    have_llm_completions = 'completions_llm' in samples[0]
    if have_llm_completions:
        if data_name == "mbpp":
            # For MBPP: evaluate using test cases
            params_llm = [
                (idx, completion, sample)
                for idx, sample in enumerate(samples)
                for completion in sample.get('completions_llm', [])
            ]
            completion_scores_llm = []
            progress_bar = tqdm(total=len(params_llm), desc="Evaluate per-completion (llm-only, MBPP)")
            for idx, completion, sample in params_llm:
                try:
                    result = mbpp_equal_process((idx, completion, sample))
                    completion_scores_llm.append(result)
                except TimeoutError as error:
                    print(error)
                    completion_scores_llm.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error)
                    print(f"Error evaluating llm sample {idx}: {error}")
                    completion_scores_llm.append(False)
                progress_bar.update(1)
            progress_bar.close()
            
            # Extract code for display
            for sample in samples:
                sample['pred_completions_llm'] = [
                    extract_answer_with_context(c, data_name, sample) for c in sample.get('completions_llm', [])
                ]
        else:
            # For math/BBH: extract answer and compare
            for sample in samples:
                sample['pred_completions_llm'] = [
                    extract_answer_with_context(c, data_name, sample) for c in sample.get('completions_llm', [])
                ]
            params_llm = [
                (idx, pred, sample['gt'])
                for idx, sample in enumerate(samples)
                for pred in sample['pred_completions_llm']
            ]
            completion_scores_llm = []
            progress_bar = tqdm(total=len(params_llm), desc="Evaluate per-completion (llm-only)")
            for idx, pred, gt in params_llm:
                try:
                    result = math_equal_process((idx, pred, gt))
                    completion_scores_llm.append(result)
                except TimeoutError as error:
                    print(error)
                    completion_scores_llm.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error)
                    exit()
                progress_bar.update(1)
            progress_bar.close()
    else:
        completion_scores_llm = None

    # fill back to samples
    # pred_keys matrix
    idx = 0
    score_mat = []
    for sample in samples:
        sample['correct'] = scores[idx: idx + len(sample['preds'])]
        assert len(sample['correct']) == len(sample['preds'])
        score_mat.append(sample['correct'])
        idx += len(sample['preds'])

    # the correctness of the main completions
    idx = 0
    for sample in samples:
        n = len(sample['pred_completions'])
        sample['correct_completions'] = completion_scores[idx: idx + n]
        assert len(sample['correct_completions']) == n
        idx += n

    # the correctness of the random completions
    if completion_scores_random is not None:
        idx = 0
        for sample in samples:
            n = len(sample.get('pred_completions_random', []))
            sample['correct_completions_random'] = completion_scores_random[idx: idx + n]
            assert len(sample['correct_completions_random']) == n
            idx += n

    # the correctness of the slm-only completions
    if completion_scores_slm is not None:
        idx = 0
        for sample in samples:
            n = len(sample.get('pred_completions_slm', []))
            sample['correct_completions_slm'] = completion_scores_slm[idx: idx + n]
            assert len(sample['correct_completions_slm']) == n
            idx += n

    # the correctness of the llm-only completions
    if completion_scores_llm is not None:
        idx = 0
        for sample in samples:
            n = len(sample.get('pred_completions_llm', []))
            sample['correct_completions_llm'] = completion_scores_llm[idx: idx + n]
            assert len(sample['correct_completions_llm']) == n
            idx += n

    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    # output mean of each column of scores
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['preds'][-1]]),
        "acc": {pred_key: score for pred_key, score in zip(pred_keys, mean_score)}
    }


    return samples, result_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--prompt_type", type=str, default="tool-integrated")
    parser.add_argument("--file_path", type=str, default=None)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(data_name=args.data_name, prompt_type=args.prompt_type, file_path=args.file_path,
             max_num_samples=args.max_num_samples, execute=args.execute)
