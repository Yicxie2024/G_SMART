"""
MBPP (Mostly Basic Programming Problems) Evaluation Script

This script evaluates generated Python code against test cases from the MBPP dataset.
It supports pass@k metric calculation for code generation tasks.

Usage:
    python mbpp_eval.py --file_path /path/to/mbpp_results.jsonl --max_samples 100

Key Features:
    - Executes generated code against test cases
    - Computes pass@k metrics (pass@1, pass@5, pass@10)
    - Safe code execution with timeout protection
    - Handles multiple completions per problem
"""

import re
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from concurrent.futures import TimeoutError

from python_executor import PythonExecutor
from utils import load_jsonl


def extract_python_code(text: str) -> str:
    """
    Extract Python code from model output.
    Handles various formats:
    - Code blocks with ```python markers
    - Plain code without markers
    - Multiple code blocks (takes the first one)
    """
    # Try to find code in markdown code blocks
    pattern = r'```python\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Try without python specifier
    pattern = r'```\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # If no code blocks, try to extract function definition
    # Look for def or class at the beginning of a line
    lines = text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        if line.strip().startswith('def ') or line.strip().startswith('class '):
            in_code = True
        if in_code:
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines).strip()
    
    # Last resort: return the whole text
    return text.strip()


def run_test_case(code: str, test_case: str, executor: PythonExecutor, timeout: int = 5) -> Tuple[bool, str]:
    """
    Run a single test case against generated code.
    
    Args:
        code: Generated Python code
        test_case: Test assertion (e.g., "assert func(1) == 2")
        executor: PythonExecutor instance
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (passed: bool, error_message: str)
    """
    # Combine code with test case
    full_code = f"{code}\n{test_case}"
    
    try:
        result, report = executor.apply(full_code)
        
        # If execution completed without error, test passed
        if report == "Done":
            return True, ""
        else:
            return False, report
    except Exception as e:
        return False, str(e)


def evaluate_mbpp_sample(
    sample: Dict[str, Any],
    completions_key: str = 'completions',
    executor: PythonExecutor = None
) -> Dict[str, Any]:
    """
    Evaluate a single MBPP sample with multiple completions.
    
    Args:
        sample: Dictionary containing problem, test_cases, and completions
        completions_key: Key to access completions in sample dict
        executor: PythonExecutor instance for code execution
        
    Returns:
        Dictionary with evaluation results including pass@k scores
    """
    if executor is None:
        executor = PythonExecutor(timeout_length=5)
    
    # Get test cases
    test_cases = sample.get('test_list', sample.get('test_cases', []))
    if isinstance(test_cases, str):
        # Sometimes test cases are stored as a single string
        test_cases = [tc.strip() for tc in test_cases.split('\n') if tc.strip().startswith('assert')]
    
    # Get completions
    completions = sample.get(completions_key, [])
    if not completions:
        return {
            'passed': [],
            'num_passed': 0,
            'pass_rate': 0.0,
            'error': 'No completions found'
        }
    
    # Evaluate each completion against all test cases
    results = []
    for completion in completions:
        code = extract_python_code(completion)
        
        # Test if all test cases pass for this completion
        all_passed = True
        errors = []
        
        for test_case in test_cases:
            passed, error = run_test_case(code, test_case, executor)
            if not passed:
                all_passed = False
                errors.append(error)
        
        results.append({
            'passed': all_passed,
            'code': code,
            'errors': errors
        })
    
    # Calculate statistics
    num_passed = sum(1 for r in results if r['passed'])
    pass_rate = num_passed / len(results) if results else 0.0
    
    return {
        'passed': [r['passed'] for r in results],
        'num_passed': num_passed,
        'total_completions': len(results),
        'pass_rate': pass_rate,
        'details': results
    }


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k metric.
    
    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k in pass@k
        
    Returns:
        pass@k score
        
    Formula from "Evaluating Large Language Models Trained on Code" (Chen et al. 2021)
    pass@k = E[1 - comb(n-c, k) / comb(n, k)]
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def evaluate_mbpp(
    file_path: str = None,
    samples: List[Dict] = None,
    max_samples: int = None,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, Any]:
    """
    Evaluate MBPP results and compute pass@k metrics.
    
    Args:
        file_path: Path to JSONL file with MBPP results
        samples: List of samples (alternative to file_path)
        max_samples: Maximum number of samples to evaluate
        k_values: List of k values for pass@k calculation
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load samples
    if samples is None:
        if file_path is None:
            raise ValueError("Either file_path or samples must be provided")
        samples = list(load_jsonl(file_path))
    
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"Evaluating {len(samples)} MBPP samples...")
    
    # Initialize executor
    executor = PythonExecutor(timeout_length=5)
    
    # Evaluate each sample
    results = []
    for sample in tqdm(samples, desc="Evaluating MBPP"):
        result = evaluate_mbpp_sample(sample, executor=executor)
        results.append(result)
        sample['eval_result'] = result
    
    # Calculate pass@k for different k values
    pass_at_k_scores = {}
    
    for k in k_values:
        total_pass_at_k = []
        
        for result in results:
            n = result['total_completions']
            c = result['num_passed']
            
            if n >= k:
                pass_at_k_val = calculate_pass_at_k(n, c, k)
                total_pass_at_k.append(pass_at_k_val)
        
        if total_pass_at_k:
            pass_at_k_scores[f'pass@{k}'] = np.mean(total_pass_at_k) * 100
        else:
            pass_at_k_scores[f'pass@{k}'] = 0.0
    
    # Calculate simple accuracy (at least one correct)
    accuracy = np.mean([1 if r['num_passed'] > 0 else 0 for r in results]) * 100
    
    # Calculate average pass rate across all completions
    avg_pass_rate = np.mean([r['pass_rate'] for r in results]) * 100
    
    metrics = {
        'num_samples': len(samples),
        'accuracy': accuracy,  # At least one correct
        'avg_pass_rate': avg_pass_rate,  # Average across all completions
        **pass_at_k_scores
    }
    
    print("\n" + "="*50)
    print("MBPP Evaluation Results")
    print("="*50)
    print(f"Number of samples: {metrics['num_samples']}")
    print(f"Accuracy (≥1 correct): {metrics['accuracy']:.2f}%")
    print(f"Average pass rate: {metrics['avg_pass_rate']:.2f}%")
    for k in k_values:
        if f'pass@{k}' in metrics:
            print(f"Pass@{k}: {metrics[f'pass@{k}']:.2f}%")
    print("="*50)
    
    return samples, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MBPP code generation results")
    parser.add_argument("--file_path", type=str, help="Path to JSONL file with results")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--k_values", type=int, nargs='+', default=[1, 5, 10], 
                        help="K values for pass@k calculation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    samples, metrics = evaluate_mbpp(
        file_path=args.file_path,
        max_samples=args.max_samples,
        k_values=args.k_values
    )

