"""
MBPP Code Grading Module
Evaluates Python code by executing test cases from MBPP dataset
"""

import re
from typing import Tuple, List
from python_executor import PythonExecutor


def extract_python_code(completion: str) -> str:
    """
    Extract Python code from model completion.
    Handles various formats:
    - Code blocks with ```python markers
    - Plain code without markers
    - Multiple code blocks (takes the last one)
    """
    # Try to find code in markdown code blocks with python specifier
    pattern = r'```python\s*(.*?)```'
    matches = re.findall(pattern, completion, re.DOTALL)
    
    if matches:
        # Return the last code block (most likely the final solution)
        return matches[-1].strip()
    
    # Try without python specifier
    pattern = r'```\s*(.*?)```'
    matches = re.findall(pattern, completion, re.DOTALL)
    
    if matches:
        # Return the last code block
        return matches[-1].strip()
    
    # If no code blocks, try to extract function definition
    # Look for def or class at the beginning of a line
    lines = completion.split('\n')
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
    return completion.strip()


def extract_function_name(code: str) -> str:
    """Extract the main function name from Python code."""
    pattern = r'def\s+(\w+)\s*\('
    match = re.search(pattern, code)
    if match:
        return match.group(1)
    return None


def rename_function_in_code(code: str, old_name: str, new_name: str) -> str:
    """
    Rename a function in Python code.
    Handles function definition and all calls to it.
    """
    # Replace function definition
    code = re.sub(
        rf'\bdef\s+{re.escape(old_name)}\s*\(',
        f'def {new_name}(',
        code
    )
    return code


def extract_expected_function_name(test_cases: List[str]) -> str:
    """
    Extract the expected function name from test cases.
    
    Args:
        test_cases: List of assert statements like ["assert func(x) == y", ...]
        
    Returns:
        The function name used in test cases
    """
    if not test_cases:
        return None
    
    # Look at the first test case
    first_test = test_cases[0]
    
    # Pattern to match function calls in assert statements
    # E.g., "assert remove_Occ(...)" -> "remove_Occ"
    pattern = r'assert\s+(\w+)\s*\('
    match = re.search(pattern, first_test)
    
    if match:
        return match.group(1)
    
    return None


def align_function_names(code: str, test_cases: List[str]) -> Tuple[str, bool]:
    """
    Align function name in code with the one expected by test cases.
    
    Args:
        code: Generated Python code
        test_cases: List of test assertions
        
    Returns:
        (modified_code, was_renamed): Tuple of aligned code and whether renaming occurred
    """
    generated_func_name = extract_function_name(code)
    expected_func_name = extract_expected_function_name(test_cases)
    
    if not generated_func_name or not expected_func_name:
        return code, False
    
    if generated_func_name == expected_func_name:
        return code, False
    
    # Rename the generated function to match test expectations
    aligned_code = rename_function_in_code(code, generated_func_name, expected_func_name)
    return aligned_code, True


def execute_mbpp_test_cases(
    code: str,
    test_cases: List[str],
    test_setup_code: str = "",
    timeout: int = 5
) -> Tuple[bool, List[str]]:
    """
    Execute MBPP test cases against generated code.
    
    Args:
        code: Generated Python code
        test_cases: List of test assertions (e.g., ["assert func(1) == 2", ...])
        test_setup_code: Setup code to run before tests
        timeout: Timeout in seconds
        
    Returns:
        (all_passed, errors): Tuple of overall success and list of error messages
    """
    executor = PythonExecutor(timeout_length=timeout)
    
    errors = []
    all_passed = True
    
    for test_case in test_cases:
        # Combine setup code, function code, and test case
        full_code = ""
        if test_setup_code:
            full_code += test_setup_code + "\n"
        full_code += code + "\n"
        full_code += test_case
        
        try:
            result, report = executor.apply(full_code)
            
            # If execution completed without error, test passed
            if report == "Done":
                continue  # Test passed
            else:
                all_passed = False
                errors.append(f"Test '{test_case}' failed: {report}")
        except Exception as e:
            all_passed = False
            errors.append(f"Test '{test_case}' raised exception: {str(e)}")
    
    return all_passed, errors


def grade_mbpp_completion(
    completion: str,
    test_cases: List[str],
    test_setup_code: str = "",
    align_names: bool = True,
    timeout: int = 5
) -> Tuple[bool, dict]:
    """
    Grade a single MBPP completion.
    
    Args:
        completion: Model's complete output
        test_cases: List of test assertions from MBPP dataset
        test_setup_code: Setup code from MBPP dataset
        align_names: Whether to automatically align function names
        timeout: Timeout in seconds
        
    Returns:
        (passed, details): Tuple of whether all tests passed and detailed results
    """
    # Step 1: Extract Python code from completion
    code = extract_python_code(completion)
    
    if not code or 'def ' not in code:
        return False, {
            'error': 'No valid Python function found in completion',
            'code_extracted': code[:200] if code else None
        }
    
    # Step 2: Align function names if needed
    was_renamed = False
    if align_names:
        code, was_renamed = align_function_names(code, test_cases)
    
    # Step 3: Execute test cases
    all_passed, errors = execute_mbpp_test_cases(
        code, test_cases, test_setup_code, timeout
    )
    
    return all_passed, {
        'code_extracted': code,
        'was_renamed': was_renamed,
        'tests_passed': all_passed,
        'num_tests': len(test_cases),
        'errors': errors if errors else None
    }


def mbpp_equal_process(param):
    """
    Process function for MBPP evaluation (compatible with evaluate.py interface).
    
    Args:
        param: Tuple of (idx, completion, sample_dict)
               - idx: sample index
               - completion: the completion text or extracted code
               - sample_dict: sample containing 'test_list', 'test_setup_code', etc.
               
    Returns:
        bool: Whether all test cases passed
    """
    try:
        idx, completion, sample = param
        
        # Get test cases from sample
        test_cases = sample.get('test_list', [])
        test_setup_code = sample.get('test_setup_code', '')
        
        if not test_cases:
            return False
        
        # Grade the completion
        passed, details = grade_mbpp_completion(
            completion,
            test_cases,
            test_setup_code,
            align_names=True,
            timeout=5
        )
        
        return passed
    
    except Exception as e:
        print(f"Error in mbpp_equal_process: {e}")
        return False


# Test function
def _test():
    # Test case 1: Correct code with matching function name
    code1 = """
def remove_Occ(s, ch):
    for i in range(len(s)):
        if s[i] == ch:
            s = s[0:i] + s[i + 1:]
            break
    for i in range(len(s) - 1, -1, -1):
        if s[i] == ch:
            s = s[0:i] + s[i + 1:]
            break
    return s
"""
    
    # Test case 2: Correct code with different function name
    completion2 = """
```python
def remove_first_last(s, ch):
    for i in range(len(s)):
        if s[i] == ch:
            s = s[0:i] + s[i + 1:]
            break
    for i in range(len(s) - 1, -1, -1):
        if s[i] == ch:
            s = s[0:i] + s[i + 1:]
            break
    return s
```
"""
    
    test_cases = [
        'assert remove_Occ("hello","l") == "heo"',
        'assert remove_Occ("abcda","a") == "bcd"',
        'assert remove_Occ("PHP","P") == "H"'
    ]
    
    print("=" * 80)
    print("Test 1: Correct code with matching name")
    print("=" * 80)
    passed1, details1 = grade_mbpp_completion(code1, test_cases)
    print(f"Passed: {passed1}")
    print(f"Details: {details1}")
    
    print("\n" + "=" * 80)
    print("Test 2: Correct code with different name (should auto-rename)")
    print("=" * 80)
    passed2, details2 = grade_mbpp_completion(completion2, test_cases)
    print(f"Passed: {passed2}")
    print(f"Was renamed: {details2.get('was_renamed')}")
    print(f"Details: {details2}")


if __name__ == '__main__':
    _test()

