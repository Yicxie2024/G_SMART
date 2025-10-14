"""
MMLU-Pro specific parsing functions
Extends the parser.py with MMLU-Pro specific functionality
"""

import re
import regex
from typing import Dict, Any


def extract_mmlu_pro_answer(pred_str: str) -> str:
    """
    Extract answer from MMLU-Pro predictions.
    Supports answers in format: (A), A, "A", etc.
    Handles 10 options (A-J)
    """
    pred_str = pred_str.strip()
    
    # Common answer patterns
    patterns = [
        r"[Tt]herefore,?\s*(?:the\s*)?answer\s*is\s*\(?([A-J])\)?",  # "Therefore, the answer is (A)"
        r"[Tt]he\s*(?:correct\s*)?answer\s*is\s*\(?([A-J])\)?",      # "The answer is A"
        r"[Tt]he\s*(?:best\s*)?choice\s*is\s*\(?([A-J])\)?",         # "The choice is (A)"
        r"^answer:\s*\(?([A-J])\)?",                                  # "Answer: A"
        r"\*\*\(?([A-J])\)?\*\*",                                     # **A**
    ]
    
    # Try each pattern
    for pattern in patterns:
        match = re.search(pattern, pred_str, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # If no pattern matches, look for standalone capital letters A-J
    # Prefer letters at the end of the response
    matches = re.findall(r"\b([A-J])\b", pred_str.upper())
    if matches:
        return matches[-1]  # Return the last match
    
    # If still no match, try to find in parentheses
    matches = re.findall(r"\(([A-J])\)", pred_str.upper())
    if matches:
        return matches[-1]
    
    return ""  # Return empty if no answer found


def parse_mmlu_pro_question(example: Dict[str, Any]) -> str:
    """
    Format MMLU-Pro question with options.
    MMLU-Pro typically has 10 options (A-J).
    """
    question = example.get("question", "").strip()
    options = example.get("options", example.get("choices", []))
    
    if not options:
        return question
    
    # Format options as (A) option1 (B) option2 ...
    formatted_options = []
    option_letters = "ABCDEFGHIJ"[:len(options)]  # Support up to 10 options
    
    for i, (letter, option) in enumerate(zip(option_letters, options)):
        formatted_options.append(f"({letter}) {str(option).strip()}")
    
    options_str = " ".join(formatted_options)
    full_question = f"{question}\nAnswer Choices: {options_str}"
    
    return full_question


def parse_mmlu_pro_ground_truth(example: Dict[str, Any]) -> tuple:
    """
    Parse ground truth answer for MMLU-Pro.
    Returns: (ground_truth_cot, ground_truth_answer)
    """
    # MMLU-Pro doesn't have CoT, only the answer
    gt_cot = None
    
    # Answer could be in different formats
    answer = example.get("answer", example.get("answer_index", None))
    
    if answer is None:
        raise ValueError(f"No answer field found in example: {example.keys()}")
    
    # If answer is an integer (index), convert to letter
    if isinstance(answer, int):
        option_letters = "ABCDEFGHIJ"
        if 0 <= answer < 10:
            gt_ans = option_letters[answer]
        else:
            raise ValueError(f"Invalid answer index: {answer}")
    else:
        # If answer is already a letter string
        gt_ans = str(answer).strip().upper()
        # Ensure it's a single letter A-J
        if len(gt_ans) == 1 and gt_ans in "ABCDEFGHIJ":
            pass
        else:
            # Try to extract letter from string like "(A)" or "A."
            match = re.search(r"([A-J])", gt_ans)
            if match:
                gt_ans = match.group(1)
            else:
                raise ValueError(f"Invalid answer format: {answer}")
    
    return gt_cot, gt_ans


def get_mmlu_pro_system_prompt(style: str = "detailed") -> str:
    """
    Get system prompt for MMLU-Pro.
    
    Args:
        style: "detailed", "simple", or "cot" (chain-of-thought)
    """
    if style == "detailed":
        return (
            "You are an expert in multiple domains including science, technology, "
            "engineering, mathematics, humanities, and social sciences. "
            "Answer the following multiple-choice question by:\n"
            "1. Reading the question and all options carefully\n"
            "2. Analyzing each option systematically\n"
            "3. Eliminating incorrect choices with clear reasoning\n"
            "4. Selecting the best answer\n\n"
            "Format your response to end with: Therefore, the answer is (X) "
            "where X is the option letter from A to J."
        )
    elif style == "simple":
        return (
            "Answer the following multiple-choice question with clear reasoning. "
            "End your response with: Therefore, the answer is (X) where X is the option letter."
        )
    elif style == "cot":
        return (
            "Answer the following multiple-choice question using step-by-step reasoning:\n\n"
            "Step 1: Understand the question\n"
            "Step 2: Analyze each option\n"
            "Step 3: Eliminate wrong answers\n"
            "Step 4: Select the best answer\n\n"
            "Conclude with: Therefore, the answer is (X)."
        )
    else:
        raise ValueError(f"Unknown style: {style}")


# Test function
def test_mmlu_pro_parser():
    """Test the parsing functions"""
    
    # Test answer extraction
    test_cases = [
        ("Therefore, the answer is (B)", "B"),
        ("The correct answer is A", "A"),
        ("I think the best choice is (D).", "D"),
        ("**C**", "C"),
        ("Answer: J", "J"),
        ("After analysis, the answer should be E.", "E"),
    ]
    
    print("Testing answer extraction:")
    for text, expected in test_cases:
        result = extract_mmlu_pro_answer(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text[:40]}...' -> {result} (expected: {expected})")
    
    # Test question formatting
    example = {
        "question": "What is the capital of France?",
        "options": ["London", "Paris", "Berlin", "Madrid", "Rome", 
                   "Vienna", "Athens", "Lisbon", "Warsaw", "Prague"]
    }
    formatted = parse_mmlu_pro_question(example)
    print(f"\nFormatted question:\n{formatted}")
    
    # Test ground truth parsing
    print("\nTesting ground truth parsing:")
    test_gt_cases = [
        ({"answer": 1}, "B"),
        ({"answer": "A"}, "A"),
        ({"answer": "(C)"}, "C"),
        ({"answer_index": 0}, "A"),
    ]
    
    for example, expected in test_gt_cases:
        try:
            _, result = parse_mmlu_pro_ground_truth(example)
            status = "✓" if result == expected else "✗"
            print(f"  {status} {example} -> {result} (expected: {expected})")
        except Exception as e:
            print(f"  ✗ {example} -> Error: {e}")


if __name__ == "__main__":
    test_mmlu_pro_parser()

