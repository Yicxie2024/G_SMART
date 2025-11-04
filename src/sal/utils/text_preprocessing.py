#!/usr/bin/env python
"""
Text preprocessing utilities for embedding models.
Normalizes mathematical expressions, numbers, and LaTeX formatting.
"""

import re
from typing import List, Union


def normalize_text_for_embedding(text: str, large_num_threshold: int = 1000) -> str:
    """
    Normalize text before embedding to reduce noise and improve semantic consistency.
    
    Steps:
    1. Normalize LaTeX to plain text format (e.g., \frac{a}{b} -> a/b, ^ -> **)
    2. Replace large numbers with <NUM> placeholder (preserves structure, reduces noise)
    3. Normalize mathematical expressions using SymPy (if available)
    
    Args:
        text: Input text to normalize
        large_num_threshold: Threshold above which numbers are replaced with <NUM>
        
    Returns:
        Normalized text string
    """
    if not text:
        return text
    
    # Step 1: Normalize LaTeX formatting to plain text
    text = _normalize_latex(text)
    
    # Step 2: Replace large numbers with <NUM> placeholder
    text = _replace_large_numbers(text, large_num_threshold)
    
    # Step 3: Normalize mathematical expressions with SymPy (if available)
    text = _normalize_math_expressions(text)
    
    return text


def _normalize_latex(text: str) -> str:
    """
    Convert LaTeX formatting to plain text equivalents.
    
    Conversions:
    - \frac{a}{b} -> a/b
    - ^{...} or ^... -> **
    - Remove LaTeX style commands
    - Normalize other LaTeX constructs
    """
    # Replace \frac{a}{b} with a/b
    # Handle both \frac{a}{b} and \frac{a}{b} with various spacing
    def replace_frac(match):
        num = match.group(1).strip()
        den = match.group(2).strip()
        return f"({num})/({den})"
    
    # Match \frac{...}{...} or \frac ... ... 
    text = re.sub(r'\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}', replace_frac, text)
    
    # Replace ^{...} or ^... with **
    # First handle ^{...} form
    text = re.sub(r'\^\s*\{([^}]+)\}', r'**(\1)', text)
    # Then handle simple ^n form (but be careful with ^2, ^3, etc.)
    text = re.sub(r'\^(\d+)', r'**\1', text)
    # Handle ^{expr} already handled above
    
    # Remove LaTeX style commands like \text, \mathrm, \mathit, etc.
    text = re.sub(r'\\(?:text|mathrm|mathit|mathbf|mathcal|mathbb)\s*\{([^}]+)\}', r'\1', text)
    
    # Replace other common LaTeX commands that don't affect meaning
    text = re.sub(r'\\cdot', '*', text)
    text = re.sub(r'\\times', '*', text)
    text = re.sub(r'\\div', '/', text)
    text = re.sub(r'\\pm', '±', text)
    text = re.sub(r'\\mp', '∓', text)
    
    # Remove LaTeX braces that are just for grouping
    # But be careful: only remove braces that contain simple content (no nested braces)
    # This preserves structure while removing unnecessary grouping
    def remove_simple_braces(match):
        content = match.group(1)
        # Only remove if content doesn't contain nested braces or complex commands
        if '{' not in content and '}' not in content and '\\' not in content:
            return content
        return match.group(0)  # Keep original if complex
    
    # Remove simple braces (but preserve those in fractions/exponents which we already handled)
    # Use a more conservative pattern that avoids already processed structures
    text = re.sub(r'\{([^{}]+)\}', remove_simple_braces, text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def _replace_large_numbers(text: str, threshold: int = 1000) -> str:
    """
    Replace large numbers with <NUM> placeholder while preserving structure.
    
    This reduces numerical noise in embeddings while maintaining the semantic
    structure of the expression.
    """
    def replace_num(match):
        try:
            # Try to parse as integer first
            num_str = match.group(0)
            if '.' in num_str or 'e' in num_str.lower() or 'E' in num_str:
                # Float or scientific notation
                num = float(num_str)
            else:
                # Integer
                num = int(num_str)
            
            # Check if number exceeds threshold (consider absolute value)
            if abs(num) >= threshold:
                return '<NUM>'
            else:
                return num_str
        except (ValueError, AttributeError):
            # If parsing fails, return original
            return match.group(0)
    
    # Pattern to match numbers (integers, floats, scientific notation)
    # Match integers and floats (including negative)
    # Pattern: optional minus, digits, optional decimal point and more digits
    # Also match scientific notation like 1e10, 1.5E-3, etc.
    pattern = r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
    text = re.sub(pattern, replace_num, text)
    
    return text


def _normalize_math_expressions(text: str) -> str:
    """
    Normalize mathematical expressions using SymPy.
    
    Attempts to:
    - Simplify expressions (reduce fractions, combine like terms)
    - Order polynomials in canonical form
    - Normalize equivalent expressions
    
    Falls back gracefully if SymPy is not available or if parsing fails.
    
    Note: This function is conservative and only normalizes expressions that
    can be safely parsed and simplified. It preserves the overall structure
    of the text while normalizing mathematical subexpressions.
    """
    try:
        from sympy import sympify, simplify
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
        
        # Only try to normalize if text looks like it contains mathematical expressions
        # Simple heuristic: check for common math operators or patterns
        math_pattern = r'[+\-*/=()\[\]]|[0-9]+\s*[+\-*/]\s*[0-9]+|<NUM>'
        if not re.search(math_pattern, text):
            return text
        
        # Try to find and normalize standalone mathematical expressions
        # Match patterns like: number operators, expressions in parentheses, etc.
        # This is a conservative approach - we'll try to normalize isolated expressions
        
        # Pattern to match potential math expressions
        # Matches things like: "x + y", "2*3", "(a+b)", "x^2", etc.
        # But be careful not to match too broadly (e.g., natural language)
        
        def normalize_expr(match):
            expr_str = match.group(0).strip()
            if not expr_str:
                return expr_str
            
            # Skip if it's too short or looks like natural language
            if len(expr_str) < 3:
                return expr_str
            
            try:
                # Try to parse as a mathematical expression
                # Use transformations that handle implicit multiplication and standard conversions
                transformations = (standard_transformations + 
                                 (implicit_multiplication_application,))
                
                expr = parse_expr(expr_str, transformations=transformations, evaluate=False)
                
                # Try to simplify
                try:
                    simplified = simplify(expr)
                    # Only use simplified if it's actually different and shorter/more canonical
                    simplified_str = str(simplified)
                    if simplified_str != expr_str and len(simplified_str) <= len(expr_str) * 2:
                        return simplified_str
                    return expr_str
                except:
                    return expr_str
            except:
                # If parsing fails, return original
                return expr_str
        
        # Try to find isolated mathematical expressions
        # Pattern: sequences with operators and numbers/variables
        # This is conservative - only match clear math expressions
        math_expr_pattern = r'(?:[a-zA-Z_][a-zA-Z0-9_]*|<NUM>|[0-9]+)\s*[+\-*/]\s*(?:[a-zA-Z_][a-zA-Z0-9_]*|<NUM>|[0-9]+)'
        
        # Replace expressions that can be normalized
        text = re.sub(math_expr_pattern, normalize_expr, text)
        
        return text
    
    except ImportError:
        # SymPy not available, return original text
        return text
    except Exception:
        # Any other error, return original text
        return text


def normalize_texts_for_embedding(
    texts: List[str], 
    large_num_threshold: int = 1000
) -> List[str]:
    """
    Normalize a list of texts for embedding.
    
    Args:
        texts: List of text strings to normalize
        large_num_threshold: Threshold above which numbers are replaced with <NUM>
        
    Returns:
        List of normalized text strings
    """
    return [normalize_text_for_embedding(text, large_num_threshold) for text in texts]

