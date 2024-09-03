import re
from enum import Enum
from typing import List, Tuple

class ComparisonOp(Enum):
    LESS = '<'
    LEQ = '<='
    GREATER = '>'
    GEQ = '>='
    EQ = '='

class ArithmeticOp(Enum):
    ADD = '+'
    SUBTRACT = '-'
    

def parse_expr(expression: str) -> Tuple[List[str], List[float], List[ArithmeticOp], ComparisonOp, float]:
    """
    Parse a linear inequality string into its components.

    Args:
    expression (str): A string containing a linear inequality (e.g., "2x + 3y - z <= 10")

    Returns:
    Tuple containing:
    - List of variable names (str)
    - List of coefficients (float)
    - List of arithmetic operations (ArithmeticOp)
    - Comparison operator (ComparisonOp)
    - Right-hand side value (float)

    Assumptions:
    - The input string is well-formed and follows the pattern: coef*var op coef*var ... cmp_op value
    - Variables are single letters
    - Coefficients are integers or floats (including multiple digits)
    - The comparison operator is one of: <, <=, >, >=, =
    - There's exactly one comparison operator and one right-hand side value

    Future improvements:
    - Add input validation and error handling for malformed expressions
    - Support for multi-letter variable names
    - Handle more complex expressions (e.g., parentheses, fractions)
    """

    # Remove all whitespace from the expression
    expression = re.sub(r'\s+', '', expression)

    # Extract the comparison operator and right-hand side value
    match = re.split(r'(<=|>=|<|>|=)', expression)
    if len(match) != 3:
        raise ValueError("Invalid expression format")

    left_side, cmp_op, right_side = match

    # Parse the comparison operator
    cmp_op = ComparisonOp(cmp_op)

    # Parse the right-hand side value
    value = float(right_side)

    # Split the left side into terms
    terms = re.findall(r'([+-]?\d*\.?\d*[a-zA-Z])', left_side)

    variables = []
    coefs = []
    arithmetic_ops = []

    for i, term in enumerate(terms):
        # Extract coefficient and variable
        # This regex handles multiple-digit coefficients, including decimals
        match = re.match(r'([+-]?\d*\.?\d*)([a-zA-Z])', term)
        if match:
            coef, var = match.groups()
            
            # Handle implicit coefficients ('+x' -> '+1x', '-x' -> '-1x', 'x' -> '1x')
            if coef in ('+', '-', ''):
                coef += '1'
            
            coefs.append(float(coef))
            variables.append(var)

            # Determine the arithmetic operation
            if i > 0:
                op = ArithmeticOp.ADD if coef[0] != '-' else ArithmeticOp.SUBTRACT
                arithmetic_ops.append(op)

    # Assertions to check array sizes
    assert len(variables) == len(coefs), "Number of variables and coefficients must be equal"
    assert len(arithmetic_ops) == len(variables) - 1, "Number of arithmetic operations must be one less than the number of variables"

    return variables, coefs, arithmetic_ops, cmp_op, value

# Example usage:
# expression = "2x + 3.5y - 10z <= 10"
# result = parse_expression(expression)
# print(result)