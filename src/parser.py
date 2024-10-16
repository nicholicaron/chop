"""
Note: This parser is a fickle one and is therefore susceptible to abuse:
	Assumptions:
    - Order of variables is consistent,
    - Variables are only used once in an expression
    - Coef*Var pairs are interspersed by linear arithmetic operators
    - Ad infinitum
"""

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
    
class ObjectiveType(Enum):
    MAXIMIZE = 'max'
    MINIMIZE = 'min'

def parse_ineq(expression: str) -> Tuple[List[str], List[float], List[ArithmeticOp], ComparisonOp, float]:
    """
    Parse a linear inequality string into its components.

    Args:
    expression (str): A string containing a linear inequality (e.g., "2x + 3y - z + 5 <= 10")

    Returns:
    Tuple containing:
    - List of variable names (str)
    - List of coefficients (float)
    - List of arithmetic operations (ArithmeticOp)
    - Comparison operator (ComparisonOp)
    - Right-hand side value (float)
    """
    # Split the expression into left side and right side
    match = re.split(r'(<=|>=|<|>|=)', expression)
    if len(match) != 3:
        raise ValueError("Invalid expression format")

    left_side, cmp_op, right_side = match

    # Parse the comparison operator
    cmp_op = ComparisonOp(cmp_op)

    # Parse the left side
    variables, coefs, arithmetic_ops, constant = parse_expr(left_side)

    # Parse the right-hand side value and adjust for the constant
    value = float(right_side) - constant

    # Assertions to check array sizes
    assert len(variables) == len(coefs), "Number of variables and coefficients must be equal"
    assert len(arithmetic_ops) <= len(variables), "Number of arithmetic operations cannot be greater than the number of variables"

    return variables, coefs, arithmetic_ops, cmp_op, value

def parse_expr(expression: str) -> Tuple[List[str], List[float], List[ArithmeticOp], float]:
    """
    Parse a linear expression string into its components.
    This function is used by both parse_linear_inequality and parse_objective_function.

    Args:
    expression (str): A string containing a linear expression (e.g., "2x + 3y - z + 5")

    Returns:
    Tuple containing:
    - List of variable names (str)
    - List of coefficients (float)
    - List of arithmetic operations (ArithmeticOp)
    - Constant term (float)
    """
    # Remove all whitespace from the expression
    expression = re.sub(r'\s+', '', expression)

    # Split the expression into terms
    terms = re.findall(r'([+-]?\d*\.?\d*[a-zA-Z]|[+-]?\d+\.?\d*)', expression)

    variables = []
    coefs = []
    arithmetic_ops = []
    constant = 0.0

    for i, term in enumerate(terms):
        # Check if the term is a constant
        if re.match(r'^[+-]?\d+\.?\d*$', term):
            constant += float(term)
        else:
            # Extract coefficient and variable
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
            op = ArithmeticOp.ADD if term[0] != '-' else ArithmeticOp.SUBTRACT
            arithmetic_ops.append(op)

    return variables, coefs, arithmetic_ops, constant

def parse_objective_function(expression: str) -> Tuple[ObjectiveType, List[str], List[float], List[ArithmeticOp], float]:
    """
    Parse an objective function string into its components.

    Args:
    expression (str): A string containing an objective function (e.g., "max 2x + 3y - z + 5")

    Returns:
    Tuple containing:
    - Objective type (ObjectiveType)
    - List of variable names (str)
    - List of coefficients (float)
    - List of arithmetic operations (ArithmeticOp)
    - Constant term (float)
    """
    # Extract the objective type and the expression
    match = re.match(r'(max|min)\s+(.+)', expression, re.IGNORECASE)
    if not match:
        raise ValueError("Invalid objective function format. Must start with 'max' or 'min'.")

    obj_type, expr = match.groups()
    obj_type = ObjectiveType(obj_type.lower())

    # Parse the expression
    variables, coefs, arithmetic_ops, obj_value = parse_expr(expr)

    # Assertions to check array sizes
    assert len(variables) == len(coefs), "Number of variables and coefficients must be equal"
    assert len(arithmetic_ops) <= len(variables), "Number of arithmetic operations cannot be greater than the number of variables"

    return obj_type, variables, coefs, arithmetic_ops, obj_value

# Example usage:
inequality = "2x + 3y - z + 5 <= 10"
obj_function = "max 2x + 3y - z + 5"

ineq_result = parse_ineq(inequality)
obj_result = parse_objective_function(obj_function)

print("Inequality:", ineq_result)
print("Objective function:", obj_result)
