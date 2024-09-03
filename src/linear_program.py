from parser import parse_ineq, parse_objective_function, ArithmeticOp, ComparisonOp, ObjectiveType
from collections import Tuple, List

class LinearProgram:
	def __init__(self):
		self.objective_function: Tuple[ObjectiveType, List[str], List[float], List[ArithmeticOp], float] = () # (objective_type, variables, coefs, arithmetic_ops, objective_value)
		self.constraints: List[Tuple[List[str], List[float], List[ArithmeticOp], ComparisonOp, float]] = [] # List[(variables, coefs, arithmetic_ops, comparison_operator, rhs value)]
		self.variables = set()
		self.type = ObjectiveType.MAXIMIZE # Standard LP: Max
		self.iteration_counter = 0 # This will come in handy later when generating the ML dataset

	def add_obj_fn(self, expression):
		obj_fn = parse_objective_function(expression)
		self.objective_function.append(obj_fn) 
		self.variables.update(obj_fn[1]) # We want to ensure that each variable in obj fn is accounted for so we union the variables found in obj fn with whatever variables were already there

	def add_cons(self, expression):
		constraint = parse_ineq(expression)
		self.constraints.append(constraint) 
		self.variables.update(constraint[0]) # Union whatever variables were found in constraint with whatever variables we already have
