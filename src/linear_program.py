from .parser import parse_expression

class LinearProgram:
	def __init__(self):
		self.objective_function = {}
		self.constraints = []
		self.variables = set()
		self.is_maximization = true
		self.iteration_counter = 0

	def add_obj_fn(self, expression):
		self.objective_function = parser.parse_expression(expression) # Extract variables and their coef's in key, value pairs and push to obj fn dictionary 
		self.variables.update(self.objective_function.keys()) # We want to ensure that each variable in obj fn is accounted for so we union the variables found in obj fn with whatever variables were already there

	def add_cons(self, expression):
		constraint = parser.parse_expression(expression)
		self.constraints.append(constraint) 
		self.variables.update(constraint.keys()) # Union whatever variables were found in constraint with whatever variables we already have
		