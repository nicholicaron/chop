from enum import Enum
from typing import List, Tuple, Dict, Set
from linear_program import LinearProgram
from parser import ObjectiveType, ComparisonOp, ArithmeticOp
from exceptions import UnboundedError, InfeasibleError, DegeneracyError

class Simplex:
    def __init__(self, lp: LinearProgram):
        self.lp = lp
        self.lp = lp
        self.basic_variables: List[str] = []
        self.non_basic_variables: List[str] = []
        self.objective_value: float = 0
        self.solution: Dict[str, float] = {}

    def solve(self) -> Tuple[Dict[str, float], float]:
        """
        Solve the linear program using the simplex method.
        
        Returns:
        Tuple[Dict[str, float], float]: A tuple containing:
            - The optimal solution (a dictionary mapping variables to values)
            - The optimal objective value
        
        Raises:
        UnboundedError: If the problem is unbounded
        InfeasibleError: If the problem is infeasible
        """
        self.initialize_solution()
        
        max_iterations = 1000  # Prevent infinite loops
        for _ in range(max_iterations):
            if self.is_optimal():
                return self.extract_solution()
            
            entering_var = self.choose_entering()
            if entering_var is None:
                # No entering variable found, solution is optimal
                return self.extract_solution()
            
            try:
                leaving_var = self.choose_leaving(entering_var)
            except UnboundedError as e:
                raise e
            
            self.pivot(entering_var, leaving_var)
            
            if not self.is_feasible():
                raise InfeasibleError("The problem became infeasible during solving.")
        
        raise RuntimeError("Simplex method did not converge within the maximum number of iterations.")

    def initialize_solution(self) -> Dict[str, float]:
        """
        Initialize the simplex dictionary with slack variables.
        
        Returns:
            Dict[str, float]: The initial basic feasible solution
        """
        slack_counter = 1
        for i, constraint in enumerate(self.lp.constraints):
            variables, coefs, arithmetic_ops, comparison_op, rhs = constraint
            
            # Add slack variable and increment slack counter
            slack_var = f"w{slack_counter}"
            slack_counter += 1
            
            # Manipulate the constraint expression
            if comparison_op in [ComparisonOp.LEQ, ComparisonOp.LESS]:
                variables.append(slack_var)
                coefs.append(1.0) # Need to double check this...
            elif comparison_op in [ComparisonOp.GEQ, ComparisonOp.GREATER]:
                variables.append(slack_var)
                coefs.append(-1.0) # Need to double check this...
                
						# Set comparison operator to equality
            comparison_op = ComparisonOp.EQ
            
            # Negate variable coefficients (isolate slack variable)
            coefs = [-coef for coef in coefs[:-1]] + [coefs[-1]]
            
            # Update the constraint in the linear program
            self.lp.constraints[i] = (variables, coefs, arithmetic_ops, comparison_op, rhs)
            
            # Add slack variable to basic variables and set its value
            self.basic_variables.append(slack_var)
            self.solution[slack_var] = rhs

        # Add original variables to non-basic variables and set their values to 0
        for var in self.lp.variables:
            if var not in self.non_basic_variables:
                self.non_basic_variables.append(var)
                self.solution[var] = 0.0

    def is_optimal(self) -> bool:
        """
        Check if the current solution is optimal.
        
        For maximization problems, the solution is optimal if all coefficients
        in the objective function are non-positive.
        For minimization problems, the solution is optimal if all coefficients
        in the objective function are non-negative.
        
        Returns:
        bool: True if the current solution is optimal, False otherwise.
        """
        obj_type, variables, coefs, _, _ = self.lp.objective_function
        
        if obj_type == ObjectiveType.MAXIMIZE:
            # For maximization, all coefficients should be non-positive
            return all(coef <= 0 for var, coef in zip(variables, coefs) if var in self.non_basic_variables)
        elif obj_type == ObjectiveType.MINIMIZE:
            # For minimization, all coefficients should be non-negative
            return all(coef >= 0 for var, coef in zip(variables, coefs) if var in self.non_basic_variables)
        else:
            raise ValueError(f"Unknown objective type: {obj_type}")

    def choose_entering(self) -> str:
        """
        Choose the entering variable for the next iteration.
        
        Returns:
        str: The name of the entering variable, or None if the solution is optimal.
        """
        obj_type, variables, coefs, _, _ = self.lp.objective_function
        
        best_var = None # What does it mean if this stays as None? Is the problem optimal? 
        best_coef = 0 # Is this a safe initialization???
        
        for var, coef in zip(variables, coefs):
            if var in self.non_basic_variables:
                if obj_type == ObjectiveType.MAXIMIZE and coef > best_coef:
                    best_var = var
                    best_coef = coef
                elif obj_type == ObjectiveType.MINIMIZE and coef < best_coef:
                    best_var = var
                    best_coef = coef
        
        return best_var

    def choose_leaving(self, entering_var: str) -> str: 
        """
        Choose the leaving variable for the next iteration.
        
        Args:
        entering_var (str): The name of the entering variable.
        
        Returns:
        str: The name of the leaving variable.
        
        Raises:
        UnboundedError: If the problem is unbounded.
        
        Note:
        This current implementation does not handle degenerate cases to prevent cycling
        Assumes LP is in standard form
        """
        min_ratio = float('inf') 
        leaving_var = None
        
        for constraint in self.lp.constraints:
            variables, coefs, _, _, rhs = constraint
            if entering_var in variables:
                entering_coef = coefs[variables.index(entering_var)]
                if entering_coef > 0:
                    for var, coef in zip(variables, coefs):
                        if var in self.basic_variables:
                            ratio = self.solution[var] / coef
                            if 0 < ratio < min_ratio:
                                min_ratio = ratio
                                leaving_var = var
        
        if leaving_var is None:
            raise UnboundedError(f"The problem is unbounded. Variable {entering_var} can increase indefinitely.")
        
        return leaving_var

    def pivot(self, entering_var: str, leaving_var: str, solution: Dict[str, float]):
        """
        Perform a pivot operation.
        
        Args:
            entering_var (str): The entering variable
            leaving_var (str): The leaving variable
            solution (Dict[str, float]): The current solution
        """
        """
        Pseudo-implementation:
					# 1. Identify the pivot row (the constraint where leaving_var is basic)
    				pivot_row = find constraint where leaving_var is basic
    				pivot_coefficient = coefficient of entering_var in pivot_row

    			# 2. Update the basic and non-basic variable sets
    				remove entering_var from non_basic_variables
    				add entering_var to basic_variables
    				remove leaving_var from basic_variables
    				add leaving_var to non_basic_variables

    			# 3. Update the pivot row
    				divide all coefficients in pivot_row by pivot_coefficient
    				set coefficient of entering_var to 1 in pivot_row
    				set coefficient of leaving_var to 0 in pivot_row

    			# 4. Update all other constraints
    				for each constraint != pivot_row:
        			factor = coefficient of entering_var in this constraint
        			subtract (factor * pivot_row) from this constraint
        			set coefficient of entering_var to 0 in this constraint

    			# 5. Update the objective function
    				factor = coefficient of entering_var in objective function
    				subtract (factor * pivot_row) from objective function
    				set coefficient of entering_var to 0 in objective function

    			# 6. Update the solution
    				solution[entering_var] = pivot_row's RHS
    				for each basic_var != entering_var:
        			update solution[basic_var] based on new constraint coefficients

    			# 7. Update the objective value
    				recalculate objective_value based on new basic variable values
        """
        # This is a placeholder for the actual pivot implementation
				pass
    
    def extract_solution(self) -> Tuple[Dict[str, float], float]:
        """
        Extract the final solution from the simplex dictionary.
        
        Returns:
        Tuple[Dict[str, float], float]: A tuple containing:
            - A dictionary mapping variable names to their optimal values
            - The optimal objective value
        """
        optimal_solution = {}
        
        # Extract values for basic variables
        for var in self.basic_variables:
            optimal_solution[var] = self.solution[var]
        
        # Set non-basic variables to zero
        for var in self.non_basic_variables:
            optimal_solution[var] = 0.0
        
        # Set the objective value
        obj_type, variables, coefs, _, constant = self.lp.objective_function
        objective_value = constant 
        
        self.objective_value = objective_value
        
        return optimal_solution, objective_value

    def is_feasible(self) -> bool:
        """
        Check if the current solution is feasible.
        
        Returns:
        bool: True if the solution is feasible, raises InfeasibleError otherwise.
        
        Raises:
        InfeasibleError: If any constraint is violated by a significant amount 
                         (to account for potential floating-point errors).
        """
        epsilon = 1e-6  # Tolerance for floating-point comparisons

        for constraint in self.lp.constraints:
            variables, coefficients, _, comparison_op, rhs = constraint
            
            # Calculate the left-hand side of the constraint
            lhs = sum(coef * self.solution.get(var, 0) for var, coef in zip(variables, coefficients))
            
            # Check if the constraint is satisfied
            if comparison_op == ComparisonOp.EQ:
                if abs(lhs - rhs) > epsilon:
                    raise InfeasibleError(f"Equality constraint violated: {lhs} != {rhs}")
            elif comparison_op == ComparisonOp.LEQ or comparison_op == ComparisonOp.LESS:
                if lhs > rhs + epsilon:
                    raise InfeasibleError(f"Upper bound constraint violated: {lhs} > {rhs}")
            elif comparison_op == ComparisonOp.GEQ or comparison_op == ComparisonOp.GREATER:
                if lhs < rhs - epsilon:
                    raise InfeasibleError(f"Lower bound constraint violated: {lhs} < {rhs}")

        # Check non-negativity constraints for all variables
        for var, value in self.solution.items():
            if value < -epsilon:
                raise InfeasibleError(f"Non-negativity constraint violated for {var}: {value} < 0")

        return True