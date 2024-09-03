from pulp import *
from queue import PriorityQueue
from typing import List, Tuple
from anytree import NodeMixin, RenderTree
from anytree.exporter import DotExporter
import graphviz

class Node(NodeMixin): 
	"""
  Represents a node in the branch-and-bound tree.

  Inherits from anytree.NodeMixin to enable tree structure
    
  Attributes:
  lp_model (pulp.LpProblem): The LP relaxation associated with this node.
  lower_bound (float): The lower bound (objective value) for this node.
  depth (int): The depth of this node in the branch-and-bound tree.
  solution (dict): The solution values for the decision variables.
  parent (Node): The parent node in the branch-and-bound tree.
  branching_variable (Tuple[int, int]): The variable branched on to create this node.
  branching_value (int): The value assigned to the branching variable (0 or 1).
	"""
	def __init__(self, lp_model: pulp.LpProblem, depth: int = 0, parent: 'Node' = None, 
								branching_variable: Tuple[int, int] = None, branching_value: int = None):
		super().__init__() # Initialize NodeMixin
		self.lp_model = lp_model
		self.lower_bound = float('inf')
		self.depth = depth
		self.solution = {}
		self.parent = parent
		self.branching_variable = branching_variable
		self.branching_value = branching_value
		self.name = self._generate_name()

	def _generate_name(self):
		if self.parent is None:
			return "Root"
		return f"Node_{self.depth}_{self.branching_variable}_{self.branching_value}"

	def __lt__(self, other: 'Node') -> bool:
		"""
		Comparison method for priority queue ordering.
		Nodes with lower bounds are given higher priority.
		"""
		return self.lower_bound < other.lower_bound
	
class BranchAndBound: 
	"""
	Implements the Branch-and-Bound algorithm for solving the Traveling Salesman Problem.

	Attributes:
		adj_matrix (List[List[float]]): The adjacency matrix representing the TSP instance
		n (int): The number of cities in the TSP
		root (Node): The root node of the branch-and-bound tree.
		best_upper_bound (float): The best known upper bound (incumbent solution value)
		best_solution (dict): The best known feasible solution
		active_nodes (PriorityQueue): Priority queue to store active nodes
	"""

	def __init__(self, adj_matrix: List[List[float]]):
		"""
		Constructor for the BranchAndBound class

		Args:
			adj_matrix (List[List[float]]): The adjacency matrix representing the TSP instance
		"""
		self.adj_matrix = adj_matrix
		self.n = len(adj_matrix)
		self.root = None
		self.best_upper_bound = float('inf')
		self.best_solution = None
		self.active_nodes = PriorityQueue()

	def setup_root_node(self):
		"""
		Performs additional, more complex initialization steps. 
		Initialize the Branch-and-Bound algorithm tree:
		1. Create the root node with the LP relaxation
		2. Set the initial best upper bound to infinity
		3. Add the root node to the priority queue of active nodes
		"""
		# Create the root node with LP relaxation
		lp_relaxation = self._create_lp_relaxation()
		self.root = Node(lp_relaxation)

		# Solve the root node's LP relaxation
		status = self.root.lp_model.solve()
		if status == pulp.LpStatusOptimal:
			self.root.lower_bound = self.root.lp_model.objective.value()
			self.root.solution = {var.name: var.value() for var in self.root.lp_model.varibales()}

		# Initialize the best upper bound to infinity
		self.best_upper_bound = float('inf')

		# Add the root node to the priority queue
		self.active_nodes.put(self.root)

	def _create_lp_relaxation(self) -> pulp.LpProblem:
		"""
		Create the LP relaxation of the TSP instance.

		Returns:
			pulp.LpProblem: The LP relaxation of the TSP
		"""
		model = pulp.LpProblem("TSP_Relaxation", pulp.LpMinimize)

		# Create continuous variables
		x = pulp.LpVariable.dicts("x", ((i, j) for i in range(self.n) for j in range(self.n) if i != j), lowBound=0, upBound=1, cat='Continuous')
		
		# Objective function
		model += pulp.lpSum(self.adj_matrix[i][j] * x[(i,j)] for i in range(self.n) for j in range(self.n) if i != j)

		# Constraints
		for i in range(self.n):
			model += pulp.lpSum(x[(i, j)] for j in range(self.n) if i != j) == 1 # Leave each city once
			model += pulp.lpSum(x[(j, i)] for j in range(self.n) if i != j) == 1 # Enter each city once

		# Subtour elimination constraints (using a flow-based formulation)
		y = pulp.LpVariable.dicts("y", ((i, j) for i in range(1, self.n) for j in range(1, self.n) if i != j), lowBound=0, cat='Continuous')

		for i in range(1, self.n):
			model += pulp.lpSum(y[i, j] for j in range(1, self.n) if i != j) - pulp.lpSum(y[(j, i)] for j in range(1, self.n) if i != j) == 1
			for j in range(1, self.n):
				if i != j:
					model += y[(i, j)] <= (self.n - 1) * x[(i, j)]

		return model

	def solve(self, time_limit: float = 3600) -> Dict[str, float]:
        """
        Main loop of the branch-and-bound algorithm.

        Args:
            time_limit (float): Maximum running time in seconds. Default is 1 hour.

        Returns:
            Dict[str, float]: The best feasible solution found, or None if no feasible solution was found.
        """
        start_time = time.time()

        while not self.active_nodes.empty():
            # Check if time limit is exceeded
            if time.time() - start_time > time_limit:
                print("Time limit exceeded.")
                break

            # Get the next node to process
            current_node = self.active_nodes.get()

            # Check if the node can be pruned
            if current_node.lower_bound >= self.best_upper_bound:
                continue  # Prune the node

            # Check if the solution is integer feasible
            is_integer, integer_solution = self._check_integer_feasibility(current_node.solution)

            if is_integer:
                # Update the best solution if necessary
                objective_value = self._calculate_objective_value(integer_solution)
                if objective_value < self.best_upper_bound:
                    self.best_upper_bound = objective_value
                    self.best_solution = integer_solution
                continue  # No need to branch further

            # Branch on a fractional variable
            branching_variable = self._select_branching_variable(current_node.solution)
            if branching_variable is None:
                continue  # No suitable branching variable found

            # Create and add child nodes
            for branch_value in [0, 1]:
                child_node = self._create_child_node(current_node, branching_variable, branch_value)
                if child_node is not None:
                    self.active_nodes.put(child_node)

	# After the solving process, visualize the tree
	self.visualize_tree()
	
        return self.best_solution

    def _check_integer_feasibility(self, solution: Dict[str, float]) -> Tuple[bool, Optional[Dict[str, int]]]:
        """
        Check if the given solution is integer feasible.

        Args:
            solution (Dict[str, float]): The solution to check.

        Returns:
            Tuple[bool, Optional[Dict[str, int]]]: 
                - Boolean indicating if the solution is integer feasible.
                - The integer solution if feasible, None otherwise.
        """
        integer_solution = {}
        for var_name, value in solution.items():
            if var_name.startswith('x_'):
                if abs(value - round(value)) > 1e-6:
                    return False, None
                integer_solution[var_name] = round(value)
        return True, integer_solution

    def _calculate_objective_value(self, solution: Dict[str, int]) -> float:
        """
        Calculate the objective value for a given solution.

        Args:
            solution (Dict[str, int]): The solution to evaluate.

        Returns:
            float: The objective value of the solution.
        """
        objective_value = 0
        for var_name, value in solution.items():
            if var_name.startswith('x_'):
                i, j = map(int, var_name.split('_')[1:])
                objective_value += self.adj_matrix[i][j] * value
        return objective_value

    def _select_branching_variable(self, solution: Dict[str, float]) -> Optional[str]:
        """
        Select a variable to branch on based on the current solution.

        Args:
            solution (Dict[str, float]): The current solution.

        Returns:
            Optional[str]: The name of the variable to branch on, or None if no suitable variable is found.
        """
        for var_name, value in solution.items():
            if var_name.startswith('x_'):
                if 0.01 < value < 0.99:
                    return var_name
        return None

    def _create_child_node(self, parent: Node, branching_variable: str, branch_value: int) -> Optional[Node]:
        """
        Create a child node by adding a new constraint to the parent's LP model.

        Args:
            parent (Node): The parent node.
            branching_variable (str): The name of the variable to branch on.
            branch_value (int): The value to fix the branching variable to (0 or 1).

        Returns:
            Optional[Node]: The new child node, or None if the resulting LP is infeasible.
        """
        child_model = parent.lp_model.copy()
        child_model += pulp.LpConstraint(
            pulp.LpAffineExpression([(child_model.variables()[branching_variable], 1)]),
            pulp.LpConstraintEQ,
            f"{branching_variable}_{branch_value}",
            branch_value
        )

        status = child_model.solve()
        if status == pulp.LpStatusOptimal:
            child_node = Node(
                child_model, 
                parent.depth + 1, 
                parent, 
                branching_variable, 
                branch_value
            )
            child_node.lower_bound = child_model.objective.value()
            child_node.solution = {var.name: var.value() for var in child_model.variables()}
            return child_node
        return None

	def visualize_tree(self):
        	"""
        	Visualize the branch-and-bound tree using graphviz.
	 	Generates a PNG image of the search tree.
        	"""
        	def node_to_string(node):
			"""Generate the label string for a node in the tree"""
            		return f"{node.name}\nLB: {node.lower_bound:.2f}"

        	def edge_to_string(node):
			"""Generate the label string for an edge in the tree"""
            		if node.parent is None:
                		return ''
            		return f"{node.branching_variable} = {node.branching_value}"

        	dot_exporter = DotExporter(self.root,
                                   nodeattrfunc=lambda node: f'label="{node_to_string(node)}"',
                                   edgeattrfunc=lambda parent, child: f'label="{edge_to_string(child)}"')
        
        	dot_data = dot_exporter.to_dotfile("branch_and_bound_tree.dot")
        
        	# Use graphviz to render the tree
        	graph = graphviz.Source.from_file("branch_and_bound_tree.dot")
        	graph.render("branch_and_bound_tree", format="png", cleanup=True)
        	print("Branch-and-bound tree visualization saved as 'branch_and_bound_tree.png'")
