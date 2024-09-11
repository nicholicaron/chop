from pulp import *
from queue import PriorityQueue
from typing import List, Tuple
from anytree import NodeMixin, RenderTree
from anytree.exporter import DotExporter
import graphviz
import time

class Node(NodeMixin): 
    """
  Represents a node in the branch-and-bound tree.

  Inherits from anytree.NodeMixin to enable tree structure

  Attributes:
  lp_model (LpProblem): The LP relaxation associated with this node.
  lower_bound (float): The lower bound (objective value) for this node.
  depth (int): The depth of this node in the branch-and-bound tree.
  solution (dict): The solution values for the decision variables.
  parent (Node): The parent node in the branch-and-bound tree.
  branching_variable (Tuple[int, int]): The variable branched on to create this node.
  branching_value (int): The value assigned to the branching variable (0 or 1).
    """
    def __init__(self, lp_model: LpProblem, depth: int = 0, parent: 'Node' = None, branching_variable: Tuple[int, int] = None, branching_value: int = None):
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
        
    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, value):
        self._depth = value

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
        self.linear_program = None

    def setup_root_node(self):
        """
        Performs initialization steps for the Branch-and-Bound algorithm:
        1. Create the root node with the IP formulation
        2. Set the initial best upper bound to infinity
        3. Initialize other necessary attributes
        """
        # Create the IP formulation
        ip_model = self._create_lp_formulation_TSP()
        
        # Create the root node
        self.root = Node(
            lp_model=ip_model,
            depth=0,
            parent=None,
            branching_variable=None,
            branching_value=None
        )
        
        # Initialize root node attributes
        self.root.lower_bound = float('-inf')  # Will be updated when the node is processed
        self.root.solution = {}  # Will be populated when the node is processed
        
        # Initialize the best upper bound to infinity
        self.best_upper_bound = float('inf')
        
        # Clear any existing nodes in the priority queue
        self.active_nodes = PriorityQueue()
        
        print("Root node initialized. Ready to start branch-and-bound process.")

    def _create_lp_formulation_TSP(self) -> LpProblem:
        """
        Create the IP formulation of the TSP instance using DFJ constraints.

        This method formulates the TSP as an integer programming problem using the 
        Dantzig-Fulkerson-Johnson formulation with subtour elimination constraints.

        Returns:
            LpProblem: The IP Formulation of the TSP
        """
        model = LpProblem("TSP_Formulation", LpMinimize)

        # Create binary variables
        x = LpVariable.dicts("x", ((i, j) for i in range(self.n) for j in range(self.n) if i != j), cat='Binary')

        # Objective function
        model += lpSum(self.adj_matrix[i][j] * x[(i,j)] for i in range(self.n) for j in range(self.n) if i != j)

        # Constraints
        for i in range(self.n):
            model += lpSum(x[(i, j)] for j in range(self.n) if i != j) == 1  # Leave each city once
            model += lpSum(x[(j, i)] for j in range(self.n) if i != j) == 1  # Enter each city once

        # Subtour elimination constraints (DFJ constraints)
        for k in range(2, self.n):
            for subset in itertools.combinations(range(self.n), k):
                model += lpSum(x[(i, j)] for i in subset for j in subset if i != j) <= len(subset) - 1

        return model

    def _create_lp_relaxation(self, ilp: LpProblem) -> LpProblem:
        """
        Create the LP relaxation of a given integer program

        This function creates a copy of the original integer program and relaxes the
        integrality constraints on all variables.

        Returns:
            LpProblem: The LP relaxation of the integer program 
        """
        # Create a deep copy of the original problem
        lp_relaxation = ilp.copy()

        # Relax the integrality constraints
        for var in lp_relaxation.variables():
            if var.cat == LpInteger:
                var.cat = LpContinuous

        return lp_relaxation

    def solve(self, time_limit: float = 3600) -> Dict[str, float]:
        """
        Main loop of the branch-and-bound algorithm.

        Args:
            time_limit (float): Maximum running time in seconds. Default is 1 hour.

        Returns:
            Dict[str, float]: The best feasible solution found, or None if no feasible solution was found.
        """
        start_time = time.time()
        iteration = 0

        print(f"Starting Branch and Bound algorithm with time limit: {time_limit} seconds")
        print(f"Initial best upper bound: {self.best_upper_bound}")

        # Process the root node
        if not self._process_node(self.root):
            print("Root node is infeasible. Problem has no solution.")
            return None

        self.active_nodes.put(self.root)

        while not self.active_nodes.empty():
            iteration += 1
            print(f"\nIteration {iteration}:")

            # Check if time limit is exceeded
            if time.time() - start_time > time_limit:
                print("Time limit exceeded.")
                break

            # Get the next node to process
            current_node = self.active_nodes.get()
            print(f"Processing node: {current_node.name}")

            # Process the node (solve its LP relaxation)
            if not self._process_node(current_node):
                print("Node is infeasible. Skipping.")
                continue

            print(f"Current node lower bound: {current_node.lower_bound}")

            # Check if the node can be pruned
            if current_node.lower_bound >= self.best_upper_bound:
                print("Node pruned: Lower bound >= Best upper bound")
                continue  # Prune the node

            # Check if the solution is integer feasible
            is_integer, integer_solution = self._check_integer_feasibility(current_node.solution)
            print(f"Is solution integer feasible? {is_integer}")

            if is_integer:
                # Update the best solution if necessary
                objective_value = self._calculate_objective_value(integer_solution)
                print(f"Integer solution found. Objective value: {objective_value}")
                if objective_value < self.best_upper_bound:
                    self.best_upper_bound = objective_value
                    self.best_solution = integer_solution
                    print(f"New best solution found! Upper bound updated to: {self.best_upper_bound}")
                else:
                    print("Integer solution not better than current best.")
                continue  # No need to branch further

            # Branch on a fractional variable
            branching_variable = self._select_branching_variable(current_node.solution)
            if branching_variable is None:
                print("No suitable branching variable found. Skipping node.")
                continue  # No suitable branching variable found

            print(f"Branching on variable: {branching_variable}")
            current_value = current_node.solution[branching_variable]
            print(f"Current value of branching variable: {current_value}")

            # Create and add child nodes
            for is_upper_branch in [False, True]:
                child_node = self._create_child_node(current_node, branching_variable, current_value, is_upper_branch)
                print(f"Created {'upper' if is_upper_branch else 'lower'} child node: {child_node.name}")
                self.active_nodes.put(child_node)

        print("\nBranch and Bound algorithm completed.")
        print(f"Best upper bound: {self.best_upper_bound}")
        print(f"Best solution found: {self.best_solution}")

        # After the solving process, visualize the tree
        # self.visualize_tree()

        return self.best_solution


    def _process_node(self, node: Node) -> bool:
        """
        Process a node by solving its LP relaxation.

        Args:
            node (Node): The node to process.

        Returns:
            bool: True if the node is feasible, False otherwise.
        """
        status = node.lp_model.solve()
        if status == LpStatusOptimal:
            node.lower_bound = node.lp_model.objective.value()
            node.solution = {var.name: var.value() for var in node.lp_model.variables()}
            return True
        return False

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
        Calculate the objective value for a given TSP solution.

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
        Uses the "most fractional" branching strategy.

        Args:
            solution (Dict[str, float]): The current solution.

        Returns:
            Optional[str]: The name of the variable to branch on, or None if no suitable variable is found.
        """
        most_fractional_var = None
        max_fractionality = 0

        for var_name, value in solution.items():
            if var_name.startswith('x_'):
                fractionality = abs(value - round(value))
                if fractionality > 1e-6 and fractionality > max_fractionality:
                    most_fractional_var = var_name
                    max_fractionality = fractionality

        return most_fractional_var

    def _create_child_node(self, parent: Node, branching_variable: str, current_value: float, is_upper_branch: bool) -> Optional[Node]:
        """
        Create a child node by adding a new constraint to the parent's LP model.

        Args:
            parent (Node): The parent node.
            branching_variable (str): The name of the variable to branch on.
            current_value (float): The current value of the branching variable.
            is_upper_branch (bool): True if this is the upper branch (>=), False for lower branch (<=)

        Returns:
            Node: The new child node.
        """
        child_model = parent.lp_model.copy()
        var = child_model.variables()[branching_variable]

        if is_upper_branch:
            branch_value = math.ceil(current_value)
            child_model += LpConstraint(
                LpAffineExpression([(var, 1)]),
                LpConstraintGE,
                f"{branching_variable}_lower_bound",                                ### Why is this here? 
                branch_value
            )
        else:
            branch_value = math.floor(current_value)
            child_model += LpConstraint(
                LpAffineExpression([(var, 1)]), 
                LpConstraintLE, 
                f"{branching_variable}_upper_bound",                                ### Why is this here? 
                branch_value
            )

        child_node = Node(
            child_model, 
            parent.depth + 1, 
            parent, 
            branching_variable, 
            branch_value
        )
    
        return child_node

    def node_to_string(node):
        """Generate the label string for a node in the tree"""
        return f"{node.name}\nLB: {node.lower_bound:.2f}"

    def edge_to_string(node):
        """Generate the label string for an edge in the tree"""
        if node.parent is None:
            return ''
        return f"{node.branching_variable} = {node.branching_value}"

    def visualize_tree(self):
            """
            Visualize the branch-and-bound tree using graphviz.
        Generates a PNG image of the search tree.
            """
            dot_exporter = DotExporter(self.root,
                                       nodeattrfunc=lambda node: f'label="{node_to_string(node)}"',
                                       edgeattrfunc=lambda parent, child: f'label="{edge_to_string(child)}"')

            dot_data = dot_exporter.to_dotfile("branch_and_bound_tree.dot")

            # Use graphviz to render the tree
            graph = graphviz.Source.from_file("branch_and_bound_tree.dot")
            graph.render("branch_and_bound_tree", format="png", cleanup=True)
            print("Branch-and-bound tree visualization saved as 'branch_and_bound_tree.png'")

