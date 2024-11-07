import numpy as np
#from scipy.optimize import linprog
from simplex import linprog_simplex, SimplexResult, PivOptions
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils.convert import from_networkx
import os
import random
import time
import heapq
import argparse
from itertools import combinations
import sys
import math


"""
Integer Linear Program (ILP) Solver using Branch-and-Bound Method

This module implements a Branch-and-Bound algorithm for solving Integer Linear Programs,
with specific optimizations for the Traveling Salesman Problem (TSP). The solver uses
a best-first search strategy to explore the solution space efficiently.

Key Features:
- Pure integer and mixed integer linear program support
- Special cuts for TSP (subtour elimination)
- Best-first search branching strategy
- Branch-and-bound tree visualization
- Solution persistence for machine learning applications
- Progress logging and monitoring
- Support for multiple example problems

The solver maintains:
- A priority queue of nodes to explore
- Global bounds on the objective value
- A tree structure representing the search space
- Visualization capabilities for analysis

Dependencies:
    numpy: For numerical computations and array operations
    networkx: For tree data structures and visualization
    torch: For graph persistence and ML integration
    torch_geometric: For graph neural network compatibility
    matplotlib: For visualization of the branch-and-bound tree
    simplex: Custom implementation of the simplex algorithm

Example Usage:
    solver = ILPSolver()
    solution, value, nodes, optimal_node = solver.solve(
        c=objective_coefficients,
        A_ub=inequality_constraint_matrix,
        b_ub=inequality_constraint_vector,
        problem_name="example_problem"
    )

Notes:
    - The solver assumes maximization problems. For minimization,
      negate the objective coefficients.
    - Solution persistence creates PyTorch geometric data objects
      suitable for machine learning applications.
    - Visualization generates both static PNG files and interactive
      network visualizations.
"""

class Node:
    """
    Represents a node in the branch-and-bound enumeration tree.
    
    Each node maintains its own state within the branch-and-bound process,
    including local bounds, branching decisions, and solution information.
    """
    
    def __init__(self, parent=None, branch_var=None, branch_val=None, 
                 branch_direction=None):
        """
        Initialize a new node in the branch-and-bound tree.
        
        Args:
            parent (Node, optional): Parent node in B&B tree.
            branch_var (int, optional): Branching variable index. 
            branch_val (float, optional): Branching value. 
            branch_direction (str, optional): Branch direction. 
        """
        # Basic node information
        self.parent = parent  # Reference to parent node in tree
        self.branch_var = branch_var  # Variable index chosen for branching
        self.branch_val = branch_val  # Value at which branching occurs
        self.branch_direction = branch_direction  # Direction of branch (floor/ceil)
        
        # Solution information
        self.solution = None  # Current solution vector
        self.value = None  # Objective value at this node
        self.id = None  # Unique node identifier
        
        # Tree structure information
        self.depth = parent.depth + 1 if parent else 0  # Node depth in tree
        
        # Bounds and relaxation information
        self.local_upper_bound = -np.inf  # Upper bound from LP relaxation
        self.relaxed_soln = None  # Solution to LP relaxation
        
        # Solution characteristics
        self.num_int = 0  # Count of integer variables
        self.num_frac = 0  # Count of fractional variables
        self.indices_frac = []  # List of fractional variable indices
        self.optimality_gap = np.inf  # Gap to best known solution
        
        # Status tracking
        self.prune_reason = None  # Why node was pruned (if applicable)
        self.A_ub = None  # Node-specific constraint matrix
        self.b_ub = None  # Node-specific RHS vector
        self.processed = False  # Processing status flag

    def set_constraints(self, A_ub: np.ndarray, b_ub: np.ndarray, tableau: np.ndarray = None) -> None:
        """
        Store the node-specific constraints and tableau.
        
        Makes deep copies of the constraint matrix, RHS vector, and tableau to ensure
        node independence in the branch-and-bound tree.
        
        Args:
            A_ub (np.ndarray): Inequality constraint matrix
            b_ub (np.ndarray): Inequality RHS vector
            tableau (np.ndarray, optional): Current tableau if available
        """
        self.A_ub = A_ub.copy()
        self.b_ub = b_ub.copy()
        if tableau is not None:
            self.tableau = tableau.copy()

    def __lt__(self, other: 'Node') -> bool:
        """
        Compare nodes for priority queue ordering.
        
        Implements '>' for maximization (max-queue ordering -- highest values prioritized first).
        To calculate value, we will combine the node's LP relaxation value multiplicatively with 
        the depth of the node using exponential decay. (Note: we prioritize shallower nodes over deeper ones.)
        
        f(node) = lp_relaxation_value * e ^ (-B * depth)
            + Parameters:
                - B: Decay factor controlling the influence of depth (chosen arbitrarily)
        
        Args:
            other (Node): Node to compare against
            
        Returns:
            bool: True if this node's value > other node's value
        """
        return self.id < other.id

        # Option 0: Use simple maximization
        # return self.value > other.value  # Changed to '>' for maximization

        # Option 1: Use exponential decay of depth
        #beta = 0.05 # random.uniform(0.001, 0.075) # Decay factor randomly chosen between 0.001 and 0.075
        #return -(self.value * math.exp(-beta * self.depth)) < -(other.value * math.exp(-beta * other.depth))
    
        # Option 2: Lexicographic ordering of depth and value
        # return (self.depth, self.value) > (other.depth, other.value)

        # Option 3: Normalized weighted sum
        # Note: Requires extra overhead for tracking max values
        #alpha = 0.5  # Example weight
        #norm_lp_self = self.value / solver.max_lp if solver.max_lp > 0 else 0
        #norm_depth_self = self.depth / solver.max_depth if solver.max_depth > 0 else 0
              
        #norm_lp_other = other.value / solver.max_lp if solver.max_lp > 0 else 0
        #norm_depth_other = other.depth / solver.max_depth if solver.max_depth > 0 else 0
              
        #return (norm_lp_self + alpha * norm_depth_self) > (norm_lp_other + alpha * norm_depth_other)
        

class ILPSolver:
    """
    Branch-and-Bound solver for Integer Linear Programs.
    
    Implements a best-first search strategy for solving ILPs with visualization 
    capabilities and solution persistence for machine learning applications.
    """

    def __init__(self):
        self.optimal_obj_value = -np.inf # Best integer feasible solution value found
        self.optimal_solution = None # Best integer feasible solution vector found
        self.enumeration_tree = nx.DiGraph() # Branch-and-bound tree structure
        self.node_counter = 0 # Number of nodes created in the tree
        self.optimal_node = None # Node containing the optimal solution
        self.global_lower_bound = -np.inf # Greatest lower bound (integer feasible solution) found among all instances so far
        self.problem_counter = 0 # Number of problems solved by this instance
        self.root_relaxation_value = None # Objective value of root node relaxation
        self.n_cities = 0 # Number of cities (for TSP instances)
        self.cut_probability = random.random()  # Probability of using Gomory cuts vs branching

    def _print_priority_queue(self, priority_queue):
        print("Priority Queue Contents:")
        for i, (value, node) in enumerate(priority_queue):
            print(f"  {i+1}. Value: {value:.4f}, Node ID: {node.id}")
        print()

    def solve(self, c, A_ub, b_ub, A_eq=None, b_eq=None, problem_name="default_name", 
          visualize=False, callback=None, tsp_instance=None):
        """
        Solve an Integer Linear Program using branch-and-bound, with special handling for TSP.
        
        This method implements a best-first search strategy for solving ILPs, with specific
        optimizations for the Traveling Salesman Problem (TSP). It handles both general ILPs
        and TSP-specific constraints and cuts.
    
        Args:
            c (np.ndarray): Objective function coefficients
            A_ub (np.ndarray): Inequality constraint matrix
            b_ub (np.ndarray): Inequality constraint RHS
            A_eq (np.ndarray, optional): Equality constraint matrix
            b_eq (np.ndarray, optional): Equality constraint RHS
            problem_name (str, optional): Name for visualization/logging
            visualize (bool, optional): Whether to generate visualizations
            callback (callable, optional): Function to call when integer solutions are found
            tsp_instance (TSPInstance, optional): Instance of TSP problem for subtour elimination
        
        Returns:
            tuple: (optimal_solution, optimal_value, node_count, optimal_node)
                - optimal_solution (np.ndarray): Best integer solution found
                - optimal_value (float): Objective value of best solution
                - node_count (int): Number of nodes explored
                - optimal_node (Node): Node containing optimal solution
        """
        print(f"\nStarting to solve problem: {problem_name}")
        self._reset()
        self._set_global_attributes(c, A_ub, b_ub)

        self.callback = callback
        self.tsp_instance = tsp_instance

        # Handle TSP-specific initialization
        if tsp_instance is not None:
            self.n_cities = tsp_instance.n_cities
            print(f"TSP instance detected with {self.n_cities} cities")
        else:
            # For non-TSP problems, calculate n_cities for consistency
            self.n_cities = int((-1 + np.sqrt(1 + 8 * len(c))) / 2) if len(c) > 0 else 0

        self.processed_nodes = set()  # Track processed nodes to avoid reprocessing

        # Initialize root node
        root_node = Node()
        root_node.id = self._get_next_node_id()

        # Handle equality constraints for TSP and other problems
        if A_eq is not None and b_eq is not None:
            # Convert equality constraints to inequality pairs
            A_ub_from_eq = np.vstack([A_eq, -A_eq])
            b_ub_from_eq = np.concatenate([b_eq, -b_eq])
            
            # Combine with any existing inequality constraints
            if A_ub.size > 0:
                root_A_ub = np.vstack([A_ub_from_eq, A_ub])
                root_b_ub = np.concatenate([b_ub_from_eq, b_ub])
            else:
                root_A_ub = A_ub_from_eq
                root_b_ub = b_ub_from_eq
            
            root_node.set_constraints(root_A_ub, root_b_ub)
        else:
            root_node.set_constraints(A_ub, b_ub)

        # Solve LP relaxation for root node
        print("Solving LP relaxation for root node...")
        result = self._solve_lp_relaxation(c, root_node.A_ub, root_node.b_ub)
        if not result.success:
            print("Root node LP relaxation failed.")
            return None, None, self.node_counter, None

        # Store root node information
        root_node.relaxed_soln = result.x
        root_node.value = result.fun
        root_node.local_upper_bound = root_node.value
        root_node.tableau = result.tableau
        self.root_relaxation_value = root_node.value
        print(f"Root node relaxation value: {self.root_relaxation_value}")

        # Initialize the branch-and-bound tree
        self._add_node_to_tree(root_node, c, root_node.A_ub, root_node.b_ub)
        self._update_node_attributes(root_node, {'color': 'blue'})

        # Initialize priority queue with root node
        priority_queue = [(root_node.value, root_node)]
        print("Starting branch and bound process...")
        self._print_priority_queue(priority_queue)

        # Main branch-and-bound loop
        while priority_queue:
            # Get next node from priority queue
            _, current_node = heapq.heappop(priority_queue)
            print(f"Popped node {current_node.id} from priority queue")
            self._print_priority_queue(priority_queue)

            # Skip if node already processed
            if current_node.id in self.processed_nodes:
                print(f"Node {current_node.id} already processed. Skipping...")
                continue

            print(f"\nProcessing node {current_node.id}")
            print(f"Node upper bound: {current_node.value}")
            print(f"Current best integer solution: {self.global_lower_bound}")
            
            # Check if node can be pruned by bound
            # For maximization: if node's upper bound ≤ best known solution, prune
            if current_node.value <= self.global_lower_bound:
                print(f"Node {current_node.id} pruned: upper bound {current_node.value} ≤ best known solution {self.global_lower_bound}")
                current_node.prune_reason = 'suboptimal'
                self._update_node_attributes(current_node, {
                    'color': 'orange', 
                    'prune_reason': 'suboptimal'
                })
                continue

            self.processed_nodes.add(current_node.id)

            # Verify node has a solution
            if current_node.relaxed_soln is None:
                print("Warning: Node has no stored solution")
                continue

            # Early stopping check - if gap is within tolerance of 1e-4, we can stop
            #if (self.global_lower_bound > -np.inf and  # Ensure we have a feasible solution
            #    abs(current_node.local_upper_bound - self.global_lower_bound) <= 
            #    1e-4 * (1 + abs(self.global_lower_bound))):
            #    print(f"Early stopping: gap {abs(current_node.local_upper_bound - self.global_lower_bound):.2e} within tolerance {1e-4}")
            #    break

            # Check for integer solution
            is_integer_solution = all(abs(x - round(x)) < 1e-6 
                                    for x in current_node.relaxed_soln)
            print(f"Is integer solution: {is_integer_solution}")

            if is_integer_solution:
                # Handle TSP-specific subtour elimination
                if self.tsp_instance is not None:
                    subtours = self.tsp_instance.find_subtours(current_node.relaxed_soln)
                    if len(subtours) > 1:
                        print(f"Found {len(subtours)} subtours")
                        # Add constraints for each subtour
                        for subtour in subtours:
                            constraint, rhs = self.tsp_instance.generate_subtour_constraint(subtour)
                            new_A_ub = np.vstack([current_node.A_ub, constraint])
                            new_b_ub = np.append(current_node.b_ub, rhs)

                            # Create new node with subtour elimination constraint
                            new_node = Node(parent=current_node)
                            new_node.id = self._get_next_node_id()
                            new_node.set_constraints(new_A_ub, new_b_ub)

                            # Solve new LP relaxation
                            result = self._solve_lp_relaxation(c, new_A_ub, new_b_ub)
                            if result.success:
                                new_node.relaxed_soln = result.x
                                new_node.value = result.fun
                                new_node.local_upper_bound = new_node.value
                                new_node.tableau = result.tableau
                                self._add_node_to_tree(new_node, c, new_A_ub, new_b_ub)
                                heapq.heappush(priority_queue, (-(new_node.value) * math.exp(-0.05 * new_node.depth), new_node))
                                print(f"Added node with subtour elimination constraint")
                            else:
                                print(f"Warning: LP relaxation with new subtour constraint is infeasible")
                                new_node.prune_reason = 'infeasible'
                                self._add_node_to_tree(new_node, c, new_A_ub, new_b_ub)
                                self._update_node_attributes(new_node, {
                                    'color': 'red', 
                                    'prune_reason': 'infeasible'
                                })

                        # Call callback to visualize the subtours if provided
                        if self.callback is not None:
                            self.callback(current_node.relaxed_soln, False, problem_name)
                        
                        continue  # Continue to next iteration

                # Process integer solution (either non-TSP or TSP without subtours)
                if current_node.value > self.global_lower_bound:
                    print("New best integer solution found!")
                    # Reset color of previous optimal node
                    if self.optimal_node is not None:
                        self._update_node_attributes(self.optimal_node, {'color': 'lightblue'})

                    # Update global bounds and optimal solution
                    self.global_lower_bound = current_node.value
                    self.optimal_obj_value = current_node.value
                    self.optimal_solution = current_node.relaxed_soln
                    self.optimal_node = current_node
                    self._update_node_attributes(current_node, {
                        'color': 'green',
                        'relaxed_obj_value': current_node.value
                    })

                    # Call callback if provided
                    if self.callback is not None:
                        self.callback(current_node.relaxed_soln, True, problem_name)
                else:
                    print("Integer solution found but not better than current best.")
                    self._update_node_attributes(current_node, {'color': 'lightblue'})
                    # Call callback for non-optimal integer solutions
                    if self.callback is not None:
                        self.callback(current_node.relaxed_soln, False, problem_name)
            else:
                # Handle fractional solution through branching or cutting
                self.add_constraints(current_node, c, priority_queue)

        print("\nBranch and bound process completed.")
        print(f"Optimal objective value: {self.optimal_obj_value}")
        print(f"Number of nodes explored: {self.node_counter}")

        # Generate visualization if requested
        if visualize:
            self._visualize_tree(problem_name)

        # Save the final graph
        #self._save_graph_to_disk(problem_name)

        return (self.optimal_solution, self.optimal_obj_value, 
                self.node_counter, self.optimal_node)

    def _reset(self):
        """
        Reset the solver's state for a new problem.
        
        Reinitializes all internal tracking variables and data structures,
        preparing the solver for a fresh optimization problem. This includes
        clearing bounds, solutions, tree structure, and counters.
        """
        self.optimal_obj_value = -np.inf # Reset best integer solution value found
        self.optimal_solution = None # Reset best integer solution vector found
        self.enumeration_tree = nx.DiGraph() # Reset branch-and-bound tree structure
        self.node_counter = 0 # Reset node counter
        self.optimal_node = None # Reset optimal node
        self.global_lower_bound = -np.inf # Reset global lower bound
        self.global_upper_bound = np.inf # Reset global upper bound
        self.root_relaxation_value = None # Reset root node relaxation value

    def _add_node_to_tree(self, node, c, A_ub, b_ub):
        """
        Add a new node to the branch-and-bound tree with its attributes.
        
        Creates a node in the NetworkX graph structure with all relevant
        problem data and solution information. Also establishes parent-child
        relationships in the tree.
        
        Args:
            node (Node): Node to add to the tree
            c (np.ndarray): Objective coefficients
            A_ub (np.ndarray): Constraint matrix for this node
            b_ub (np.ndarray): RHS vector for this node
        """
        # Get default attributes then update with node-specific values
        attributes = self._get_default_node_attributes()
        attributes.update({
            'depth': node.depth,
            'branch_variable': node.branch_var,
            'branch_value': node.branch_val,
            'branch_direction': node.branch_direction,
            'local_upper_bound': node.local_upper_bound,
            'current_constraints': A_ub.tolist(), # Convert numpy arrays to lists for NetworkX
            'current_rhs': b_ub.tolist(), # Convert numpy arrays to lists for NetworkX
            'active_constraints': [],
            'slack_values': [],
            'optimality_gap': np.inf,
            'children_pruned': 0,
            'prune_reason': None,
        })

        # Add node to the tree with its attributes
        self.enumeration_tree.add_node(node.id, **attributes)

        # If node has a parent, add edge to represent relationship
        if node.parent:
            self.enumeration_tree.add_edge(node.parent.id, node.id)

    def _calculate_node_attributes(self, node, c, A_ub, b_ub, result):
        """
        Calculate and update various node metrics and attributes.
        
        Computes important node characteristics including:
        - Active constraints identification
        - Slack values in constraints
        - Optimality gap
        - Integer/fractional variable counts
        
        Args:
            node (Node): Node to update
            c (np.ndarray): Objective coefficients
            A_ub (np.ndarray): Constraint matrix
            b_ub (np.ndarray): RHS vector
            result (SimplexResult): Result from LP relaxation
        """
        # Find constraints that are binding (slack near zero)
        node.active_constraints = np.where(np.isclose(A_ub @ result.x, b_ub))[0].tolist()

        # Calculate slack variables (difference between LHS and RHS)
        node.slack_values = (b_ub - A_ub @ result.x).tolist()

        # Calculate gap between node value and global lower bound
        if self.global_lower_bound > -np.inf:
            node.optimality_gap = (node.value - self.global_lower_bound) / abs(self.global_lower_bound)

        # Count integer and fractional variables in solution
        node.num_int = sum(1 for x in result.x if abs(x - round(x)) < 1e-6)
        node.num_frac = len(c) - node.num_int
        # Store indices of fractional variables for branching decisions
        node.indices_frac = [i for i, x in enumerate(result.x) if abs(x - round(x)) > 1e-6]

        # Update node attributes in the enumeration tree
        self._update_node_attributes(node, {
            'relaxed_soln': node.relaxed_soln.tolist(),
            'relaxed_obj_value': node.relaxed_obj_value,
            'value': node.value,
            'active_constraints': node.active_constraints,
            'slack_values': node.slack_values,
            'optimality_gap': node.optimality_gap,
            'num_int': node.num_int,
            'num_frac': node.num_frac,
            'indices_frac': node.indices_frac,
            'children_pruned': node.children_pruned
        })


    def _get_default_node_attributes(self):
        """
        Get default attributes for a new node in the enumeration tree.
        
        Returns a dictionary of initial values for all possible node attributes,
        ensuring consistent attribute presence across all nodes.
        
        Returns:
            dict: Default node attributes with initial values
        """
        return {
            'depth': 0,  # Level in the B&B tree
            'branch_variable': None,  # Variable used for branching
            'branch_value': None,  # Value used for branching
            'branch_direction': None,  # Direction of branching decision
            'fractionality': None,  # Measure of non-integrality
            'local_lower_bound': -np.inf,  # Best integer solution in subtree
            'local_upper_bound': np.inf,  # Best possible value in subtree
            'global_lower_bound': -np.inf,  # Best known integer solution
            'global_upper_bound': np.inf,  # Best possible solution overall
            'current_constraints': None,  # Node's constraint matrix
            'current_rhs': None,  # Node's RHS vector
            'relaxed_soln': None,  # LP relaxation solution
            'relaxed_obj_value': None,  # LP relaxation objective value
            'num_int': 0,  # Count of integer variables
            'num_frac': 0,  # Count of fractional variables
            'indices_frac': [],  # Indices of fractional variables
            'active_constraints': [],  # Binding constraints
            'slack_values': [],  # Constraint slack values
            'optimality_gap': np.inf,  # Gap to best known solution
            'children_pruned': 0,  # Number of pruned child nodes
            'color': 'lightgray'  # Visual attribute for plotting
        }

    def _update_node_attributes(self, node, attributes):
        """
        Update attributes of a node in the enumeration tree.
        
        Safely updates node attributes while ensuring proper data type conversion
        for NetworkX storage (converting numpy arrays to lists).
        
        Args:
            node (Node): Node to update
            attributes (dict): New attributes to set or update
        """
        # Get current attributes from tree
        current_attributes = self.enumeration_tree.nodes[node.id]

        # Update only existing attributes, converting numpy arrays to lists
        for key, value in attributes.items():
            if key in current_attributes:
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                current_attributes[key] = value

    def _update_constraints(self, A_ub, b_ub, branch_var, branch_val, direction):
        """
        Update constraint matrices for a new branching decision.
        
        Creates new constraint matrices that enforce the branching decision,
        either upper or lower bounding a variable.
        
        Args:
            A_ub (np.ndarray): Current constraint matrix
            b_ub (np.ndarray): Current RHS vector
            branch_var (int): Variable to branch on
            branch_val (float): Value to branch at
            direction (str): 'floor' or 'ceil' for branching direction
            
        Returns:
            tuple: (new_A_ub, new_b_ub) Updated constraint matrices
        """
        # Create new constraint row for branching decision
        new_row = np.zeros(A_ub.shape[1])  
        new_row[branch_var] = 1 if direction == 'floor' else -1

        # Stack new row to constraint matrix and append new RHS value
        new_A_ub = np.vstack([A_ub, new_row])
        new_b_ub = np.append(b_ub, branch_val if direction == 'floor' else -branch_val)

        return new_A_ub, new_b_ub

    def _get_next_node_id(self):
        """
        Generate unique identifier for a new node.
        
        Increments the node counter and returns a string identifier
        that maintains ordering of node creation.
        
        Returns:
            str: Unique node identifier
        """
        self.node_counter += 1
        return f"Node {self.node_counter}"

    def _solve_lp_relaxation(self, c, A_ub, b_ub, A_eq=None, b_eq=None):
        """
        Solve the LP relaxation of the current node.
        
        Uses custom simplex implementation to solve the linear programming
        relaxation at the current node.
        
        Args:
            c (np.ndarray): Objective coefficients
            A_ub (np.ndarray): Inequality constraint matrix
            b_ub (np.ndarray): Inequality RHS vector
            A_eq (np.ndarray, optional): Equality constraint matrix
            b_eq (np.ndarray, optional): Equality RHS vector
            
        Returns:
            SimplexResult: Solution information from LP solver
        """
        # Ensure equality constraints are numpy arrays
        if A_eq is None:
            A_eq = np.empty((0, len(c)))
        if b_eq is None:
            b_eq = np.empty(0)

        # Create PivOptions with default values for simplex solver
        piv_options = PivOptions()

        # Solve the LP Relaxation
        result = linprog_simplex(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            max_iter=10**6,
            piv_options=piv_options
        )

        # Return formatted result
        return SimplexResult(
            x=result.x,
            lambd=result.lambd,
            fun=result.fun,
            success=result.success,
            status=result.status,
            num_iter=result.num_iter,
            tableau=result.tableau
        )

    def _set_global_attributes(self, c, A_ub, b_ub):
        """
        Set global problem attributes in the enumeration tree.
        
        Stores the original problem data in the graph object for
        reference and persistence.
        
        Args:
            c (np.ndarray): Original objective coefficients
            A_ub (np.ndarray): Original constraint matrix
            b_ub (np.ndarray): Original RHS vector
        """
        # Store original problem data as lists in graph attributes
        self.enumeration_tree.graph['og_obj_coefs'] = c.tolist() # Original objective coefficients
        self.enumeration_tree.graph['og_constraints'] = A_ub.tolist() # Original constraint matrix
        self.enumeration_tree.graph['og_rhs'] = b_ub.tolist() # Original right-hand side

    def _visualize_tree(self, problem_name):
        """
        Generates a comprehensive visualization of the enumeration tree showing:
        - Node relationships and branching decisions
        - Solution values and branching information
        - Color coding for different node types (root, pruned, optimal, etc.)
        - Legend explaining node colors
        
        Args:
            problem_name (str): Name of the problem for file naming
            
        The visualization is saved as a PNG file in the 'plots' directory.
        Node colors indicate:
        - Blue: Root node
        - Red: Pruned (infeasible)
        - Orange: Pruned (suboptimal)
        - Green: Optimal solution
        - Light blue: Integer feasible but not optimal
        - Light gray: Non-integral solution
        """
        # Create a new figure with a size that scales with the number of nodes
        n_nodes = len(self.enumeration_tree.nodes)
        fig_size = max(24, int(np.sqrt(n_nodes) * 3))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        # Calculate node positions using spring layout
        pos = nx.spring_layout(self.enumeration_tree, k=2, iterations=50)

        # Adjust x and y coordinates based on depth and number of nodes at each level
        levels = {}
        for node in self.enumeration_tree.nodes():
            depth = nx.shortest_path_length(self.enumeration_tree, source="Node 1", target=node)
            if depth not in levels:
                levels[depth] = []
            levels[depth].append(node)

        max_nodes_at_level = max(len(nodes) for nodes in levels.values())
        for depth, nodes in levels.items():
            y = 1 - depth * 0.1
            for i, node in enumerate(nodes):
                x = (i + 1) / (len(nodes) + 1)
                pos[node] = (x, y)

        # Prepare node colors
        colors = []
        for node in self.enumeration_tree.nodes:
            node_data = self.enumeration_tree.nodes[node]
            if node_data.get('color') == 'blue':  # Root node
                colors.append('blue')
            elif node_data['prune_reason'] == 'infeasible': # the node is infeasible
                colors.append('red')
            elif node_data['prune_reason'] == 'suboptimal': # i.e. the objective value is less than the global lower bound
                colors.append('orange')
            elif node_data.get('color') == 'green':  # Optimal node
                colors.append('green')
            elif node_data.get('color') == 'lightblue':  # Integer feasible but not optimal
                colors.append('lightblue')
            else:
                colors.append('lightgray')

        # Prepare node labels
        labels = {}
        for node in self.enumeration_tree.nodes:
            node_data = self.enumeration_tree.nodes[node]
            label = f"{node}\n"
            
            # Add objective value to label if available
            if 'relaxed_obj_value' in node_data:
                value = node_data['relaxed_obj_value']
                if value is not None:
                    label += f"Value: {value:.2f}\n"
                else:
                    label += "Value: N/A\n"
            else:
                label += "Value: N/A\n"

            # Add node type and constraint information
            if node_data.get('color') == 'blue':
                label += "Root Node"
            elif node_data['branch_direction'] is not None:
                if node_data['branch_direction'] == 'floor':
                    label += f"Floor Branch:\nX{node_data['branch_variable']} <= {node_data['branch_value']}"
                elif node_data['branch_direction'] == 'ceil':
                    label += f"Ceil Branch:\nX{node_data['branch_variable']} >= {node_data['branch_value']}"
                elif node_data['branch_direction'] == 'gomory':
                    label += f"Gomory Cut:\n{node_data['branch_value']}"

            labels[node] = label

        # Draw the graph with all components
        nx.draw(self.enumeration_tree, pos, with_labels=True, labels=labels,
                node_color=colors, node_size=4000, font_size=6, ax=ax,
                arrows=True, arrowsize=10)

        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1,fc="blue", edgecolor='none', label='Root'),
            plt.Rectangle((0,0),1,1,fc="red", edgecolor='none', label='Pruned (Infeasible)'),
            plt.Rectangle((0,0),1,1,fc="orange", edgecolor='none', label='Pruned (Suboptimal)'),
            plt.Rectangle((0,0),1,1,fc="green", edgecolor='none', label='Optimal Solution'),
            plt.Rectangle((0,0),1,1,fc="lightblue", edgecolor='none', label='Integer Feasible'),
            plt.Rectangle((0,0),1,1,fc="lightgray", edgecolor='none', label='Non-integral')
        ]
        ax.legend(handles=legend_elements, loc='best')

        plt.title(f"ILP Branch and Bound Enumeration Tree (Best-First Search) - {problem_name}")
        plt.tight_layout()
        
        # Create a directory for plots if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        # Save the plot as a PNG file with high DPI
        plot_filename = f"plots/{problem_name.replace(' ', '_').lower()}_tree.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved as {plot_filename}")

    def _save_graph_to_disk(self, problem_name):
        """
        Persist the branch-and-bound tree for machine learning applications.
        
        Converts the NetworkX graph to a PyTorch Geometric data object and saves
        it to disk. Ensures all data is in a format compatible with PyTorch.
        
        Args:
            problem_name (str): Name of the problem for file naming
            
        The graph is saved in the 'saved_graphs' directory with a timestamp
        to ensure unique filenames.
        """
        # Ensure all node attributes are lists or basic Python types
        for node, data in self.enumeration_tree.nodes(data=True):
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    data[key] = value.tolist()
                elif isinstance(value, np.integer):
                    data[key] = int(value)
                elif isinstance(value, np.floating):
                    data[key] = float(value)

        # Convert NetworkX graph to Pytorch Geometric format
        data = from_networkx(self.enumeration_tree)

        # Create a unique name for the graph
        timestamp = int(time.time() * 1000)
        graph_name = f"graph_{problem_name.replace(' ', '_').lower()}_{self.problem_counter}_{timestamp}"
        self.problem_counter += 1

        # Ensure the 'saved_graphs' directory exists
        os.makedirs('saved_graphs', exist_ok=True)

        # Save the graph
        torch.save(data, f'saved_graphs/{graph_name}.pt')
        print(f"Graph saved as {graph_name}.pt")

    def branch(self, current_node, c, priority_queue):
        """
        Create two new nodes by branching on a fractional variable.
        
        Args:
            current_node (Node): Current node to branch from
            c (np.ndarray): Objective coefficients
            priority_queue (list): Priority queue of nodes
            
        Returns:
            None (modifies priority queue in place)
        """
        print("Branching on a fractional variable...")
        fractional_vars = [i for i, x in enumerate(current_node.relaxed_soln) 
                         if abs(x - round(x)) > 1e-6]
        branch_var = random.choice(fractional_vars)
        branch_val = current_node.relaxed_soln[branch_var]
        print(f"Branching on variable X{branch_var} which has value {branch_val}")

        for direction in ['floor', 'ceil']:
            new_A_ub = current_node.A_ub.copy()
            new_b_ub = current_node.b_ub.copy()
            
            if direction == 'floor':
                new_val = np.floor(branch_val)
                print(f"Creating {direction} branch (adding constraint X{branch_var} <= {new_val})...")
                new_constraint = np.zeros(len(c))
                new_constraint[branch_var] = 1
                new_A_ub = np.vstack([new_A_ub, new_constraint])
                new_b_ub = np.append(new_b_ub, new_val)
            else:
                new_val = -np.ceil(branch_val)
                print(f"Creating {direction} branch (adding constraint -X{branch_var} <= -{new_val})...")
                new_constraint = np.zeros(len(c))
                new_constraint[branch_var] = -1
                new_A_ub = np.vstack([new_A_ub, new_constraint])
                new_b_ub = np.append(new_b_ub, -np.ceil(branch_val))
                new_val = np.ceil(branch_val)

            self._add_node_with_new_constraints(current_node, c, new_A_ub, new_b_ub, 
                                              branch_var, new_val, direction, priority_queue)

    def gomory_cut(self, current_node, c, priority_queue):
        """
        Add Gomory cuts to create a new node.
        """
        print("Generating Gomory cut...")

        # Check if we have a valid tableau
        if not hasattr(current_node, 'tableau') or current_node.tableau is None:
            print("No tableau available for Gomory cut generation")
            return

        # Use a small tolerance for numerical stability
        FRAC_TOL = 1e-6

        tableau = current_node.tableau
        n_vars = len(c)

        # Get basic variables from the tableau
        basis_indices = []
        for i in range(tableau.shape[0]):
            # Find which variable is basic in this row
            basic_col = -1
            for j in range(tableau.shape[1] - 1):  # Exclude RHS
                if abs(tableau[i, j] - 1.0) < FRAC_TOL and \
               all(abs(tableau[k, j]) < FRAC_TOL for k in range(tableau.shape[0]) if k != i):
                    basic_col = j
                    break
            if basic_col >= 0 and basic_col < n_vars:  # Only consider original variables
                basis_indices.append((i, basic_col))

        # Find rows with fractional RHS that correspond to basic variables
        fractional_rows = []
        for row_idx, basic_var in basis_indices:
            rhs_val = tableau[row_idx, -1]
            frac_part = rhs_val - np.floor(rhs_val)
            if FRAC_TOL < frac_part < 1.0 - FRAC_TOL:
                fractional_rows.append((row_idx, frac_part))

        if not fractional_rows:
            print("No suitable rows found for Gomory cut generation")
            self.branch(current_node, c, priority_queue)
            return

        # Choose the row with the most fractional RHS
        cut_row_idx = max(fractional_rows, key=lambda x: min(x[1], 1.0 - x[1]))[0]

        # Generate the Gomory cut
        cut_coeffs = tableau[cut_row_idx, :n_vars].copy()  # Only use original variables
        cut_rhs = tableau[cut_row_idx, -1]

        # Apply the fractional part operation with numerical stability
        def frac_part(x):
            return x - np.floor(x) if x >= 0 else x - np.ceil(x)

        cut_coeffs = np.array([frac_part(coeff) for coeff in cut_coeffs])
        cut_rhs = frac_part(cut_rhs)

        # Validate that the cut is violated by the current solution
        violation = abs(np.dot(cut_coeffs, current_node.relaxed_soln) - cut_rhs)
        if violation < FRAC_TOL:
            print("Generated cut is not violated by current solution")
            return

        print(f"Generated Gomory cut from row {cut_row_idx}")
        print(f"Cut coefficients: {cut_coeffs}")
        print(f"Cut RHS: {cut_rhs}")
        print(f"Cut violation: {violation}")

        # Add the cut to the current constraints
        new_A_ub = np.vstack([current_node.A_ub, cut_coeffs])
        new_b_ub = np.append(current_node.b_ub, cut_rhs)

        # Update the tableau with the new cut
        # Add a new row to the tableau for the cut
        n_slack = current_node.tableau.shape[1] - len(c) - 1  # number of slack variables
        new_tableau_row = np.zeros(current_node.tableau.shape[1])
        new_tableau_row[:len(c)] = cut_coeffs
        new_tableau_row[-1] = cut_rhs
        
        # Add a new slack variable for the new constraint
        slack_col = np.zeros((current_node.tableau.shape[0] + 1, 1))
        slack_col[-1] = 1  # The new slack variable is basic in the new row
        
        # Construct the new tableau
        new_tableau = np.vstack([
            np.hstack([current_node.tableau, np.zeros((current_node.tableau.shape[0], 1))]),
            np.hstack([new_tableau_row, slack_col[-1]])
        ])

        # Store the cut details as a string for visualization
        cut_description = f"Cut: {' + '.join([f'{coeff:.2f}x{i}' for i, coeff in enumerate(cut_coeffs) if abs(coeff) > 1e-6])} ≤ {cut_rhs:.2f}"

        # Create new node with the Gomory cut and updated tableau
        self._add_node_with_new_constraints(
            current_node, c, new_A_ub, new_b_ub, 
            None, cut_description, "gomory", priority_queue,
            new_tableau
        )

    def add_constraints(self, current_node, c, priority_queue):
        """
        Randomly choose between branching and adding Gomory cuts.
        
        Args:
            current_node (Node): Current node to process
            c (np.ndarray): Objective coefficients
            priority_queue (list): Priority queue of nodes
            
        Returns:
            None (modifies priority queue in place)
        """
        if random.random() < self.cut_probability:
            self.gomory_cut(current_node, c, priority_queue)
        else:
            self.branch(current_node, c, priority_queue)

    def _add_node_with_new_constraints(self, parent_node, c, new_A_ub, new_b_ub, 
                                     branch_var, branch_val, direction, priority_queue,
                                     tableau=None):
        """
        Create and add a new node with the given constraints.
        
        Args:
            parent_node (Node): Parent node
            c (np.ndarray): Objective coefficients
            new_A_ub (np.ndarray): New constraint matrix
            new_b_ub (np.ndarray): New RHS vector
            branch_var (int): Variable used for branching (None for cuts)
            branch_val (float): Value used for branching (None for cuts)
            direction (str): Branch direction or cut type
            priority_queue (list): Priority queue of nodes
            tableau (np.ndarray, optional): Updated tableau if available
        """
        new_node = Node(parent=parent_node, branch_var=branch_var, 
                       branch_val=branch_val, branch_direction=direction)
        new_node.id = self._get_next_node_id()
        new_node.set_constraints(new_A_ub, new_b_ub, tableau)

        result = self._solve_lp_relaxation(c, new_A_ub, new_b_ub)
        if result.success:
            new_node.relaxed_soln = result.x
            new_node.value = result.fun
            new_node.local_upper_bound = new_node.value
            new_node.tableau = result.tableau if tableau is None else tableau
            print(f"New {direction} node value: {new_node.value}")
            self._add_node_to_tree(new_node, c, new_A_ub, new_b_ub)
            self._update_node_attributes(new_node, {
                'relaxed_obj_value': new_node.value
            })
            if new_node.local_upper_bound > self.global_lower_bound:
                heapq.heappush(priority_queue, (-(new_node.value * math.exp(-0.05 * new_node.depth)), new_node))
                print(f"Added {direction} node to priority queue")
                self._print_priority_queue(priority_queue)
            else:
                print(f"{direction} node pruned: suboptimal")
                new_node.prune_reason = 'suboptimal'
                self._update_node_attributes(new_node, {'color': 'orange', 
                                                      'prune_reason': 'suboptimal'})
        else:
            print(f"{direction} node is infeasible")
            new_node.prune_reason = 'infeasible'
            self._add_node_to_tree(new_node, c, new_A_ub, new_b_ub)
            self._update_node_attributes(new_node, {'color': 'red', 
                                                   'prune_reason': 'infeasible'})



def solve_and_print_results(solver, c, A_ub, b_ub, A_eq=None, b_eq=None, 
                          problem_name="unnamed", visualize=False, 
                          callback=None, tsp_instance=None):
    """
    Helper function to solve ILP and display results.
    
    Args:
        solver (ILPSolver): Instance of ILP solver
        c (np.ndarray): Objective coefficients
        A_ub (np.ndarray): Inequality constraint matrix
        b_ub (np.ndarray): Inequality RHS vector
        A_eq (np.ndarray, optional): Equality constraint matrix
        b_eq (np.ndarray, optional): Equality RHS vector
        problem_name (str): Name for the problem
        visualize (bool): Whether to generate visualization
        callback (callable): Optional callback function for solution visualization
        tsp_instance (TSPInstance): Optional TSP instance for subtour elimination
    """
    result = solver.solve(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        problem_name=problem_name,
        visualize=visualize,
        callback=callback,
        tsp_instance=tsp_instance
    )
    
    # Unpack the result
    solution = result[0]  # Optimal solution vector
    value = result[1]    # Optimal objective value
    num_nodes_explored = result[2] if len(result) > 2 else None
    optimal_node = result[3] if len(result) > 3 else None
    
    print(f"\nResults for {problem_name}:")
    print(f"Optimal solution: {solution}")
    print(f"Optimal objective value: {value}")
    
    if num_nodes_explored is not None:
        print(f"Number of nodes explored: {num_nodes_explored}")
    
    if optimal_node is not None:
        print(f"Optimal node ID: {optimal_node.id}")
    
    print("\n" + "="*50 + "\n")

class OutputRedirector:
    """
    Utility class for redirecting output to both console and log file.
    
    Enables simultaneous writing of output to both the terminal and a log file,
    useful for debugging and analysis of the branch-and-bound process.
    
    Attributes:
        terminal: Original stdout stream
        log: File stream for logging
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main(visualize):
    # Create a directory for logs if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Set up output redirection
    sys.stdout = OutputRedirector('logs/branch_and_bound_output.txt')

    solver = ILPSolver()

    # Example 1: 2 variables, 2 constraints
    c1 = np.array([1, 1])
    A_ub1 = np.array([[-1, 1], [8, 2]])
    b_ub1 = np.array([2, 19])
    solve_and_print_results(
        solver=solver,
        c=c1,
        A_ub=A_ub1,
        b_ub=b_ub1,
        problem_name="Example 1 (2 var, 2 cons)",
        visualize=visualize
    )

    # Example 2: 5 variables, 3 constraints
    c2 = np.array([3, 2, 5, 4, 1])
    A_ub2 = np.array([
        [2, 1, 3, 2, 1],
        [1, 2, 1, 1, 3],
        [1, 1, 2, 3, 1]
    ])
    b_ub2 = np.array([10, 8, 15])
    solve_and_print_results(
        solver=solver,
        c=c2,
        A_ub=A_ub2,
        b_ub=b_ub2,
        problem_name="Example 2 (5 var, 3 cons)",
        visualize=visualize
    )

    # Example 3: 8 variables, 5 constraints
    c3 = np.array([5, 7, 3, 2, 6, 4, 8, 1])
    A_ub3 = np.array([
        [3, 2, 1, 4, 2, 5, 1, 3],
        [1, 3, 2, 1, 4, 3, 2, 1],
        [2, 1, 4, 3, 1, 2, 3, 2],
        [4, 3, 2, 1, 3, 1, 2, 4],
        [1, 2, 3, 4, 2, 1, 3, 2]
    ])
    b_ub3 = np.array([20, 25, 30, 22, 18])
    solve_and_print_results(
        solver=solver,
        c=c3,
        A_ub=A_ub3,
        b_ub=b_ub3,
        problem_name="Example 3 (8 var, 5 cons)",
        visualize=visualize
    )

    # Example 4: 10 variables, 7 constraints
    c4 = np.array([4, 6, 2, 3, 7, 5, 8, 1, 9, 3])
    A_ub4 = np.array([
        [2, 3, 1, 4, 2, 5, 1, 3, 2, 1],
        [1, 2, 3, 1, 4, 2, 3, 1, 2, 4],
        [3, 1, 2, 4, 1, 3, 2, 5, 1, 2],
        [4, 2, 3, 1, 5, 2, 1, 3, 4, 2],
        [1, 3, 2, 5, 1, 4, 3, 2, 1, 3],
        [2, 4, 1, 3, 2, 1, 5, 4, 3, 1],
        [3, 2, 4, 1, 3, 5, 2, 1, 4, 2]
    ])
    b_ub4 = np.array([30, 25, 35, 40, 20, 28, 32])
    solve_and_print_results(
        solver=solver,
        c=c4,
        A_ub=A_ub4,
        b_ub=b_ub4,
        problem_name="Example 4 (10 var, 7 cons)",
        visualize=visualize
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ILP Solver with Branch and Bound")
    parser.add_argument("--visualize", action="store_true", help="Generate and save plots")
    args = parser.parse_args()

    print("Starting branch and bound solver...")
    main(args.visualize)

    # Close the log file
    sys.stdout.log.close()
    # Restore the original stdout
    sys.stdout = sys.stdout.terminal