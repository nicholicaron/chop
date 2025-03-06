"""
Core Branch-and-Bound solver implementation.

This module provides the main solver class that integrates all components
(priority queue, branching strategies, cutting planes, etc.).
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import time
import math
from typing import Dict, List, Tuple, Optional, Callable, Any, Set

from src.core.node import Node
from src.core.priority_queue import PriorityQueue
from src.strategies.priority_queue import BestBoundPrioritizer
from src.strategies.branching import MostFractionalBranching
from src.utils.logging import BnBLogger, Timer


class BranchAndBoundSolver:
    """
    Branch-and-Bound solver for Integer Linear Programs.
    
    Implements a best-first search strategy for solving ILPs with visualization 
    capabilities and solution persistence for machine learning applications.
    """
    
    def __init__(self, 
                 prioritizer=None, 
                 branching_strategy=None,
                 logger=None,
                 max_nodes=1000000,
                 tolerance=1e-6,
                 early_stop_gap=1e-4,
                 use_cuts=True,
                 cut_probability=0.3):
        """
        Initialize the solver with configurable components.
        
        Args:
            prioritizer: Strategy for prioritizing nodes (default: BestBoundPrioritizer)
            branching_strategy: Strategy for selecting branching variables (default: MostFractionalBranching)
            logger: Logger instance for monitoring and statistics
            max_nodes: Maximum number of nodes to explore
            tolerance: Numerical tolerance for integer feasibility
            early_stop_gap: Gap tolerance for early stopping (0.0 for exact solution)
            use_cuts: Whether to use cutting planes
            cut_probability: Probability of using cuts vs. branching
        """
        # Core components
        self.prioritizer = prioritizer or BestBoundPrioritizer()
        self.branching_strategy = branching_strategy or MostFractionalBranching()
        self.logger = logger or BnBLogger()
        
        # Parameters
        self.max_nodes = max_nodes
        self.tolerance = tolerance
        self.early_stop_gap = early_stop_gap
        self.use_cuts = use_cuts
        self.cut_probability = cut_probability
        
        # State variables
        self.reset()
        
    def reset(self):
        """
        Reset the solver's state for a new problem.
        
        Reinitializes all internal tracking variables and data structures,
        preparing the solver for a fresh optimization problem.
        """
        # Bounds and solutions
        self.global_lower_bound = -np.inf  # Best integer feasible solution value
        self.global_upper_bound = np.inf   # Best possible solution value
        self.optimal_solution = None       # Best solution vector found
        self.optimal_obj_value = -np.inf   # Best objective value found
        self.optimal_node = None           # Node with optimal solution
        
        # Problem data
        self.root_relaxation_value = None
        self.problem_counter = 0
        
        # Tree management
        self.node_counter = 0
        self.enumeration_tree = nx.DiGraph()
        self.priority_queue = PriorityQueue(self.prioritizer)
        self.processed_nodes = set()
        
        # Problem-specific data
        self.tsp_instance = None
        self.callback = None
        
    def _get_next_node_id(self) -> str:
        """
        Generate unique identifier for a new node.
        
        Returns:
            str: Unique node identifier
        """
        self.node_counter += 1
        self.logger.increment_stat("nodes.created")
        return f"Node {self.node_counter}"
    
    def solve(self, c, A_ub, b_ub, A_eq=None, b_eq=None, problem_name="default_name", 
             visualize=False, callback=None, tsp_instance=None):
        """
        Solve an Integer Linear Program using branch-and-bound.
        
        This method implements a best-first search strategy for solving ILPs, with specific
        optimizations for the Traveling Salesman Problem (TSP) if provided.
    
        Args:
            c (np.ndarray): Objective function coefficients
            A_ub (np.ndarray): Inequality constraint matrix
            b_ub (np.ndarray): Inequality constraint RHS
            A_eq (np.ndarray, optional): Equality constraint matrix
            b_eq (np.ndarray, optional): Equality constraint RHS
            problem_name (str, optional): Name for visualization/logging
            visualize (bool, optional): Whether to generate visualizations
            callback (callable, optional): Function to call when integer solutions are found
            tsp_instance (object, optional): Instance of TSP problem for subtour elimination
        
        Returns:
            tuple: (optimal_solution, optimal_value, node_count, optimal_node)
                - optimal_solution (np.ndarray): Best integer solution found
                - optimal_value (float): Objective value of best solution
                - node_count (int): Number of nodes explored
                - optimal_node (Node): Node containing optimal solution
        """
        self.logger.info(f"Starting to solve problem: {problem_name}")
        
        # Reset solver state
        self.reset()
        
        # Store callback and TSP instance
        self.callback = callback
        self.tsp_instance = tsp_instance
        
        # Set global problem attributes
        self._set_global_attributes(c, A_ub, b_ub)
        
        # Handle TSP-specific initialization
        if tsp_instance is not None:
            n_cities = tsp_instance.n_cities
            self.logger.info(f"TSP instance detected with {n_cities} cities")
        
        # Initialize root node
        root_node = self._initialize_root_node(c, A_ub, b_ub, A_eq, b_eq)
        if root_node is None:
            self.logger.error("Root node LP relaxation failed.")
            return None, None, self.node_counter, None
        
        # Add root node to priority queue
        self.priority_queue.push(root_node)
        
        # Main branch-and-bound loop
        with Timer("bnb_loop") as timer:
            result = self._branch_and_bound_loop(c, visualize, problem_name)
        
        self.logger.info(f"Branch and bound loop completed in {timer.elapsed:.3f} seconds")
        self.logger.info(f"Optimal objective value: {self.optimal_obj_value}")
        self.logger.info(f"Number of nodes explored: {self.node_counter}")
        
        # Generate visualization if requested
        if visualize:
            self._visualize_tree(problem_name)
        
        # Log queue statistics
        queue_stats = self.priority_queue.get_statistics()
        for key, value in queue_stats.items():
            self.logger.update_stats(f"queue.{key}", value)
        
        # Finalize and return results
        self.logger.finish()
        return result
    
    def _initialize_root_node(self, c, A_ub, b_ub, A_eq=None, b_eq=None):
        """
        Initialize the root node of the branch-and-bound tree.
        
        Args:
            c: Objective coefficients
            A_ub: Inequality constraint matrix
            b_ub: Inequality RHS vector
            A_eq: Equality constraint matrix (optional)
            b_eq: Equality RHS vector (optional)
            
        Returns:
            Node: Initialized root node or None if infeasible
        """
        self.logger.info("Initializing root node")
        
        # Create root node
        root_node = Node()
        root_node.id = self._get_next_node_id()
        
        # Handle equality constraints
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
        self.logger.info("Solving LP relaxation for root node")
        with self.logger.timers["lp_solving"]:
            result = self._solve_lp_relaxation(c, root_node.A_ub, root_node.b_ub)
        
        self.logger.increment_stat("lp_relaxations")
        
        if not result.success:
            self.logger.error("Root node LP relaxation infeasible")
            return None
        
        # Store root node information
        root_node.relaxed_soln = result.x
        root_node.value = result.fun
        root_node.local_upper_bound = root_node.value
        root_node.tableau = result.tableau
        self.root_relaxation_value = root_node.value
        
        # Initialize the branch-and-bound tree
        self._add_node_to_tree(root_node, c, root_node.A_ub, root_node.b_ub)
        self._update_node_attributes(root_node, {'color': 'blue'})
        
        self.logger.info(f"Root node relaxation value: {self.root_relaxation_value}")
        
        return root_node
    
    def _branch_and_bound_loop(self, c, visualize, problem_name):
        """
        Main branch-and-bound loop.
        
        Args:
            c: Objective coefficients
            visualize: Whether to visualize results
            problem_name: Name of the problem
            
        Returns:
            tuple: (optimal_solution, optimal_value, node_count, optimal_node)
        """
        self.logger.info("Starting branch and bound loop")
        
        nodes_processed = 0
        
        # Main loop
        while self.priority_queue and nodes_processed < self.max_nodes:
            # Get next node from priority queue
            with self.logger.timers["node_processing"]:
                current_node = self.priority_queue.pop()
                
                # Skip if node already processed
                if current_node.id in self.processed_nodes:
                    self.logger.debug(f"Node {current_node.id} already processed. Skipping...")
                    continue
                
                self.logger.debug(f"Processing node {current_node.id}")
                self.logger.debug(f"Node upper bound: {current_node.value}")
                self.logger.debug(f"Current best integer solution: {self.global_lower_bound}")
                
                # Check if node can be pruned by bound
                if current_node.value <= self.global_lower_bound:
                    self.logger.debug(f"Node {current_node.id} pruned: upper bound {current_node.value} ≤ best known solution {self.global_lower_bound}")
                    current_node.prune_reason = 'suboptimal'
                    self._update_node_attributes(current_node, {
                        'color': 'orange', 
                        'prune_reason': 'suboptimal'
                    })
                    self.logger.increment_stat("nodes.pruned_bound")
                    continue
                
                self.processed_nodes.add(current_node.id)
                self.logger.increment_stat("nodes.processed")
                nodes_processed += 1
                
                # Verify node has a solution
                if current_node.relaxed_soln is None:
                    self.logger.warning("Node has no stored solution")
                    continue
                
                # Early stopping check - if gap is within tolerance, we can stop
                if (self.global_lower_bound > -np.inf and  # Ensure we have a feasible solution
                    self.early_stop_gap > 0 and 
                    abs(current_node.value - self.global_lower_bound) <= 
                    self.early_stop_gap * (1 + abs(self.global_lower_bound))):
                    
                    self.logger.info(f"Early stopping: gap {abs(current_node.value - self.global_lower_bound):.2e} within tolerance {self.early_stop_gap}")
                    break
                
                # Check for integer solution
                is_integer_solution = all(abs(x - round(x)) < self.tolerance 
                                      for x in current_node.relaxed_soln)
                
                self.logger.debug(f"Is integer solution: {is_integer_solution}")
                
                if is_integer_solution:
                    self._process_integer_solution(current_node, c, problem_name)
                else:
                    self._process_fractional_solution(current_node, c)
            
        self.logger.info(f"Branch and bound process completed. Processed {nodes_processed} nodes.")
        
        return (self.optimal_solution, self.optimal_obj_value, 
                self.node_counter, self.optimal_node)
    
    def _process_integer_solution(self, node, c, problem_name):
        """
        Process a node with an integer solution.
        
        Handles TSP-specific subtour elimination and updates bounds.
        
        Args:
            node: Node with integer solution
            c: Objective coefficients
            problem_name: Name of the problem
        """
        # Handle TSP-specific subtour elimination
        if self.tsp_instance is not None:
            subtours = self.tsp_instance.find_subtours(node.relaxed_soln)
            if len(subtours) > 1:
                self.logger.info(f"Found {len(subtours)} subtours")
                
                # Add constraints for each subtour
                for subtour in subtours:
                    constraint, rhs = self.tsp_instance.generate_subtour_constraint(subtour)
                    new_A_ub = np.vstack([node.A_ub, constraint])
                    new_b_ub = np.append(node.b_ub, rhs)
                    
                    # Create new node with subtour elimination constraint
                    new_node = Node(parent=node)
                    new_node.id = self._get_next_node_id()
                    new_node.set_constraints(new_A_ub, new_b_ub)
                    
                    # Solve new LP relaxation
                    result = self._solve_lp_relaxation(c, new_A_ub, new_b_ub)
                    self.logger.increment_stat("lp_relaxations")
                    
                    if result.success:
                        new_node.relaxed_soln = result.x
                        new_node.value = result.fun
                        new_node.local_upper_bound = new_node.value
                        new_node.tableau = result.tableau
                        self._add_node_to_tree(new_node, c, new_A_ub, new_b_ub)
                        self.priority_queue.push(new_node)
                        self.logger.debug(f"Added node with subtour elimination constraint")
                    else:
                        self.logger.warning(f"LP relaxation with new subtour constraint is infeasible")
                        new_node.prune_reason = 'infeasible'
                        self._add_node_to_tree(new_node, c, new_A_ub, new_b_ub)
                        self._update_node_attributes(new_node, {
                            'color': 'red', 
                            'prune_reason': 'infeasible'
                        })
                        self.logger.increment_stat("nodes.pruned_infeasible")
                
                # Call callback to visualize the subtours if provided
                if self.callback is not None:
                    self.callback(node.relaxed_soln, False, problem_name)
                
                return  # Continue to next iteration
        
        # Process integer solution (either non-TSP or TSP without subtours)
        self.logger.increment_stat("nodes.integer_feasible")
        
        if node.value > self.global_lower_bound:
            self.logger.info("New best integer solution found!")
            # Reset color of previous optimal node
            if self.optimal_node is not None:
                self._update_node_attributes(self.optimal_node, {'color': 'lightblue'})
                
            # Update global bounds and optimal solution
            self.global_lower_bound = node.value
            self.optimal_obj_value = node.value
            self.optimal_solution = node.relaxed_soln
            self.optimal_node = node
            
            self._update_node_attributes(node, {
                'color': 'green',
                'relaxed_obj_value': node.value
            })
            
            self.logger.update_stats("best_objective", node.value)
            self.logger.increment_stat("nodes.optimal")
            
            # Call callback if provided
            if self.callback is not None:
                self.callback(node.relaxed_soln, True, problem_name)
        else:
            self.logger.debug("Integer solution found but not better than current best.")
            self._update_node_attributes(node, {'color': 'lightblue'})
            
            # Call callback for non-optimal integer solutions
            if self.callback is not None:
                self.callback(node.relaxed_soln, False, problem_name)
    
    def _process_fractional_solution(self, node, c):
        """
        Process a node with a fractional solution.
        
        Performs branching or cutting.
        
        Args:
            node: Node with fractional solution
            c: Objective coefficients
        """
        # Identify fractional variables
        node.indices_frac = [i for i, x in enumerate(node.relaxed_soln) 
                           if abs(x - round(x)) > self.tolerance]
        node.num_frac = len(node.indices_frac)
        node.num_int = len(c) - node.num_frac
        
        # Choose between branching and cutting
        if self.use_cuts and np.random.random() < self.cut_probability:
            self._add_gomory_cut(node, c)
        else:
            self._branch(node, c)
    
    def _add_gomory_cut(self, node, c):
        """
        Add Gomory cuts to create a new node.
        
        Args:
            node: Node to cut from
            c: Objective coefficients
        """
        self.logger.debug("Generating Gomory cut")
        
        # Check if we have a valid tableau
        if not hasattr(node, 'tableau') or node.tableau is None:
            self.logger.warning("No tableau available for Gomory cut generation")
            self._branch(node, c)
            return
        
        # Use a small tolerance for numerical stability
        FRAC_TOL = self.tolerance
        
        tableau = node.tableau
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
            self.logger.debug("No suitable rows found for Gomory cut generation")
            self._branch(node, c)
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
        violation = abs(np.dot(cut_coeffs, node.relaxed_soln) - cut_rhs)
        if violation < FRAC_TOL:
            self.logger.debug("Generated cut is not violated by current solution")
            self._branch(node, c)
            return
        
        self.logger.debug(f"Generated Gomory cut from row {cut_row_idx}")
        self.logger.debug(f"Cut violation: {violation}")
        
        # Add the cut to the current constraints
        new_A_ub = np.vstack([node.A_ub, cut_coeffs])
        new_b_ub = np.append(node.b_ub, cut_rhs)
        
        # Store the cut details as a string for visualization
        cut_description = f"Cut: {' + '.join([f'{coeff:.2f}x{i}' for i, coeff in enumerate(cut_coeffs) if abs(coeff) > FRAC_TOL])} ≤ {cut_rhs:.2f}"
        
        # Create new node with the Gomory cut
        child_node = Node(parent=node, branch_direction="gomory")
        child_node.id = self._get_next_node_id()
        child_node.set_constraints(new_A_ub, new_b_ub, node.tableau)
        child_node.branch_val = cut_description
        
        # Solve LP relaxation for the new node
        result = self._solve_lp_relaxation(c, new_A_ub, new_b_ub)
        self.logger.increment_stat("lp_relaxations")
        self.logger.increment_stat("cuts_added")
        
        if result.success:
            child_node.relaxed_soln = result.x
            child_node.value = result.fun
            child_node.local_upper_bound = child_node.value
            child_node.tableau = result.tableau
            
            self.logger.debug(f"New cut node value: {child_node.value}")
            self._add_node_to_tree(child_node, c, new_A_ub, new_b_ub)
            
            if child_node.value > self.global_lower_bound:
                self.priority_queue.push(child_node)
                self.logger.debug(f"Added cut node to priority queue")
            else:
                self.logger.debug(f"Cut node pruned: suboptimal")
                child_node.prune_reason = 'suboptimal'
                self._update_node_attributes(child_node, {'color': 'orange', 
                                                       'prune_reason': 'suboptimal'})
                self.logger.increment_stat("nodes.pruned_bound")
        else:
            self.logger.debug(f"Cut node is infeasible")
            child_node.prune_reason = 'infeasible'
            self._add_node_to_tree(child_node, c, new_A_ub, new_b_ub)
            self._update_node_attributes(child_node, {'color': 'red', 
                                                   'prune_reason': 'infeasible'})
            self.logger.increment_stat("nodes.pruned_infeasible")
    
    def _branch(self, node, c):
        """
        Create two new nodes by branching on a fractional variable.
        
        Args:
            node: Node to branch from
            c: Objective coefficients
        """
        self.logger.debug("Branching on a fractional variable")
        
        with self.logger.timers["branching"]:
            # Select branching variable
            try:
                branch_var = self.branching_strategy.select_branching_variable(node)
            except Exception as e:
                self.logger.error(f"Error selecting branching variable: {e}")
                # Fall back to most fractional
                branch_var = node.indices_frac[0] if node.indices_frac else 0
                
            branch_val = node.relaxed_soln[branch_var]
            self.logger.debug(f"Branching on variable X{branch_var} with value {branch_val}")
            
            # Create branches
            for direction in ['floor', 'ceil']:
                new_A_ub = node.A_ub.copy()
                new_b_ub = node.b_ub.copy()
                
                if direction == 'floor':
                    new_val = np.floor(branch_val)
                    self.logger.debug(f"Creating {direction} branch (x{branch_var} <= {new_val})")
                    new_constraint = np.zeros(len(c))
                    new_constraint[branch_var] = 1
                    new_A_ub = np.vstack([new_A_ub, new_constraint])
                    new_b_ub = np.append(new_b_ub, new_val)
                else:  # ceiling branch
                    new_val = np.ceil(branch_val)
                    self.logger.debug(f"Creating {direction} branch (x{branch_var} >= {new_val})")
                    new_constraint = np.zeros(len(c))
                    new_constraint[branch_var] = -1
                    new_A_ub = np.vstack([new_A_ub, new_constraint])
                    new_b_ub = np.append(new_b_ub, -new_val)
                
                # Create child node
                child_node = Node(parent=node, branch_var=branch_var, 
                                 branch_val=new_val, branch_direction=direction)
                child_node.id = self._get_next_node_id()
                child_node.set_constraints(new_A_ub, new_b_ub)
                
                # Solve LP relaxation for child node
                result = self._solve_lp_relaxation(c, new_A_ub, new_b_ub)
                self.logger.increment_stat("lp_relaxations")
                
                if result.success:
                    child_node.relaxed_soln = result.x
                    child_node.value = result.fun
                    child_node.local_upper_bound = child_node.value
                    child_node.tableau = result.tableau
                    
                    self.logger.debug(f"New {direction} node value: {child_node.value}")
                    self._add_node_to_tree(child_node, c, new_A_ub, new_b_ub)
                    
                    if child_node.value > self.global_lower_bound:
                        self.priority_queue.push(child_node)
                        self.logger.debug(f"Added {direction} node to priority queue")
                    else:
                        self.logger.debug(f"{direction} node pruned: suboptimal")
                        child_node.prune_reason = 'suboptimal'
                        self._update_node_attributes(child_node, {'color': 'orange', 
                                                              'prune_reason': 'suboptimal'})
                        self.logger.increment_stat("nodes.pruned_bound")
                else:
                    self.logger.debug(f"{direction} node is infeasible")
                    child_node.prune_reason = 'infeasible'
                    self._add_node_to_tree(child_node, c, new_A_ub, new_b_ub)
                    self._update_node_attributes(child_node, {'color': 'red', 
                                                          'prune_reason': 'infeasible'})
                    self.logger.increment_stat("nodes.pruned_infeasible")
    
    def _solve_lp_relaxation(self, c, A_ub, b_ub, A_eq=None, b_eq=None):
        """
        Solve the LP relaxation of a node.
        
        Args:
            c: Objective coefficients
            A_ub: Inequality constraint matrix
            b_ub: Inequality RHS vector
            A_eq: Equality constraints (optional)
            b_eq: Equality RHS (optional)
            
        Returns:
            SimplexResult: Result of LP relaxation
        """
        from src.simplex import linprog_simplex, SimplexResult, PivOptions
        
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
        
        Args:
            c: Original objective coefficients
            A_ub: Original constraint matrix
            b_ub: Original RHS vector
        """
        # Store original problem data as lists in graph attributes
        self.enumeration_tree.graph['og_obj_coefs'] = c.tolist()
        self.enumeration_tree.graph['og_constraints'] = A_ub.tolist()
        self.enumeration_tree.graph['og_rhs'] = b_ub.tolist()
    
    def _add_node_to_tree(self, node, c, A_ub, b_ub):
        """
        Add a node to the branch-and-bound tree.
        
        Args:
            node: Node to add
            c: Objective coefficients
            A_ub: Constraint matrix
            b_ub: RHS vector
        """
        # Get default attributes then update with node-specific values
        attributes = self._get_default_node_attributes()
        attributes.update({
            'depth': node.depth,
            'branch_variable': node.branch_var,
            'branch_value': node.branch_val,
            'branch_direction': node.branch_direction,
            'local_upper_bound': node.local_upper_bound,
            'current_constraints': A_ub.tolist(),
            'current_rhs': b_ub.tolist(),
        })
        
        # Add node to the tree with its attributes
        self.enumeration_tree.add_node(node.id, **attributes)
        
        # If node has a parent, add edge to represent relationship
        if node.parent:
            self.enumeration_tree.add_edge(node.parent.id, node.id)
    
    def _update_node_attributes(self, node, attributes):
        """
        Update attributes of a node in the enumeration tree.
        
        Args:
            node: Node to update
            attributes: New attributes
        """
        # Get current attributes from tree
        current_attributes = self.enumeration_tree.nodes[node.id]
        
        # Update only existing attributes, converting numpy arrays to lists
        for key, value in attributes.items():
            if key in current_attributes:
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                current_attributes[key] = value
    
    def _get_default_node_attributes(self):
        """
        Get default attributes for a new node.
        
        Returns:
            dict: Default node attributes
        """
        return {
            'depth': 0,
            'branch_variable': None,
            'branch_value': None,
            'branch_direction': None,
            'fractionality': None,
            'local_lower_bound': -np.inf,
            'local_upper_bound': np.inf,
            'global_lower_bound': -np.inf,
            'global_upper_bound': np.inf,
            'current_constraints': None,
            'current_rhs': None,
            'relaxed_soln': None,
            'relaxed_obj_value': None,
            'num_int': 0,
            'num_frac': 0,
            'indices_frac': [],
            'active_constraints': [],
            'slack_values': [],
            'optimality_gap': np.inf,
            'children_pruned': 0,
            'color': 'lightgray'
        }
    
    def _visualize_tree(self, problem_name):
        """
        Generate visualization of branch-and-bound tree.
        
        Args:
            problem_name: Name of the problem for file naming
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
            elif node_data['prune_reason'] == 'infeasible':
                colors.append('red')
            elif node_data['prune_reason'] == 'suboptimal':
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
        
        plt.title(f"ILP Branch and Bound Enumeration Tree - {problem_name}")
        plt.tight_layout()
        
        # Create a directory for plots if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        # Save the plot as a PNG file with high DPI
        plot_filename = f"plots/{problem_name.replace(' ', '_').lower()}_tree.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Plot saved as {plot_filename}")