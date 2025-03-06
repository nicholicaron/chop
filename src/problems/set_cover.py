"""
Set Cover Problem implementation.

This module provides a Set Cover Problem class that implements the OptimizationProblem interface,
with utilities for creating, visualizing, and solving Set Cover instances.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Any, Union, Callable

from .base import OptimizationProblem


class SetCover(OptimizationProblem):
    """
    Set Cover Problem implementation.
    
    In the Set Cover problem, we have a universe of elements and a collection of sets,
    where each set covers some elements of the universe and has an associated cost.
    The goal is to select a subset of the sets such that all elements in the universe
    are covered, while minimizing the total cost.
    
    Attributes:
        n_elements (int): Number of elements in the universe
        n_sets (int): Number of available sets
        coverage_matrix (np.ndarray): Binary matrix where coverage_matrix[i,j] = 1 if set j covers element i
        set_costs (np.ndarray): Cost of each set
        problem_name (str): Name of this Set Cover instance
        problem_difficulty (str): Difficulty level of this instance
    """
    
    def size_metrics(self) -> Dict[str, int]:
        """
        Get detailed size metrics for benchmarking purposes.
        
        Returns:
            Dict[str, int]: Dictionary of size metrics specific to Set Cover
        """
        return {
            'size_elements': self.n_elements,
            'size_sets': self.n_sets,
            'size_variables': self.n_sets,
            'size_constraints': self.n_elements + 1  # Element coverage + one constraint for objective
        }
    
    def get_specific_metrics(self) -> Dict[str, Any]:
        """
        Get problem-specific metrics for Set Cover benchmarking.
        
        Returns:
            Dict[str, Any]: Dictionary of Set Cover-specific metrics
        """
        # Calculate metrics like density, redundancy, etc.
        density = np.mean(self.coverage_matrix)
        
        # Calculate how many sets cover each element on average
        sets_per_element = np.sum(self.coverage_matrix, axis=1).mean()
        
        # Calculate how many elements each set covers on average
        elements_per_set = np.sum(self.coverage_matrix, axis=0).mean()
        
        # Calculate cost statistics
        cost_mean = np.mean(self.set_costs) if hasattr(self, 'set_costs') and self.set_costs is not None else 1.0
        cost_std = np.std(self.set_costs) if hasattr(self, 'set_costs') and self.set_costs is not None else 0.0
        
        # Min number of sets needed is a lower bound on the optimal solution
        min_sets_required = min(self.n_sets, int(np.ceil(self.n_elements / elements_per_set)))
        
        return {
            'density': density,
            'sets_per_element': sets_per_element,
            'elements_per_set': elements_per_set,
            'cost_mean': cost_mean,
            'cost_std': cost_std,
            'min_sets_required': min_sets_required,
            'is_minimization': True
        }
    
    def __init__(self, coverage_matrix: np.ndarray, set_costs: np.ndarray = None, 
                 name: str = None, difficulty: str = 'medium'):
        """
        Initialize a Set Cover Problem instance.
        
        Args:
            coverage_matrix: Binary matrix where coverage_matrix[i,j] = 1 if set j covers element i
            set_costs: Cost of each set (defaults to all 1's for unweighted set cover)
            name: Name for this problem instance
            difficulty: Difficulty level for this instance
        """
        self.coverage_matrix = np.array(coverage_matrix, dtype=int)
        
        # Validate input
        if len(self.coverage_matrix.shape) != 2:
            raise ValueError("Coverage matrix must be 2-dimensional")
        
        # Initialize dimensions
        self.n_elements, self.n_sets = self.coverage_matrix.shape
        
        # If no costs provided, use uniform costs (unweighted set cover)
        if set_costs is None:
            self.set_costs = np.ones(self.n_sets)
        else:
            self.set_costs = np.array(set_costs)
            
        # Validate costs
        if len(self.set_costs) != self.n_sets:
            raise ValueError("Number of set costs must match number of sets")
        if any(cost <= 0 for cost in self.set_costs):
            raise ValueError("Set costs must be positive")
            
        # Set name and difficulty
        self._name = name or f"SetCover_{self.n_elements}x{self.n_sets}"
        self._difficulty = difficulty
        
        # Create a directory for plots if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
    def to_ilp(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert Set Cover Problem to ILP formulation.
        
        The ILP formulation uses binary variables x_j where:
        x_j = 1 if set j is selected, 0 otherwise
        
        With constraints:
        - For each element i, at least one set that covers i must be selected
        
        Returns:
            Tuple containing:
            - c: Objective coefficients (set costs to minimize)
            - A_eq: Equality constraint matrix (empty)
            - b_eq: Equality constraint right-hand side (empty)
            - A_ub: Inequality constraint matrix (coverage constraints)
            - b_ub: Inequality constraint right-hand side (all -1's)
        """
        # Objective: minimize sum of costs of selected sets
        # Since our solver maximizes, we negate the costs
        c = -self.set_costs.copy()
        
        # Coverage constraints: For each element i, sum of x_j for sets containing i >= 1
        # We rearrange to: -sum_j a_ij * x_j <= -1 for all i
        # where a_ij = 1 if set j contains element i, 0 otherwise
        
        A_ub = -self.coverage_matrix  # Negate for standard form
        b_ub = -np.ones(self.n_elements)  # Each element must be covered at least once
        
        # No equality constraints
        A_eq = np.zeros((0, self.n_sets))
        b_eq = np.zeros(0)
        
        return c, A_eq, b_eq, A_ub, b_ub
    
    def validate_solution(self, solution: np.ndarray) -> Tuple[bool, float]:
        """
        Check if a solution is valid and compute its objective value.
        
        A valid set cover solution must:
        1. Be binary (0 or 1 values)
        2. Cover all elements in the universe
        
        Args:
            solution: Binary solution vector indicating which sets are selected
            
        Returns:
            Tuple containing:
            - is_valid: Boolean indicating if the solution is valid
            - objective_value: The objective value (total cost of selected sets)
        """
        # Check binary values
        TOLERANCE = 1e-4
        if not all((x < TOLERANCE) or (abs(x - 1) < TOLERANCE) for x in solution):
            return False, 0.0
        
        # Check that all elements are covered
        selected_sets = solution > 0.5
        covered_elements = np.zeros(self.n_elements, dtype=bool)
        
        for j in range(self.n_sets):
            if selected_sets[j]:
                # Mark elements covered by this set
                for i in range(self.n_elements):
                    if self.coverage_matrix[i, j]:
                        covered_elements[i] = True
        
        # All elements must be covered
        if not np.all(covered_elements):
            return False, 0.0
        
        # Calculate objective value: sum of costs of selected sets
        objective_value = np.sum(solution * self.set_costs)
        
        return True, objective_value
    
    def visualize_instance(self, title: str = None, figsize: Tuple[int, int] = (12, 8), **kwargs) -> str:
        """
        Visualize the Set Cover Problem instance.
        
        Creates a bipartite graph showing elements, sets, and their relationships.
        
        Args:
            title: Optional title for the visualization
            figsize: Figure size as (width, height)
            **kwargs: Additional visualization parameters
            
        Returns:
            str: Path to the saved visualization file
        """
        plt.figure(figsize=figsize)
        
        # Create a bipartite graph
        G = nx.Graph()
        
        # Add element nodes on the left
        element_nodes = [f"e{i}" for i in range(self.n_elements)]
        for i, node in enumerate(element_nodes):
            G.add_node(node, bipartite=0, pos=(-1, self.n_elements - i))
        
        # Add set nodes on the right
        set_nodes = [f"S{j}" for j in range(self.n_sets)]
        for j, node in enumerate(set_nodes):
            G.add_node(node, bipartite=1, pos=(1, self.n_sets - j))
        
        # Add edges between elements and sets that cover them
        for i in range(self.n_elements):
            for j in range(self.n_sets):
                if self.coverage_matrix[i, j]:
                    G.add_edge(element_nodes[i], set_nodes[j])
        
        # Get node positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Draw the graph
        plt.figure(figsize=figsize)
        
        # Draw element nodes
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=element_nodes, 
                              node_color='lightblue', 
                              node_size=500,
                              label='Elements')
        
        # Draw set nodes with size proportional to cost
        set_sizes = [300 + 200 * (cost / max(self.set_costs)) for cost in self.set_costs]
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=set_nodes, 
                              node_color='lightgreen', 
                              node_size=set_sizes,
                              label='Sets')
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        
        # Draw labels
        element_labels = {node: node for node in element_nodes}
        set_labels = {node: f"{node}\nCost: {self.set_costs[j]:.1f}" 
                     for j, node in enumerate(set_nodes)}
        nx.draw_networkx_labels(G, pos, labels=element_labels)
        nx.draw_networkx_labels(G, pos, labels=set_labels)
        
        # Add coverage matrix visualization
        coverage_ax = plt.axes([0.05, 0.05, 0.3, 0.3])  # [left, bottom, width, height]
        coverage_ax.matshow(self.coverage_matrix, cmap='Blues')
        coverage_ax.set_title("Coverage Matrix")
        coverage_ax.set_xlabel("Sets")
        coverage_ax.set_ylabel("Elements")
        coverage_ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Add title and adjust layout
        plt_title = title or f"Set Cover Instance: {self.name}"
        plt.suptitle(plt_title)
        plt.axis('off')
        
        # Add legend with instance information
        info_text = (
            f"Elements: {self.n_elements}\n"
            f"Sets: {self.n_sets}\n"
            f"Total cost: {np.sum(self.set_costs):.1f}\n"
            f"Min possible sets: {min(self.n_elements, self.n_sets)}"
        )
        plt.figtext(0.05, 0.95, info_text, ha='left', va='top', 
                   bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
        
        # Save plot
        filename = f"plots/set_cover_instance_{self.name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        return filename
   
    def visualize_solution(self, solution: np.ndarray, is_optimal: bool = False, 
                         figsize: Tuple[int, int] = (12, 8), **kwargs) -> str:
        """
        Visualize a solution to the Set Cover Problem.
        
        Args:
            solution: Binary solution vector indicating which sets are selected
            is_optimal: Whether the solution is optimal
            figsize: Figure size as (width, height)
            **kwargs: Additional visualization parameters
            
        Returns:
            str: Path to the saved visualization file
        """
        plt.figure(figsize=figsize)
        
        # Create a bipartite graph
        G = nx.Graph()
        
        # Add element nodes on the left
        element_nodes = [f"e{i}" for i in range(self.n_elements)]
        for i, node in enumerate(element_nodes):
            G.add_node(node, bipartite=0, pos=(-1, self.n_elements - i))
        
        # Add set nodes on the right
        set_nodes = [f"S{j}" for j in range(self.n_sets)]
        for j, node in enumerate(set_nodes):
            G.add_node(node, bipartite=1, pos=(1, self.n_sets - j), selected=solution[j] > 0.5)
        
        # Add edges between elements and sets that cover them
        for i in range(self.n_elements):
            for j in range(self.n_sets):
                if self.coverage_matrix[i, j]:
                    G.add_edge(element_nodes[i], set_nodes[j], 
                             used=self.coverage_matrix[i, j] and solution[j] > 0.5)
        
        # Get node positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Get selected sets
        selected = [node for node, data in G.nodes(data=True) 
                  if data.get('bipartite') == 1 and data.get('selected', False)]
        unselected = [node for node, data in G.nodes(data=True) 
                    if data.get('bipartite') == 1 and not data.get('selected', False)]
        
        # Get edges for visualization
        used_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('used', False)]
        unused_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('used', False)]
        
        # Draw the graph
        plt.figure(figsize=figsize)
        
        # Draw element nodes
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=element_nodes, 
                              node_color='lightblue', 
                              node_size=500)
        
        # Draw unselected set nodes (faded)
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=unselected, 
                              node_color='lightgray', 
                              node_size=300,
                              alpha=0.5)
        
        # Draw selected set nodes (highlighted)
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=selected, 
                              node_color='lightgreen', 
                              node_size=500,
                              alpha=1.0)
        
        # Draw unused edges (faded)
        nx.draw_networkx_edges(G, pos, 
                              edgelist=unused_edges, 
                              width=0.5, 
                              alpha=0.2, 
                              edge_color='gray')
        
        # Draw used edges (highlighted)
        nx.draw_networkx_edges(G, pos, 
                              edgelist=used_edges, 
                              width=2.0, 
                              alpha=1.0, 
                              edge_color='green')
        
        # Draw labels
        element_labels = {node: node for node in element_nodes}
        set_labels = {}
        for j, node in enumerate(set_nodes):
            if solution[j] > 0.5:
                set_labels[node] = f"{node}\nCost: {self.set_costs[j]:.1f}"
            else:
                set_labels[node] = node
        nx.draw_networkx_labels(G, pos, labels=element_labels)
        nx.draw_networkx_labels(G, pos, labels=set_labels)
        
        # Add coverage matrix visualization with selected sets highlighted
        coverage_ax = plt.axes([0.05, 0.05, 0.3, 0.3])
        coverage_display = self.coverage_matrix.copy()
        
        # Highlight selected sets in the coverage matrix
        selected_idx = np.where(solution > 0.5)[0]
        coverage_display_rgb = np.zeros((self.n_elements, self.n_sets, 3))
        
        for i in range(self.n_elements):
            for j in range(self.n_sets):
                if j in selected_idx and coverage_display[i, j]:
                    # Green for selected sets that provide coverage
                    coverage_display_rgb[i, j] = [0, 0.8, 0]
                elif j in selected_idx:
                    # Light green for selected sets
                    coverage_display_rgb[i, j] = [0.7, 1, 0.7]
                elif coverage_display[i, j]:
                    # Blue for coverage by unselected sets
                    coverage_display_rgb[i, j] = [0, 0, 0.8]
                else:
                    # White for no coverage
                    coverage_display_rgb[i, j] = [1, 1, 1]
        
        coverage_ax.imshow(coverage_display_rgb, aspect='auto')
        coverage_ax.set_title("Coverage Matrix (green = selected)")
        coverage_ax.set_xlabel("Sets")
        coverage_ax.set_ylabel("Elements")
        coverage_ax.set_xticks(range(self.n_sets))
        coverage_ax.set_yticks(range(self.n_elements))
        coverage_ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Calculate solution statistics
        selected_sets = np.where(solution > 0.5)[0]
        total_cost = sum(self.set_costs[j] for j in selected_sets)
        redundancy = 0
        for i in range(self.n_elements):
            # Count how many selected sets cover each element
            covers = sum(self.coverage_matrix[i, j] for j in selected_sets)
            # Redundancy is the sum of extra covers beyond the required one
            redundancy += max(0, covers - 1)
        
        # Add title and statistics
        status = "Optimal" if is_optimal else "Candidate"
        info_text = (
            f"Sets selected: {len(selected_sets)}/{self.n_sets}\n"
            f"Total cost: {total_cost:.1f}\n"
            f"Coverage redundancy: {redundancy}"
        )
        plt.figtext(0.05, 0.95, info_text, ha='left', va='top', 
                  bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
        
        plt.suptitle(f"{status} Solution for {self.name}")
        plt.axis('off')
        
        # Save plot
        counter = kwargs.get('counter', 1)
        filename = f"plots/{self.name}_solution_{counter:03d}_{status.lower()}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        return filename
    
    @classmethod
    def generate_random_instance(cls, n_elements: int = 20, 
                               n_sets: int = 10,
                               density: float = 0.3,
                               min_cost: float = 1.0,
                               max_cost: float = 10.0,
                               ensure_feasible: bool = True,
                               seed: int = None,
                               name: str = None,
                               difficulty: str = 'medium') -> 'SetCover':
        """
        Generate a random Set Cover instance.
        
        Args:
            n_elements: Number of elements in the universe
            n_sets: Number of available sets
            density: Probability that a set covers an element (0.0-1.0)
            min_cost: Minimum cost of a set
            max_cost: Maximum cost of a set
            ensure_feasible: Whether to ensure that the instance is feasible
            seed: Random seed for reproducibility
            name: Name for the instance
            difficulty: Difficulty level
            
        Returns:
            SetCover: A new randomly generated Set Cover instance
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Generate random coverage matrix
        coverage_matrix = np.random.random((n_elements, n_sets)) < density
        
        # If ensuring feasibility, make sure each element is covered by at least one set
        if ensure_feasible:
            for i in range(n_elements):
                if not np.any(coverage_matrix[i, :]):
                    # If element i is not covered, randomly select a set to cover it
                    j = random.randint(0, n_sets - 1)
                    coverage_matrix[i, j] = True
        
        # Generate random set costs
        set_costs = np.random.uniform(min_cost, max_cost, n_sets)
        
        instance_name = name or f"Random_SetCover_{n_elements}x{n_sets}"
        
        return cls(coverage_matrix, set_costs, name=instance_name, difficulty=difficulty)
    
    @classmethod
    def generate_benchmark_suite(cls, difficulty_levels: List[str] = None) -> Dict[str, List['SetCover']]:
        """
        Generate a suite of benchmark Set Cover instances at different difficulty levels.
        
        Args:
            difficulty_levels: List of difficulty levels to generate
                             (defaults to ['easy', 'medium', 'hard'])
            
        Returns:
            Dict[str, List[SetCover]]: Dictionary mapping difficulty levels to lists of instances
        """
        if difficulty_levels is None:
            difficulty_levels = ['easy', 'medium', 'hard']
            
        suite = {}
        
        # Define instance parameters for each difficulty level
        configs = {
            'easy': {
                'element_sizes': [10, 15, 20],
                'set_multiplier': 0.8,  # fewer sets than elements
                'density': 0.4,         # high density
                'cost_range': (1.0, 5.0)
            },
            'medium': {
                'element_sizes': [30, 50, 75],
                'set_multiplier': 1.0,  # same number of sets as elements
                'density': 0.2,         # medium density
                'cost_range': (1.0, 10.0)
            },
            'hard': {
                'element_sizes': [100, 150, 200],
                'set_multiplier': 2.0,  # more sets than elements
                'density': 0.1,         # low density
                'cost_range': (1.0, 20.0)
            }
        }
        
        # Generate instances for each difficulty level
        for level in difficulty_levels:
            if level not in configs:
                continue
                
            suite[level] = []
            config = configs[level]
            
            for n_elements in config['element_sizes']:
                # Calculate number of sets based on multiplier
                n_sets = max(5, int(n_elements * config['set_multiplier']))
                
                for i in range(3):  # Generate 3 instances of each size
                    # Use deterministic seed for reproducibility
                    seed = hash(f"{level}_{n_elements}_{i}") % 10000
                    
                    name = f"{level.capitalize()}_SetCover_{n_elements}x{n_sets}_{i+1}"
                    min_cost, max_cost = config['cost_range']
                    
                    instance = cls.generate_random_instance(
                        n_elements=n_elements,
                        n_sets=n_sets,
                        density=config['density'],
                        min_cost=min_cost,
                        max_cost=max_cost,
                        ensure_feasible=True,
                        seed=seed,
                        name=name,
                        difficulty=level
                    )
                    suite[level].append(instance)
        
        return suite
    
    def get_constraint_generator(self) -> Optional[Callable]:
        """
        Return a function that generates additional constraints during branch-and-bound.
        
        For Set Cover Problem, we don't need lazy constraints, so return None.
        
        Returns:
            None: Set Cover doesn't require lazy constraint generation
        """
        return None
    
    @property
    def name(self) -> str:
        """Get the name of this Set Cover instance."""
        return self._name
    
    @property
    def size(self) -> Dict[str, int]:
        """Get size metrics for this Set Cover instance."""
        return {
            'elements': self.n_elements,
            'sets': self.n_sets,
            'variables': self.n_sets,
            'constraints': self.n_elements
        }
    
    @property
    def difficulty(self) -> str:
        """Get the difficulty level of this Set Cover instance."""
        return self._difficulty


def create_predefined_instances() -> Dict[str, SetCover]:
    """
    Create predefined Set Cover instances for examples and testing.
    
    Returns:
        Dict[str, SetCover]: Dictionary of named Set Cover instances
    """
    instances = {}
    
    # Example 1: Small instance
    # 5 elements, 4 sets
    coverage_1 = np.array([
        [1, 1, 0, 0],  # Element 0 is covered by sets 0, 1
        [1, 0, 1, 0],  # Element 1 is covered by sets 0, 2
        [1, 0, 0, 1],  # Element 2 is covered by sets 0, 3
        [0, 1, 1, 0],  # Element 3 is covered by sets 1, 2
        [0, 1, 0, 1]   # Element 4 is covered by sets 1, 3
    ])
    costs_1 = np.array([5.0, 3.0, 3.0, 3.0])
    instances["small"] = SetCover(
        coverage_matrix=coverage_1, 
        set_costs=costs_1, 
        name="Small_SetCover",
        difficulty="easy"
    )
    
    # Example 2: Medium instance
    # 8 elements, 5 sets with varied coverage
    coverage_2 = np.array([
        [1, 0, 0, 0, 1],  # Element 0 is covered by sets 0, 4
        [1, 1, 0, 0, 0],  # Element 1 is covered by sets 0, 1
        [1, 0, 1, 0, 0],  # Element 2 is covered by sets 0, 2
        [1, 0, 0, 1, 0],  # Element 3 is covered by sets 0, 3
        [0, 1, 0, 0, 1],  # Element 4 is covered by sets 1, 4
        [0, 1, 1, 0, 0],  # Element 5 is covered by sets 1, 2
        [0, 0, 1, 1, 0],  # Element 6 is covered by sets 2, 3
        [0, 0, 0, 1, 1]   # Element 7 is covered by sets 3, 4
    ])
    costs_2 = np.array([10.0, 5.0, 5.0, 5.0, 5.0])
    instances["medium"] = SetCover(
        coverage_matrix=coverage_2, 
        set_costs=costs_2, 
        name="Medium_SetCover",
        difficulty="medium"
    )
    
    # Example 3: Classic set cover instance
    # 10 elements, 8 sets with overlapping coverage
    coverage_3 = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0],  # Elements 0, 1 covered by sets 0, 1
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],  # Elements 2, 3 covered by sets 1, 2
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],  # Elements 4, 5 covered by sets 2, 3
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],  # Elements 6, 7 covered by sets 3, 4
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],  # Elements 8, 9 covered by sets 4, 5, 6
        [0, 0, 0, 0, 0, 1, 1, 1]   # Elements 9 also covered by set 7
    ])
    costs_3 = np.array([3.0, 4.0, 4.0, 4.0, 3.0, 2.0, 2.0, 1.0])
    instances["classic"] = SetCover(
        coverage_matrix=coverage_3, 
        set_costs=costs_3, 
        name="Classic_SetCover",
        difficulty="hard"
    )
    
    return instances