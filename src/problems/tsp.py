"""
Traveling Salesman Problem implementation.

This module provides a TSP problem class that implements the OptimizationProblem interface,
with utilities for creating, visualizing, and solving TSP instances.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import random
import os
from typing import Dict, List, Tuple, Set, Optional, Any, Union, Callable

from .base import OptimizationProblem


class TSP(OptimizationProblem):
    """
    Traveling Salesman Problem implementation.
    
    Attributes:
        n_cities (int): Number of cities
        coordinates (dict): Dictionary mapping city index to (x,y) coordinates
        distances (dict): Dictionary mapping city pairs to distances
        graph (nx.Graph): NetworkX graph representation
        plot_counter (int): Counter for naming plot files
        problem_name (str): Name of this TSP instance
        problem_difficulty (str): Difficulty level of this instance
    """
    
    def size_metrics(self) -> Dict[str, int]:
        """
        Get detailed size metrics for benchmarking purposes.
        
        Returns:
            Dict[str, int]: Dictionary of size metrics specific to TSP
        """
        return {
            'size_cities': self.n_cities,
            'size_edges': self.n_cities * (self.n_cities - 1) // 2,
            'size_variables': self.n_cities * (self.n_cities - 1),
            'size_constraints': 2 * self.n_cities  # Degree constraints
        }
    
    def get_specific_metrics(self) -> Dict[str, Any]:
        """
        Get problem-specific metrics for TSP benchmarking.
        
        Returns:
            Dict[str, Any]: Dictionary of TSP-specific metrics
        """
        # Calculate metrics like average distance, distance variance, etc.
        distances = list(self.distances.values())
        return {
            'avg_distance': np.mean(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'std_distance': np.std(distances),
            'distance_range': np.max(distances) - np.min(distances)
        }
    
    def __init__(self, n_cities: int, coordinates: Dict[int, Tuple[float, float]] = None, 
                 name: str = None, difficulty: str = 'medium'):
        """
        Initialize a TSP instance.
        
        Args:
            n_cities: Number of cities
            coordinates: Dictionary mapping city index to (x,y) coordinates (random if None)
            name: Name for this problem instance
            difficulty: Difficulty level for this instance
        """
        self.n_cities = n_cities
        self._name = name or f"TSP_{n_cities}"
        self._difficulty = difficulty
        
        # Generate or use provided coordinates
        if coordinates is None:
            self.coordinates = {i: (random.uniform(0, 100), random.uniform(0, 100)) 
                              for i in range(n_cities)}
        else:
            self.coordinates = coordinates
            
        # Compute distances between all pairs of cities
        self.distances = {}
        for i, j in combinations(range(n_cities), 2):
            x1, y1 = self.coordinates[i]
            x2, y2 = self.coordinates[j]
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            self.distances[(i, j)] = dist
            self.distances[(j, i)] = dist
            
        # Create graph representation
        self.graph = nx.Graph()
        for i in range(n_cities):
            self.graph.add_node(i, pos=self.coordinates[i])
        for (i, j), dist in self.distances.items():
            if i < j:
                self.graph.add_edge(i, j, weight=dist)
        
        # Initialize plot counter
        self.plot_counter = 0
        os.makedirs('plots', exist_ok=True)
    
    def to_ilp(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert TSP to ILP formulation with degree constraints.
        
        Returns:
            Tuple containing:
            - c: Objective coefficients (maximizing negative distances)
            - A_eq: Equality constraint matrix (degree constraints)
            - b_eq: Equality constraint RHS
            - A_ub: Inequality constraint matrix (initial empty)
            - b_ub: Inequality constraint RHS (initial empty)
        """
        n = self.n_cities
        num_vars = n * (n - 1) // 2
        
        # Objective: Minimize distances (maximize negative distances)
        c = np.zeros(num_vars)
        for i, j in combinations(range(n), 2):
            idx = self._get_variable_index(i, j)
            c[idx] = -self.distances[(i, j)]
        
        # Degree constraints: Each city must have exactly 2 edges
        A_eq = []
        b_eq = []
        
        for i in range(n):
            row = np.zeros(num_vars)
            for j in range(n):
                if i != j:
                    idx = self._get_variable_index(min(i, j), max(i, j))
                    row[idx] = 1
            A_eq.append(row)
            b_eq.append(2)  # Exactly 2 edges per vertex
        
        # Initialize empty inequality constraints (will be added during B&B)
        A_ub = np.zeros((0, num_vars))
        b_ub = np.zeros(0)
        
        return np.array(c), np.array(A_eq), np.array(b_eq), np.array(A_ub), np.array(b_ub)
    
    def validate_solution(self, solution: np.ndarray) -> Tuple[bool, float]:
        """
        Check if a solution is valid and compute its objective value.
        
        A valid TSP solution must:
        1. Include exactly 2 edges for each city
        2. Form a single connected tour (no subtours)
        
        Args:
            solution: Binary solution vector
            
        Returns:
            Tuple containing:
            - is_valid: Boolean indicating if the solution is valid
            - objective_value: The objective value (total tour length)
        """
        # Use a tolerance for binary values
        EDGE_TOLERANCE = 1e-4
        
        # Create a graph from the solution
        G = nx.Graph()
        for i in range(self.n_cities):
            G.add_node(i)
        
        edge_count = 0
        for i, j in combinations(range(self.n_cities), 2):
            idx = self._get_variable_index(i, j)
            if solution[idx] > 1 - EDGE_TOLERANCE:
                G.add_edge(i, j)
                edge_count += 1
        
        # Check degree constraints (each city must have exactly 2 edges)
        for node in G.nodes():
            if G.degree(node) != 2:
                return False, 0.0
        
        # Check connectedness (must be a single tour)
        if not nx.is_connected(G) or len(list(nx.connected_components(G))) > 1:
            return False, 0.0
        
        # Calculate total tour length (negative because we maximize negative distances)
        tour_length = -sum(solution[self._get_variable_index(i, j)] * self.distances[(i, j)]
                     for i, j in combinations(range(self.n_cities), 2))
        
        return True, tour_length
    
    def visualize_instance(self, title: str = None, **kwargs) -> str:
        """
        Visualize the TSP instance and save to file.
        
        Args:
            title: Optional title for the visualization
            **kwargs: Additional visualization parameters
            
        Returns:
            str: Path to the saved visualization file
        """
        plt.figure(figsize=(10, 10))
        pos = nx.get_node_attributes(self.graph, 'pos')
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', 
                             node_size=500)
        
        # Draw edges with weights
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        edge_labels = {k: f"{v:.1f}" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels)
        nx.draw_networkx_edges(self.graph, pos)
        
        # Draw labels
        labels = {i: f"City {i}" for i in range(self.n_cities)}
        nx.draw_networkx_labels(self.graph, pos, labels)
        
        # Set title
        plt_title = title or f"TSP Instance: {self.name}"
        plt.title(plt_title)
        plt.axis('equal')
        
        # Save plot
        filename = f"plots/tsp_instance_{self.name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        return filename
   
    def visualize_solution(self, solution: np.ndarray, is_optimal: bool = False, **kwargs) -> str:
        """
        Visualize a TSP solution with rich map-like representation.
        
        Args:
            solution: Binary solution vector
            is_optimal: Whether the solution is optimal
            **kwargs: Additional visualization parameters
            
        Returns:
            str: Path to the saved visualization file
        """
        # Get additional information from kwargs
        step = kwargs.get('step', 0)
        nodes_explored = kwargs.get('nodes_explored', 0)
        elapsed_time = kwargs.get('elapsed_time', 0)
        best_obj_value = kwargs.get('best_obj_value', 0)
        title = kwargs.get('title', f"TSP Solution - {self.name}")
        animated = kwargs.get('animated', False)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), 
                                    gridspec_kw={'width_ratios': [1.5, 1]})
        pos = nx.get_node_attributes(self.graph, 'pos')
        
        # Use tolerance for binary values
        EDGE_TOLERANCE = 1e-4
        
        # Create a new graph for the solution
        solution_graph = nx.Graph()
        solution_graph.add_nodes_from(self.graph.nodes(data=True))
                
        # Add edges from the solution
        for i, j in combinations(range(self.n_cities), 2):
            idx = self._get_variable_index(i, j)
            if solution[idx] > 1 - EDGE_TOLERANCE:
                solution_graph.add_edge(i, j)

        # Calculate total distance
        total_distance = -sum(solution[self._get_variable_index(i, j)] * self.distances[(i, j)]
                         for i, j in combinations(range(self.n_cities), 2))
                         
        # Find subtours
        subtours = self.find_subtours(solution)
        
        # Main map visualization (left subplot)
        # ----------------------------------
        
        # Optional: Add a map background
        map_image = kwargs.get('map_image', None)
        if map_image:
            try:
                from PIL import Image
                img = Image.open(map_image)
                ax1.imshow(img, extent=[0, 100, 0, 100], alpha=0.3, zorder=0)
            except:
                # If map image loading fails, continue without it
                pass
        else:
            # Draw light grid for map-like background
            ax1.grid(True, linestyle='-', alpha=0.2, zorder=0)
            
        # Create dictionary of colors for subtours
        subtour_colors = {}
        color_cycle = plt.cm.tab10.colors
        if len(subtours) > 1:
            for i, subtour in enumerate(subtours):
                color_idx = i % len(color_cycle)
                for city in subtour:
                    subtour_colors[city] = color_cycle[color_idx]
                    
        # Draw edges with arrows for tour direction
        for i, j in solution_graph.edges():
            x1, y1 = pos[i]
            x2, y2 = pos[j]
            
            # Edge color based on subtour
            if len(subtours) > 1:
                # Use subtour color if cities belong to same subtour
                same_subtour = any(i in subtour and j in subtour for subtour in subtours)
                edge_color = 'r' if same_subtour else 'gray'
            else:
                edge_color = 'r'
                
            # Draw arrow
            ax1.arrow(x1, y1, (x2-x1)*0.8, (y2-y1)*0.8, 
                    head_width=2, head_length=3, 
                    fc=edge_color, ec=edge_color, 
                    length_includes_head=True, alpha=0.8, zorder=2)
        
        # Draw cities as nodes
        if len(subtours) > 1:
            # If multiple subtours, color by subtour
            for i in range(self.n_cities):
                city_color = subtour_colors.get(i, 'lightblue')
                ax1.scatter(pos[i][0], pos[i][1], s=300, color=city_color, 
                           edgecolor='black', zorder=3)
        else:
            # Single tour, uniform color
            ax1.scatter([pos[i][0] for i in range(self.n_cities)], 
                       [pos[i][1] for i in range(self.n_cities)], 
                       s=300, color='lightblue', edgecolor='black', zorder=3)
        
        # Add city labels
        for i in range(self.n_cities):
            ax1.text(pos[i][0], pos[i][1], f"{i}", fontsize=12, 
                    ha='center', va='center', fontweight='bold', zorder=4)
                    
        # Add distance labels to edges
        if self.n_cities <= 10:  # Only show for smaller instances to avoid clutter
            for i, j in solution_graph.edges():
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                # Position label at midpoint of edge
                ax1.text((x1+x2)/2, (y1+y2)/2, f"{self.distances[(i,j)]:.1f}", 
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                        ha='center', va='center', fontsize=8, zorder=5)
        
        # Set title for map
        ax1.set_title("Tour Map", fontsize=12)
        ax1.set_xlim([min(p[0] for p in pos.values())-10, max(p[0] for p in pos.values())+10])
        ax1.set_ylim([min(p[1] for p in pos.values())-10, max(p[1] for p in pos.values())+10])
        ax1.set_xlabel("X Coordinate")
        ax1.set_ylabel("Y Coordinate")
        ax1.set_aspect('equal')
        
        # Information panel (right subplot)
        # -------------------------------
        ax2.axis('off')  # Turn off axes for info panel
        
        # Solution status
        status = "Optimal" if is_optimal else "Candidate"
        
        # Calculate tour validity
        valid, _ = self.validate_solution(solution)
        validity_text = "Valid Tour" if valid else "Invalid Tour"
        if not valid:
            if len(subtours) > 1:
                validity_text += f" (Subtours: {len(subtours)})"
            else:
                validity_text += " (Degree constraints violated)"
                
        # Format tour
        tour_str = "N/A"
        if len(subtours) == 1:
            # We have a valid tour, extract the order
            tour = list(subtours)[0]
            # Convert to a cycle by finding a Hamiltonian cycle
            try:
                cycle = list(nx.find_cycle(solution_graph, source=next(iter(tour))))
                tour_str = " → ".join([str(u) for u, v in cycle]) + f" → {cycle[0][0]}"
            except:
                # Fallback if cycle finding fails
                tour_str = ", ".join([str(c) for c in tour])
        else:
            # If multiple subtours, show them separately
            tour_str = " | ".join([", ".join([str(c) for c in subtour]) for subtour in subtours])
        
        # Create information table
        info_text = (
            f"Status: {status}\n"
            f"Validity: {validity_text}\n\n"
            f"Total Distance: {total_distance:.2f}\n"
            f"Number of Cities: {self.n_cities}\n"
            f"Number of Subtours: {len(subtours)}\n\n"
            f"Tour:\n{tour_str}\n"
        )
        
        # Add exploration info if provided
        if animated:
            info_text += (
                f"\nExploration Stats:\n"
                f"Step: {step}\n"
                f"Nodes Explored: {nodes_explored}\n"
                f"Time: {elapsed_time:.2f}s\n"
            )
        
        # Draw subtour legend if needed
        if len(subtours) > 1:
            legend_text = "Subtours:\n"
            for i, subtour in enumerate(subtours):
                color_idx = i % len(color_cycle)
                legend_text += f"Subtour {i+1}: {', '.join([str(c) for c in subtour])}\n"
                
            info_text += f"\n{legend_text}"
            
            # Add visual subtour legend
            legend_y = 0.4
            for i, subtour in enumerate(subtours):
                color_idx = i % len(color_cycle)
                ax2.add_patch(plt.Rectangle((0.1, legend_y - i*0.05), 0.04, 0.04, 
                                          facecolor=color_cycle[color_idx], 
                                          edgecolor='black'))
                ax2.text(0.16, legend_y - i*0.05, f"Subtour {i+1}", fontsize=10, va='center')
        
        # Add info text
        ax2.text(0.1, 0.9, info_text, fontsize=10, va='top', 
                family='monospace', linespacing=1.5,
                bbox=dict(facecolor='white', edgecolor='gray', 
                         boxstyle='round,pad=1.0', alpha=0.9))
        
        # Main title for the entire figure
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        counter = kwargs.get('counter', step if animated else self.plot_counter + 1)
        self.plot_counter = counter
        status_str = "optimal" if is_optimal else "candidate"
        filename = f"plots/{self.name}_solution_{counter:03d}_{status_str}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        return filename
    
    @classmethod
    def generate_random_instance(cls, n_cities: int = 10, 
                               min_coord: float = 0.0, 
                               max_coord: float = 100.0,
                               seed: int = None,
                               name: str = None,
                               difficulty: str = 'medium') -> 'TSP':
        """
        Generate a random TSP instance.
        
        Args:
            n_cities: Number of cities
            min_coord: Minimum coordinate value
            max_coord: Maximum coordinate value
            seed: Random seed for reproducibility
            name: Name for the instance
            difficulty: Difficulty level for the instance
            
        Returns:
            TSP: A new randomly generated TSP instance
        """
        if seed is not None:
            random.seed(seed)
            
        coordinates = {i: (random.uniform(min_coord, max_coord), 
                        random.uniform(min_coord, max_coord)) 
                      for i in range(n_cities)}
        
        instance_name = name or f"Random_TSP_{n_cities}"
        
        return cls(n_cities, coordinates, name=instance_name, difficulty=difficulty)
    
    @classmethod
    def generate_benchmark_suite(cls, difficulty_levels: List[str] = None) -> Dict[str, List['TSP']]:
        """
        Generate a suite of benchmark TSP instances at different difficulty levels.
        
        Args:
            difficulty_levels: List of difficulty levels to generate
                             (defaults to ['easy', 'medium', 'hard'])
            
        Returns:
            Dict[str, List[TSP]]: Dictionary mapping difficulty levels to lists of TSP instances
        """
        if difficulty_levels is None:
            difficulty_levels = ['easy', 'medium', 'hard']
            
        suite = {}
        
        # Define instance sizes for each difficulty level
        sizes = {
            'easy': [5, 7, 10],
            'medium': [15, 20, 25],
            'hard': [30, 40, 50]
        }
        
        # Generate instances for each difficulty level
        for level in difficulty_levels:
            if level not in sizes:
                continue
                
            suite[level] = []
            for size in sizes[level]:
                for i in range(3):  # Generate 3 instances of each size
                    name = f"{level.capitalize()}_TSP_{size}_{i+1}"
                    instance = cls.generate_random_instance(
                        n_cities=size,
                        seed=hash(f"{level}_{size}_{i}") % 10000,
                        name=name,
                        difficulty=level
                    )
                    suite[level].append(instance)
        
        return suite
    
    def find_subtours(self, solution: np.ndarray) -> List[Set[int]]:
        """
        Find all subtours in the current solution.
        
        Args:
            solution: Binary solution vector
            
        Returns:
            List of sets, where each set contains the cities in a subtour
        """
        # Create a graph from the solution
        G = nx.Graph()
        for i in range(self.n_cities):
            G.add_node(i)
        
        # Use a tolerance for binary values
        EDGE_TOLERANCE = 1e-4
        
        # Add edges from solution
        for i, j in combinations(range(self.n_cities), 2):
            idx = self._get_variable_index(i, j)
            if solution[idx] > 1 - EDGE_TOLERANCE:
                G.add_edge(i, j)
        
        # Find connected components (subtours)
        subtours = list(nx.connected_components(G))
        
        return subtours
    
    def generate_subtour_constraint(self, subtour: Set[int]) -> Tuple[np.ndarray, float]:
        """
        Generate subtour elimination constraint for a given subtour.
        
        For a subtour S, the constraint is:
        sum(x[i,j] for i,j in S) <= |S| - 1
        
        Args:
            subtour: Set of cities forming a subtour
            
        Returns:
            Tuple of (constraint_coefficients, rhs)
        """
        n = self.n_cities
        constraint = np.zeros(n * (n - 1) // 2)
        
        # For each pair of cities in the subtour
        for i, j in combinations(subtour, 2):
            idx = self._get_variable_index(min(i, j), max(i, j))
            constraint[idx] = 1
        
        # RHS: |S| - 1 ensures we can't have a complete subtour
        rhs = len(subtour) - 1
        
        return constraint, rhs
    
    def get_constraint_generator(self) -> Optional[Callable]:
        """
        Return a function that generates subtour elimination constraints.
        
        Returns:
            Callable: A function that takes a solution vector and returns 
                     a list of (constraint, rhs) tuples for subtour elimination
        """
        def subtour_constraint_generator(solution: np.ndarray) -> List[Tuple[np.ndarray, float]]:
            subtours = self.find_subtours(solution)
            
            # Only add constraints if we have multiple subtours
            if len(subtours) <= 1:
                return []
            
            # Generate a constraint for each subtour
            constraints = []
            for subtour in subtours:
                constraint, rhs = self.generate_subtour_constraint(subtour)
                constraints.append((constraint, rhs))
            
            return constraints
        
        return subtour_constraint_generator
    
    def _get_variable_index(self, i: int, j: int) -> int:
        """
        Get the index of the decision variable for edge (i,j) in the ILP formulation.
    
        For a TSP with n cities, we create n*(n-1)/2 binary variables, one for each
        possible undirected edge. This method converts a city pair (i,j) to the
        corresponding variable index in our flattened representation.
    
        Args:
            i: First city index
            j: Second city index
        
        Returns:
            int: Index in the flattened variable array
        """
        # Ensure i < j for consistent indexing
        if i > j:
            i, j = j, i
        
        # Calculate index using combinatorial formula
        return i * (self.n_cities - 1) - i * (i + 1) // 2 + j - 1
    
    @property
    def name(self) -> str:
        """Get the name of this TSP instance."""
        return self._name
    
    @property
    def size(self) -> Dict[str, int]:
        """Get size metrics for this TSP instance."""
        n = self.n_cities
        return {
            'cities': n,
            'variables': n * (n - 1) // 2,
            'constraints': n  # Degree constraints (initial)
        }
    
    @property
    def difficulty(self) -> str:
        """Get the difficulty level of this TSP instance."""
        return self._difficulty


def create_predefined_instances() -> Dict[str, TSP]:
    """
    Create predefined TSP instances for examples and testing.
    
    Returns:
        Dict[str, TSP]: Dictionary of named TSP instances
    """
    instances = {}
    
    # Example 1: 3 cities in a triangle
    coords_3 = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0)
    }
    instances["triangle"] = TSP(3, coords_3, name="Triangle_TSP", difficulty="easy")
    
    # Example 2: 4 cities in a square
    coords_4 = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 1),
        3: (1, 0)
    }
    instances["square"] = TSP(4, coords_4, name="Square_TSP", difficulty="easy")
    
    # Example 3: 5 cities in a star pattern
    coords_5 = {
        0: (0, 0),    # center
        1: (1, 1),    # top right
        2: (-1, 1),   # top left
        3: (-1, -1),  # bottom left
        4: (1, -1)    # bottom right
    }
    instances["star"] = TSP(5, coords_5, name="Star_TSP", difficulty="easy")
    
    return instances