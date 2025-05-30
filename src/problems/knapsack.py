"""
Knapsack Problem implementation.

This module provides a Knapsack Problem class that implements the OptimizationProblem interface,
with utilities for creating, visualizing, and solving Knapsack instances.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

from .base import OptimizationProblem


class Knapsack(OptimizationProblem):
    """
    Knapsack Problem implementation.
    
    In the Knapsack problem, we have a set of items, each with a weight and a value.
    We need to select a subset of items to maximize the total value while keeping
    the total weight below a given capacity.
    
    Attributes:
        n_items (int): Number of items
        values (np.ndarray): Array of item values
        weights (np.ndarray): Array of item weights
        capacity (float): Capacity of the knapsack
        problem_name (str): Name of this Knapsack instance
        problem_difficulty (str): Difficulty level of this instance
    """
    
    def __init__(self, values: np.ndarray, weights: np.ndarray, capacity: float,
                 name: str = None, difficulty: str = 'medium'):
        """
        Initialize a Knapsack Problem instance.
        
        Args:
            values: Array of item values
            weights: Array of item weights
            capacity: Capacity of the knapsack
            name: Name for this problem instance
            difficulty: Difficulty level for this instance
        """
        self.values = np.array(values)
        self.weights = np.array(weights)
        self.capacity = float(capacity)
        self.n_items = len(values)
        
        # Validate input
        if len(values) != len(weights):
            raise ValueError("Values and weights must have the same length")
        if any(v < 0 for v in values) or any(w <= 0 for w in weights):
            raise ValueError("Values must be non-negative and weights must be positive")
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
            
        self._name = name or f"Knapsack_{self.n_items}"
        self._difficulty = difficulty
        
        # Create a directory for plots if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
    def to_ilp(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert Knapsack Problem to ILP formulation.
        
        Returns:
            Tuple containing:
            - c: Objective coefficients (values to maximize)
            - A_eq: Equality constraint matrix (empty)
            - b_eq: Equality constraint right-hand side (empty)
            - A_ub: Inequality constraint matrix (capacity constraint)
            - b_ub: Inequality constraint right-hand side (capacity)
        """
        n = self.n_items
        
        # Objective: maximize sum of values for selected items
        c = self.values.copy()
        
        # Capacity constraint: sum of weights <= capacity
        A_ub = self.weights.reshape(1, -1)  # Single row for the capacity constraint
        b_ub = np.array([self.capacity])
        
        # No equality constraints
        A_eq = np.zeros((0, n))
        b_eq = np.zeros(0)
        
        return c, A_eq, b_eq, A_ub, b_ub
    
    def validate_solution(self, solution: np.ndarray) -> Tuple[bool, float]:
        """
        Check if a solution is valid and compute its objective value.
        
        A valid solution must:
        1. Be binary (0 or 1 values)
        2. Respect the capacity constraint
        
        Args:
            solution: Binary solution vector indicating which items are selected
            
        Returns:
            Tuple containing:
            - is_valid: Boolean indicating if the solution is valid
            - objective_value: The objective value (total value of selected items)
        """
        # Check binary values
        TOLERANCE = 1e-4
        if not all((x < TOLERANCE) or (abs(x - 1) < TOLERANCE) for x in solution):
            return False, 0.0
        
        # Check capacity constraint
        total_weight = np.sum(solution * self.weights)
        if total_weight > self.capacity + TOLERANCE:
            return False, 0.0
        
        # Calculate objective value
        total_value = np.sum(solution * self.values)
        
        return True, total_value
    
    def visualize_instance(self, title: str = None, figsize: Tuple[int, int] = (10, 6), **kwargs) -> str:
        """
        Visualize the Knapsack Problem instance.
        
        Creates a plot showing the items' weights vs values.
        
        Args:
            title: Optional title for the visualization
            figsize: Figure size as (width, height)
            **kwargs: Additional visualization parameters
            
        Returns:
            str: Path to the saved visualization file
        """
        plt.figure(figsize=figsize)
        
        # Calculate marker sizes based on weight % of capacity
        max_size = 500
        sizes = (self.weights / self.capacity) * max_size
        
        # Create scatter plot of values vs weights
        plt.scatter(self.weights, self.values, s=sizes, alpha=0.6, 
                   c=np.arange(self.n_items), cmap='viridis')
        
        # Add labels
        for i in range(self.n_items):
            plt.annotate(f"Item {i}", 
                        (self.weights[i], self.values[i]),
                        textcoords="offset points", 
                        xytext=(0, 5), 
                        ha='center')
        
        # Add reference line for value/weight ratio
        value_weight_ratio = np.array([self.values[i] / self.weights[i] 
                                     for i in range(self.n_items)])
        max_ratio = np.max(value_weight_ratio)
        x_range = np.array([0, np.max(self.weights) * 1.1])
        plt.plot(x_range, max_ratio * x_range, 'r--', alpha=0.3, 
                label=f"Max value/weight: {max_ratio:.2f}")
        
        # Add capacity line
        plt.axvline(x=self.capacity, color='red', linestyle='-', alpha=0.5,
                   label=f"Capacity: {self.capacity}")
        
        # Set labels and title
        plt.xlabel("Weight")
        plt.ylabel("Value")
        plt_title = title or f"Knapsack Instance: {self.name} (Capacity: {self.capacity})"
        plt.title(plt_title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        filename = f"plots/knapsack_instance_{self.name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        return filename
   
    def visualize_solution(self, solution: np.ndarray, is_optimal: bool = False, 
                         figsize: Tuple[int, int] = (14, 8), **kwargs) -> str:
        """
        Visualize a solution to the Knapsack Problem with a backpack representation.
        
        Args:
            solution: Binary solution vector indicating which items are selected
            is_optimal: Whether the solution is optimal
            figsize: Figure size as (width, height)
            **kwargs: Additional visualization parameters
            
        Returns:
            str: Path to the saved visualization file
        """
        # Enhanced figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1.5, 1]})
        
        # Get additional information from kwargs
        step = kwargs.get('step', 0)
        nodes_explored = kwargs.get('nodes_explored', 0)
        elapsed_time = kwargs.get('elapsed_time', 0)
        best_obj_value = kwargs.get('best_obj_value', 0)
        title = kwargs.get('title', f"Knapsack Solution - {self.name}")
        animated = kwargs.get('animated', False)
        
        # Determine selected items
        TOLERANCE = 1e-4
        selected = [i for i, x in enumerate(solution) if abs(x - 1) < TOLERANCE]
        not_selected = [i for i, x in enumerate(solution) if x < TOLERANCE]
        
        # Calculate total value and weight
        total_value = sum(self.values[i] for i in selected)
        total_weight = sum(self.weights[i] for i in selected)
        
        # Part 1: Traditional scatter plot (left subplot)
        # -----------------------------------------------
        # Calculate marker sizes based on weight % of capacity
        max_size = 500
        sizes = (self.weights / self.capacity) * max_size
        
        # Plot items
        ax1.scatter([self.weights[i] for i in not_selected], 
                   [self.values[i] for i in not_selected], 
                   s=[sizes[i] for i in not_selected], 
                   alpha=0.3, c='gray', label='Not Selected')
        
        ax1.scatter([self.weights[i] for i in selected], 
                   [self.values[i] for i in selected], 
                   s=[sizes[i] for i in selected], 
                   alpha=0.8, c='green', label='Selected')
        
        # Add labels
        for i in range(self.n_items):
            ax1.annotate(f"Item {i}", 
                        (self.weights[i], self.values[i]),
                        textcoords="offset points", 
                        xytext=(0, 5), 
                        ha='center',
                        fontsize=8)
        
        # Add capacity line
        ax1.axvline(x=self.capacity, color='red', linestyle='-', alpha=0.5,
                   label=f"Capacity: {self.capacity}")
        
        # Add fill to show used capacity
        ax1.axvspan(0, total_weight, alpha=0.1, color='green', 
                   label=f"Used: {total_weight:.1f} ({total_weight/self.capacity*100:.1f}%)")
        
        # Set labels and title
        ax1.set_xlabel("Weight")
        ax1.set_ylabel("Value")
        ax1.set_title("Item Distribution (Value vs Weight)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        
        # Part 2: Backpack visualization (right subplot)
        # ----------------------------------------------
        # Draw backpack outline
        backpack_height = 10
        backpack_width = 6
        backpack_color = 'navy'
        backpack_alpha = 0.2
        
        # Calculate capacity fill level
        fill_level = (total_weight / self.capacity) * backpack_height
        fill_level = min(fill_level, backpack_height)  # Cap at maximum
        
        # Draw backpack
        ax2.add_patch(plt.Rectangle((2, 0), backpack_width, backpack_height, 
                                   facecolor=backpack_color, alpha=backpack_alpha, 
                                   edgecolor='black', linewidth=2))
        
        # Draw fill level
        ax2.add_patch(plt.Rectangle((2, 0), backpack_width, fill_level, 
                                   facecolor='green', alpha=0.3, 
                                   edgecolor=None))
        
        # Draw straps
        strap_width = 0.5
        ax2.add_patch(plt.Rectangle((2, backpack_height), strap_width, 2, 
                                   facecolor=backpack_color, alpha=backpack_alpha, 
                                   edgecolor='black', linewidth=1))
        ax2.add_patch(plt.Rectangle((2 + backpack_width - strap_width, backpack_height), strap_width, 2, 
                                   facecolor=backpack_color, alpha=backpack_alpha, 
                                   edgecolor='black', linewidth=1))
        
        # Arrange items inside the backpack
        remaining_items = selected.copy()
        placed_items = []
        y_position = 0.5  # Start from bottom of backpack
        x_position = 2.5  # Start from left side of backpack
        
        # Sort selected items by weight (largest first)
        sorted_items = sorted(selected, key=lambda i: self.weights[i], reverse=True)
        
        for i in sorted_items:
            item_height = (self.weights[i] / self.capacity) * backpack_height * 0.8
            item_width = min(1.5, (self.values[i] / max(self.values)) * 3)
            
            # If we reach the top, start a new column
            if y_position + item_height > fill_level:
                y_position = 0.5
                x_position += 2
                
                # If we've run out of horizontal space, stop placing items
                if x_position > 2 + backpack_width - item_width:
                    break
            
            # Draw item
            item_color = plt.cm.viridis(i / self.n_items)  # Use colormap to make items distinct
            ax2.add_patch(plt.Rectangle((x_position, y_position), item_width, item_height, 
                                       facecolor=item_color, alpha=0.8, 
                                       edgecolor='black', linewidth=1))
            
            # Add item label
            ax2.text(x_position + item_width/2, y_position + item_height/2, f"{i}", 
                    ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Add to placed items and update position
            placed_items.append(i)
            y_position += item_height + 0.2  # Add a small gap
        
        # If some items couldn't be visualized, show a message
        if len(placed_items) < len(selected):
            ax2.text(5, backpack_height + 1, f"+{len(selected) - len(placed_items)} more items", 
                    ha='center', va='center', fontsize=8)
            
        # Set axis limits and remove ticks
        ax2.set_xlim(0, 12)
        ax2.set_ylim(0, backpack_height + 3)
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Add capacity gauge
        capacity_width = 0.5
        ax2.add_patch(plt.Rectangle((1, 0), capacity_width, backpack_height, 
                                   facecolor='lightgray', alpha=0.5, 
                                   edgecolor='black', linewidth=1))
        ax2.add_patch(plt.Rectangle((1, 0), capacity_width, fill_level, 
                                   facecolor='green', alpha=0.5, 
                                   edgecolor='black', linewidth=1))
        
        # Add labels for capacity gauge
        ax2.text(1 + capacity_width/2, -0.5, "0", ha='center', va='center', fontsize=8)
        ax2.text(1 + capacity_width/2, backpack_height + 0.5, f"{self.capacity:.1f}", ha='center', va='center', fontsize=8)
        ax2.text(1 + capacity_width/2, fill_level, f"{total_weight:.1f}", ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Add backpack title
        status = "Optimal" if is_optimal else "Current Best"
        ax2.set_title(f"Backpack Contents ({status})")
        
        # Add info box with solution details
        info_text = (
            f"Total Value: {total_value:.1f}\n"
            f"Total Weight: {total_weight:.1f}/{self.capacity:.1f}\n"
            f"Capacity Used: {total_weight/self.capacity*100:.1f}%\n"
            f"Items Selected: {len(selected)}/{self.n_items}\n"
        )
        
        # Add exploration info if provided
        if animated:
            info_text += (
                f"\nStep: {step}\n"
                f"Nodes Explored: {nodes_explored}\n"
                f"Time: {elapsed_time:.2f}s"
            )
            
        # Add a text box for the info
        ax2.text(6, backpack_height/2, info_text, 
                ha='left', va='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        # Main title for the entire figure
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        counter = kwargs.get('counter', step if animated else 1)
        status_str = "optimal" if is_optimal else "candidate"
        filename = f"plots/{self.name}_solution_{counter:03d}_{status_str}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        return filename
    
    @classmethod
    def generate_random_instance(cls, n_items: int = 10, 
                               min_value: float = 1.0, 
                               max_value: float = 100.0,
                               min_weight: float = 1.0,
                               max_weight: float = 100.0,
                               capacity_factor: float = 0.5,
                               seed: int = None,
                               name: str = None,
                               difficulty: str = 'medium') -> 'Knapsack':
        """
        Generate a random Knapsack instance.
        
        Args:
            n_items: Number of items
            min_value: Minimum value for items
            max_value: Maximum value for items
            min_weight: Minimum weight for items
            max_weight: Maximum weight for items
            capacity_factor: Capacity as a fraction of total weight (0.0-1.0)
            seed: Random seed for reproducibility
            name: Name for the instance
            difficulty: Difficulty level for the instance
            
        Returns:
            Knapsack: A new randomly generated Knapsack instance
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Generate random values and weights
        values = np.random.uniform(min_value, max_value, n_items)
        weights = np.random.uniform(min_weight, max_weight, n_items)
        
        # Set capacity as a fraction of total weight
        total_weight = np.sum(weights)
        capacity = total_weight * capacity_factor
        
        instance_name = name or f"Random_Knapsack_{n_items}"
        
        return cls(values, weights, capacity, name=instance_name, difficulty=difficulty)
    
    @classmethod
    def generate_benchmark_suite(cls, difficulty_levels: List[str] = None) -> Dict[str, List['Knapsack']]:
        """
        Generate a suite of benchmark Knapsack instances at different difficulty levels.
        
        Args:
            difficulty_levels: List of difficulty levels to generate
                             (defaults to ['easy', 'medium', 'hard'])
            
        Returns:
            Dict[str, List[Knapsack]]: Dictionary mapping difficulty levels to lists of Knapsack instances
        """
        if difficulty_levels is None:
            difficulty_levels = ['easy', 'medium', 'hard']
            
        suite = {}
        
        # Define instance parameters for each difficulty level
        configs = {
            'easy': {
                'n_items': [10, 15, 20],
                'capacity_factor': 0.5,
                'correlated': False
            },
            'medium': {
                'n_items': [30, 50, 75],
                'capacity_factor': 0.4,
                'correlated': True
            },
            'hard': {
                'n_items': [100, 150, 200],
                'capacity_factor': 0.3,
                'correlated': True
            }
        }
        
        # Generate instances for each difficulty level
        for level in difficulty_levels:
            if level not in configs:
                continue
                
            suite[level] = []
            config = configs[level]
            
            for size in config['n_items']:
                for i in range(3):  # Generate 3 instances of each size
                    # Use deterministic seed for reproducibility
                    seed = hash(f"{level}_{size}_{i}") % 10000
                    np.random.seed(seed)
                    random.seed(seed)
                    
                    name = f"{level.capitalize()}_Knapsack_{size}_{i+1}"
                    
                    # For correlated problems, make values proportional to weights with some noise
                    if config['correlated']:
                        weights = np.random.uniform(1.0, 100.0, size)
                        # Values correlated with weights, plus random noise
                        noise = np.random.uniform(0.8, 1.2, size)
                        values = weights * noise
                    else:
                        weights = np.random.uniform(1.0, 100.0, size)
                        values = np.random.uniform(1.0, 100.0, size)
                    
                    # Set capacity
                    capacity = sum(weights) * config['capacity_factor']
                    
                    instance = cls(values, weights, capacity, name=name, difficulty=level)
                    suite[level].append(instance)
        
        return suite
    
    def get_constraint_generator(self) -> Optional[Callable]:
        """
        Return a function that generates additional constraints during branch-and-bound.
        
        For Knapsack, we don't need lazy constraints like TSP, so return None.
        
        Returns:
            None: Knapsack doesn't require lazy constraint generation
        """
        return None
    
    @property
    def name(self) -> str:
        """Get the name of this Knapsack instance."""
        return self._name
    
    @property
    def size(self) -> Dict[str, int]:
        """Get size metrics for this Knapsack instance."""
        return {
            'items': self.n_items,
            'variables': self.n_items,
            'constraints': 1  # Just the capacity constraint
        }
    
    def size_metrics(self) -> Dict[str, int]:
        """
        Get detailed size metrics for benchmarking purposes.
        
        For Knapsack, this includes:
        - Number of items
        - Number of variables in the ILP formulation
        - Number of constraints in the ILP formulation
        - Capacity value
        
        Returns:
            Dict[str, int]: Dictionary of size metrics
        """
        return {
            'items': self.n_items,
            'variables': self.n_items,  # One binary variable per item
            'constraints': 1,  # Just the capacity constraint
            'capacity': int(self.capacity),  # Integer version of capacity for metrics
            'total_weight': int(np.sum(self.weights)),  # Sum of all weights
            'total_value': int(np.sum(self.values))  # Sum of all values
        }
    
    @property
    def difficulty(self) -> str:
        """Get the difficulty level of this Knapsack instance."""
        return self._difficulty
        
    def get_specific_metrics(self) -> Dict[str, Any]:
        """
        Get problem-specific metrics for the Knapsack instance.
        
        These metrics capture structural properties that may correlate with difficulty:
        - value_weight_correlation: Correlation between weights and values
        - capacity_utilization_ratio: Capacity as a fraction of total weight
        - value_density_stats: Statistics about value-to-weight ratios
        - coefficient_of_variation: Measures dispersion in value/weight ratios
        
        Returns:
            Dict[str, Any]: Dictionary of problem-specific metrics
        """
        # Calculate value-to-weight ratios
        value_weight_ratios = self.values / self.weights
        
        # Calculate correlation between weights and values
        value_weight_correlation = np.corrcoef(self.weights, self.values)[0, 1]
        
        # Calculate capacity utilization ratio
        capacity_ratio = self.capacity / np.sum(self.weights)
        
        # Calculate stats about value densities
        vw_mean = np.mean(value_weight_ratios)
        vw_median = np.median(value_weight_ratios)
        vw_min = np.min(value_weight_ratios)
        vw_max = np.max(value_weight_ratios)
        vw_std = np.std(value_weight_ratios)
        
        # Coefficient of variation (measure of dispersion)
        cv = vw_std / vw_mean if vw_mean > 0 else 0
        
        # Add is_minimization flag (knapsack is a maximization problem)
        return {
            'is_minimization': False,  # Knapsack is a maximization problem
            'value_weight_correlation': float(value_weight_correlation),
            'capacity_utilization_ratio': float(capacity_ratio),
            'value_density_mean': float(vw_mean),
            'value_density_median': float(vw_median),
            'value_density_min': float(vw_min),
            'value_density_max': float(vw_max),
            'value_density_std': float(vw_std),
            'coefficient_of_variation': float(cv)
        }


def create_predefined_instances() -> Dict[str, Knapsack]:
    """
    Create predefined Knapsack instances for examples and testing.
    
    Returns:
        Dict[str, Knapsack]: Dictionary of named Knapsack instances
    """
    instances = {}
    
    # Example 1: Simple 5-item knapsack
    values = np.array([10, 30, 20, 50, 60])
    weights = np.array([5, 10, 15, 22, 25])
    capacity = 40
    instances["simple"] = Knapsack(values, weights, capacity, 
                                  name="Simple_Knapsack", difficulty="easy")
    
    # Example 2: Medium difficulty with 10 items
    values = np.array([55, 10, 47, 5, 4, 50, 8, 61, 85, 87])
    weights = np.array([95, 4, 60, 32, 23, 72, 80, 62, 65, 46])
    capacity = 269  # About half the total weight
    instances["medium"] = Knapsack(values, weights, capacity, 
                                  name="Medium_Knapsack", difficulty="medium")
    
    # Example 3: Correlated weights and values (harder)
    weights = np.array([8, 12, 9, 14, 16, 10, 6, 7, 11, 13])
    # Values are correlated with weights, with some noise
    base_values = weights * 5
    noise = np.array([1.2, 0.9, 1.1, 0.8, 1.0, 1.1, 0.9, 1.2, 0.8, 1.0])
    values = base_values * noise
    capacity = 40
    instances["correlated"] = Knapsack(values, weights, capacity, 
                                      name="Correlated_Knapsack", difficulty="hard")
    
    return instances