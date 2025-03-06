"""
Bin Packing Problem implementation.

This module provides a Bin Packing Problem class that implements the OptimizationProblem interface,
with utilities for creating, visualizing, and solving Bin Packing instances.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os
from typing import Dict, List, Tuple, Set, Optional, Any, Union, Callable
from matplotlib.patches import Rectangle

from .base import OptimizationProblem


class BinPacking(OptimizationProblem):
    """
    Bin Packing Problem implementation.
    
    In the Bin Packing problem, we have a set of items with different sizes and 
    a set of bins with fixed capacity. The goal is to assign each item to a bin 
    such that the total size of items in each bin does not exceed the bin capacity, 
    while minimizing the number of bins used.
    
    Attributes:
        n_items (int): Number of items
        bin_capacity (float): Capacity of each bin
        item_sizes (np.ndarray): Sizes of the items
        max_bins (int): Maximum number of bins that might be needed
        problem_name (str): Name of this Bin Packing instance
        problem_difficulty (str): Difficulty level of this instance
    """
    
    def __init__(self, item_sizes: np.ndarray, bin_capacity: float, 
                 max_bins: int = None, name: str = None, difficulty: str = 'medium'):
        """
        Initialize a Bin Packing Problem instance.
        
        Args:
            item_sizes: Array of item sizes
            bin_capacity: Capacity of each bin
            max_bins: Maximum number of bins (defaults to number of items)
            name: Name for this problem instance
            difficulty: Difficulty level for this instance
        """
        self.item_sizes = np.array(item_sizes)
        self.bin_capacity = float(bin_capacity)
        self.n_items = len(item_sizes)
        
        # Validate input
        if any(size <= 0 for size in item_sizes):
            raise ValueError("Item sizes must be positive")
        if bin_capacity <= 0:
            raise ValueError("Bin capacity must be positive")
        if any(size > bin_capacity for size in item_sizes):
            raise ValueError("Item sizes cannot exceed bin capacity")
            
        # Set maximum number of bins (defaults to number of items)
        self.max_bins = max_bins if max_bins is not None else self.n_items
        
        # Set name and difficulty
        self._name = name or f"BinPacking_{self.n_items}"
        self._difficulty = difficulty
        
        # Create a directory for plots if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
    def to_ilp(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert Bin Packing Problem to ILP formulation.
        
        The ILP formulation uses two types of binary variables:
        1. x_ij = 1 if item i is placed in bin j, 0 otherwise
        2. y_j = 1 if bin j is used, 0 otherwise
        
        With constraints:
        - Each item must be assigned to exactly one bin
        - The total size of items in each bin cannot exceed the bin's capacity
        - A bin can only be used if y_j = 1
        
        Returns:
            Tuple containing:
            - c: Objective coefficients (minimize number of bins used)
            - A_eq: Equality constraint matrix (each item in exactly one bin)
            - b_eq: Equality constraint right-hand side (all 1's)
            - A_ub: Inequality constraint matrix (bin capacity constraints)
            - b_ub: Inequality constraint right-hand side (bin capacities)
        """
        n_items = self.n_items
        max_bins = self.max_bins
        
        # Number of variables: n_items * max_bins (x_ij) + max_bins (y_j)
        num_vars = n_items * max_bins + max_bins
        
        # Objective: minimize the sum of y_j (number of bins used)
        # The first n_items*max_bins variables are x_ij, the last max_bins are y_j
        c = np.zeros(num_vars)
        c[n_items * max_bins:] = -1  # Negative because we maximize -y_j
        
        # Equality constraints: each item is assigned to exactly one bin
        # sum_j x_ij = 1 for all i
        A_eq = np.zeros((n_items, num_vars))
        for i in range(n_items):
            for j in range(max_bins):
                A_eq[i, i * max_bins + j] = 1
        
        # RHS for equality constraints: all 1's
        b_eq = np.ones(n_items)
        
        # Inequality constraints:
        # 1. Bin capacity constraints: sum_i s_i * x_ij <= C * y_j for all j
        # 2. Link between x_ij and y_j: x_ij <= y_j for all i,j
        
        # For capacity constraints, we rearrange to: sum_i s_i * x_ij - C * y_j <= 0
        # For link constraints: x_ij - y_j <= 0
        # Total: max_bins capacity constraints + n_items*max_bins link constraints
        n_ineq = max_bins + n_items * max_bins
        A_ub = np.zeros((n_ineq, num_vars))
        b_ub = np.zeros(n_ineq)
        
        # Add capacity constraints
        for j in range(max_bins):
            for i in range(n_items):
                # Coefficient for x_ij in the capacity constraint
                A_ub[j, i * max_bins + j] = self.item_sizes[i]
            # Coefficient for y_j in the capacity constraint
            A_ub[j, n_items * max_bins + j] = -self.bin_capacity
        
        # Add link constraints
        idx = max_bins
        for i in range(n_items):
            for j in range(max_bins):
                # Coefficient for x_ij in the link constraint
                A_ub[idx, i * max_bins + j] = 1
                # Coefficient for y_j in the link constraint
                A_ub[idx, n_items * max_bins + j] = -1
                idx += 1
        
        return c, A_eq, b_eq, A_ub, b_ub
    
    def validate_solution(self, solution: np.ndarray) -> Tuple[bool, float]:
        """
        Check if a solution is valid and compute its objective value.
        
        A valid bin packing solution must:
        1. Assign each item to exactly one bin
        2. Not exceed the capacity of any bin
        3. Have consistent y_j values (bin used if and only if there's an item in it)
        
        Args:
            solution: Binary solution vector (x_ij and y_j variables)
            
        Returns:
            Tuple containing:
            - is_valid: Boolean indicating if the solution is valid
            - objective_value: The objective value (number of bins used)
        """
        n_items = self.n_items
        max_bins = self.max_bins
        
        # Extract x_ij and y_j variables
        x_vars = solution[:n_items * max_bins].reshape(n_items, max_bins)
        y_vars = solution[n_items * max_bins:]
        
        # Check binary values
        TOLERANCE = 1e-4
        if not all((x < TOLERANCE) or (abs(x - 1) < TOLERANCE) for x in solution):
            return False, 0.0
        
        # Check that each item is assigned to exactly one bin
        for i in range(n_items):
            if abs(np.sum(x_vars[i, :]) - 1.0) > TOLERANCE:
                return False, 0.0
        
        # Check bin capacity constraints
        for j in range(max_bins):
            # Calculate total size of items in bin j
            bin_load = sum(self.item_sizes[i] * x_vars[i, j] for i in range(n_items))
            # Check if bin is used
            bin_used = y_vars[j] > 0.5
            
            # If bin is used, check capacity
            if bin_used and bin_load > self.bin_capacity + TOLERANCE:
                return False, 0.0
            
            # Check consistency: bin is used if and only if there's an item in it
            has_items = bin_load > TOLERANCE
            if has_items != bin_used:
                return False, 0.0
        
        # Calculate objective value: number of bins used
        objective_value = sum(y_vars)
        
        return True, objective_value
    
    def visualize_instance(self, title: str = None, figsize: Tuple[int, int] = (10, 6), **kwargs) -> str:
        """
        Visualize the Bin Packing Problem instance.
        
        Creates a chart showing the items and their sizes.
        
        Args:
            title: Optional title for the visualization
            figsize: Figure size as (width, height)
            **kwargs: Additional visualization parameters
            
        Returns:
            str: Path to the saved visualization file
        """
        plt.figure(figsize=figsize)
        
        # Create bar chart of item sizes
        plt.bar(range(self.n_items), self.item_sizes, color='skyblue', edgecolor='black')
        
        # Add horizontal line for bin capacity
        plt.axhline(y=self.bin_capacity, color='red', linestyle='--', 
                   label=f"Bin capacity = {self.bin_capacity}")
        
        # Add minimal bin count estimate (sum of sizes / capacity, rounded up)
        min_bins = int(np.ceil(np.sum(self.item_sizes) / self.bin_capacity))
        
        # Add labels and title
        plt.xlabel("Item")
        plt.ylabel("Size")
        plt_title = title or f"Bin Packing Instance: {self.name} " + \
                           f"(Min bins needed: ~{min_bins}, Max bins: {self.max_bins})"
        plt.title(plt_title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(range(self.n_items))
        
        # Save plot
        filename = f"plots/bin_packing_instance_{self.name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        return filename
   
    def visualize_solution(self, solution: np.ndarray, is_optimal: bool = False, 
                         figsize: Tuple[int, int] = (12, 8), **kwargs) -> str:
        """
        Visualize a solution to the Bin Packing Problem.
        
        Shows how items are packed into bins.
        
        Args:
            solution: Binary solution vector
            is_optimal: Whether the solution is optimal
            figsize: Figure size as (width, height)
            **kwargs: Additional visualization parameters
            
        Returns:
            str: Path to the saved visualization file
        """
        plt.figure(figsize=figsize)
        
        # Extract x_ij and y_j variables
        n_items = self.n_items
        max_bins = self.max_bins
        x_vars = solution[:n_items * max_bins].reshape(n_items, max_bins)
        y_vars = solution[n_items * max_bins:]
        
        # Count used bins
        used_bins = int(np.sum(y_vars > 0.5))
        
        # Create a subplot for each used bin
        fig, axes = plt.subplots(1, used_bins, figsize=figsize, 
                                 squeeze=False, constrained_layout=True)
        
        # Set up item colors
        colors = plt.cm.tab20(np.linspace(0, 1, self.n_items))
        
        bin_idx = 0
        for j in range(max_bins):
            if y_vars[j] < 0.5:  # Skip unused bins
                continue
            
            ax = axes[0, bin_idx]
            
            # Get items in this bin
            bin_items = [i for i in range(n_items) if x_vars[i, j] > 0.5]
            
            # Sort items by size for better visualization (largest at bottom)
            bin_items.sort(key=lambda i: self.item_sizes[i], reverse=True)
            
            # Position for the bottom of the next item
            bottom = 0
            
            # Place items in the bin
            for i in bin_items:
                item_height = self.item_sizes[i]
                
                # Draw item as a rectangle
                rect = Rectangle((0.1, bottom), 0.8, item_height, 
                                 facecolor=colors[i], alpha=0.8, edgecolor='black')
                ax.add_patch(rect)
                
                # Add item label in the center of the rectangle
                ax.text(0.5, bottom + item_height / 2, f"Item {i}\n({item_height:.1f})", 
                       ha='center', va='center', fontsize=8)
                
                # Update bottom position for next item
                bottom += item_height
            
            # Set up axis limits and labels
            ax.set_xlim(0, 1)
            ax.set_ylim(0, self.bin_capacity * 1.05)
            ax.set_title(f"Bin {j + 1}")
            ax.set_xticks([])
            ax.set_ylabel("Size")
            
            # Add capacity line
            ax.axhline(y=self.bin_capacity, color='red', linestyle='--')
            
            # Add utilization information
            bin_load = sum(self.item_sizes[i] for i in bin_items)
            utilization = bin_load / self.bin_capacity * 100
            ax.text(0.5, self.bin_capacity * 1.02, 
                   f"Utilization: {utilization:.1f}%", 
                   ha='center', va='bottom', fontsize=8)
            
            bin_idx += 1
        
        # Add overall title
        status = "Optimal" if is_optimal else "Candidate"
        bins_used = np.sum(y_vars > 0.5)
        plt.suptitle(f"{status} Solution - Bins Used: {bins_used}/{self.max_bins}")
        
        # Save plot
        counter = kwargs.get('counter', 1)
        filename = f"plots/{self.name}_solution_{counter:03d}_{status.lower()}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        return filename
    
    @classmethod
    def generate_random_instance(cls, n_items: int = 20, 
                               bin_capacity: float = 100.0,
                               min_size: float = 10.0,
                               max_size: float = 50.0,
                               size_distribution: str = 'uniform',
                               max_bins: int = None,
                               seed: int = None,
                               name: str = None,
                               difficulty: str = 'medium') -> 'BinPacking':
        """
        Generate a random Bin Packing instance.
        
        Args:
            n_items: Number of items
            bin_capacity: Capacity of each bin
            min_size: Minimum item size
            max_size: Maximum item size
            size_distribution: Distribution of sizes ('uniform', 'normal', 'exp')
            max_bins: Maximum number of bins to use (defaults to n_items)
            seed: Random seed for reproducibility
            name: Name for the instance
            difficulty: Difficulty level
            
        Returns:
            BinPacking: A new randomly generated Bin Packing instance
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Generate item sizes based on specified distribution
        if size_distribution == 'uniform':
            item_sizes = np.random.uniform(min_size, max_size, n_items)
        elif size_distribution == 'normal':
            mean_size = (min_size + max_size) / 2
            std_size = (max_size - min_size) / 6  # ~99.7% of values within range
            item_sizes = np.random.normal(mean_size, std_size, n_items)
            item_sizes = np.clip(item_sizes, min_size, max_size)
        elif size_distribution == 'exp':
            # Exponential distribution with scale adjusted to fit range
            scale = (max_size - min_size) / 3  # reasonable scale
            item_sizes = np.random.exponential(scale, n_items) + min_size
            item_sizes = np.minimum(item_sizes, max_size)  # cap at max_size
        else:
            raise ValueError(f"Unknown size distribution: {size_distribution}")
        
        # Ensure no item is larger than bin capacity
        item_sizes = np.minimum(item_sizes, bin_capacity - 0.1)  # Slight buffer
        
        # Set max_bins if not provided
        if max_bins is None:
            max_bins = n_items
        
        # Round sizes to one decimal place for cleaner display
        item_sizes = np.round(item_sizes, 1)
        
        instance_name = name or f"Random_BinPacking_{n_items}"
        
        return cls(item_sizes, bin_capacity, max_bins, name=instance_name, difficulty=difficulty)
    
    @classmethod
    def generate_benchmark_suite(cls, difficulty_levels: List[str] = None) -> Dict[str, List['BinPacking']]:
        """
        Generate a suite of benchmark Bin Packing instances at different difficulty levels.
        
        Args:
            difficulty_levels: List of difficulty levels to generate
                             (defaults to ['easy', 'medium', 'hard'])
            
        Returns:
            Dict[str, List[BinPacking]]: Dictionary mapping difficulty levels to lists of instances
        """
        if difficulty_levels is None:
            difficulty_levels = ['easy', 'medium', 'hard']
            
        suite = {}
        
        # Define instance parameters for each difficulty level
        configs = {
            'easy': {
                'n_items': [10, 15, 20],
                'bin_capacity': 100.0,
                'size_range': (20.0, 60.0),  # Larger items relative to bin capacity
                'distribution': 'uniform'
            },
            'medium': {
                'n_items': [30, 50, 75],
                'bin_capacity': 100.0,
                'size_range': (10.0, 40.0),  # Medium items
                'distribution': 'normal'
            },
            'hard': {
                'n_items': [100, 150, 200],
                'bin_capacity': 100.0,
                'size_range': (5.0, 30.0),  # Smaller, more varied items
                'distribution': 'exp'
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
                    
                    name = f"{level.capitalize()}_BinPacking_{size}_{i+1}"
                    min_size, max_size = config['size_range']
                    
                    instance = cls.generate_random_instance(
                        n_items=size,
                        bin_capacity=config['bin_capacity'],
                        min_size=min_size,
                        max_size=max_size,
                        size_distribution=config['distribution'],
                        seed=seed,
                        name=name,
                        difficulty=level
                    )
                    suite[level].append(instance)
        
        return suite
    
    def get_constraint_generator(self) -> Optional[Callable]:
        """
        Return a function that generates additional constraints during branch-and-bound.
        
        For Bin Packing Problem, we don't need lazy constraints, so return None.
        
        Returns:
            None: Bin Packing doesn't require lazy constraint generation
        """
        return None
    
    @property
    def name(self) -> str:
        """Get the name of this Bin Packing instance."""
        return self._name
    
    @property
    def size(self) -> Dict[str, int]:
        """Get size metrics for this Bin Packing instance."""
        n_vars = self.n_items * self.max_bins + self.max_bins  # x_ij + y_j
        n_constraints = self.n_items + self.max_bins + self.n_items * self.max_bins
        
        return {
            'items': self.n_items,
            'max_bins': self.max_bins,
            'variables': n_vars,
            'constraints': n_constraints
        }
    
    @property
    def difficulty(self) -> str:
        """Get the difficulty level of this Bin Packing instance."""
        return self._difficulty


def create_predefined_instances() -> Dict[str, BinPacking]:
    """
    Create predefined Bin Packing instances for examples and testing.
    
    Returns:
        Dict[str, BinPacking]: Dictionary of named Bin Packing instances
    """
    instances = {}
    
    # Example 1: Simple instance with 5 items
    item_sizes_1 = np.array([25.0, 40.0, 30.0, 50.0, 20.0])
    instances["small"] = BinPacking(
        item_sizes=item_sizes_1, 
        bin_capacity=100.0, 
        max_bins=3,
        name="Small_BinPacking", 
        difficulty="easy"
    )
    
    # Example 2: Medium instance with 10 items
    item_sizes_2 = np.array([15.0, 25.0, 30.0, 45.0, 20.0, 35.0, 10.0, 40.0, 25.0, 15.0])
    instances["medium"] = BinPacking(
        item_sizes=item_sizes_2, 
        bin_capacity=70.0, 
        max_bins=6,
        name="Medium_BinPacking", 
        difficulty="medium"
    )
    
    # Example 3: Hard instance with different size distribution
    item_sizes_3 = np.array([
        7.0, 12.0, 8.0, 11.0, 25.0, 30.0, 5.0, 10.0, 15.0, 18.0,
        22.0, 9.0, 14.0, 17.0, 6.0, 16.0, 13.0, 19.0, 21.0, 24.0
    ])
    instances["varied"] = BinPacking(
        item_sizes=item_sizes_3, 
        bin_capacity=50.0, 
        max_bins=10,
        name="Varied_BinPacking", 
        difficulty="hard"
    )
    
    return instances