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
    
    def size_metrics(self) -> Dict[str, int]:
        """
        Get detailed size metrics for benchmarking purposes.
        
        Returns:
            Dict[str, int]: Dictionary of size metrics specific to Bin Packing
        """
        return {
            'size_items': self.n_items,
            'size_bins': self.max_bins,
            'size_variables': self.n_items * self.max_bins + self.max_bins, # Item-bin assignment + bin usage
            'size_constraints': self.n_items + self.max_bins + 1  # Item assignment, bin capacity, objective
        }
    
    def get_specific_metrics(self) -> Dict[str, Any]:
        """
        Get problem-specific metrics for Bin Packing benchmarking.
        
        Returns:
            Dict[str, Any]: Dictionary of Bin Packing-specific metrics
        """
        # Calculate metrics like item size statistics, bin utilization, etc.
        item_size_mean = np.mean(self.item_sizes)
        item_size_std = np.std(self.item_sizes)
        item_size_min = np.min(self.item_sizes)
        item_size_max = np.max(self.item_sizes)
        
        # Calculate the number of items that can fit in a bin
        avg_items_per_bin = self.bin_capacity / item_size_mean if item_size_mean > 0 else float('inf')
        
        # Ratio of total item size to bin capacity
        total_size = np.sum(self.item_sizes)
        min_bins_required = np.ceil(total_size / self.bin_capacity)
        
        # Bin filling efficiency (theoretical perfect utilization)
        perfect_utilization = total_size / (min_bins_required * self.bin_capacity)
        
        return {
            'item_size_mean': item_size_mean,
            'item_size_std': item_size_std,
            'item_size_min': item_size_min,
            'item_size_max': item_size_max,
            'item_size_to_capacity_ratio': item_size_mean / self.bin_capacity,
            'avg_items_per_bin': avg_items_per_bin,
            'min_bins_required': min_bins_required,
            'perfect_utilization': perfect_utilization,
            'is_minimization': True
        }
    
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
                         figsize: Tuple[int, int] = (14, 10), **kwargs) -> str:
        """
        Visualize a solution to the Bin Packing Problem with enhanced 3D representation.
        
        Creates a visualization with both 2D and 3D representations of how items are packed.
        
        Args:
            solution: Binary solution vector
            is_optimal: Whether the solution is optimal
            figsize: Figure size as (width, height)
            **kwargs: Additional visualization parameters
            
        Returns:
            str: Path to the saved visualization file
        """
        # Get additional information from kwargs
        step = kwargs.get('step', 0)
        nodes_explored = kwargs.get('nodes_explored', 0)
        elapsed_time = kwargs.get('elapsed_time', 0)
        best_obj_value = kwargs.get('best_obj_value', 0)
        title = kwargs.get('title', f"Bin Packing Solution - {self.name}")
        animated = kwargs.get('animated', False)
        use_3d = kwargs.get('use_3d', True)  # Option to disable 3D view for performance
        
        # Extract x_ij and y_j variables
        n_items = self.n_items
        max_bins = self.max_bins
        x_vars = solution[:n_items * max_bins].reshape(n_items, max_bins)
        y_vars = solution[n_items * max_bins:]
        
        # Count used bins
        used_bins = int(np.sum(y_vars > 0.5))
        
        # Create main figure with specified dimensions
        if use_3d:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, used_bins + 1, height_ratios=[2, 1],
                                 width_ratios=[2] + [1] * used_bins)
        else:
            fig, axes = plt.subplots(1, used_bins, figsize=figsize, 
                                   squeeze=False, constrained_layout=True)
        
        # Set up item colors
        colors = plt.cm.tab20(np.linspace(0, 1, self.n_items))
        
        # Create dictionary to track bin assignments and statistics
        bin_stats = {}
        
        # Process each bin to collect items and statistics
        bin_idx = 0
        for j in range(max_bins):
            if y_vars[j] < 0.5:  # Skip unused bins
                continue
            
            # Get items in this bin
            bin_items = [i for i in range(n_items) if x_vars[i, j] > 0.5]
            
            # Calculate bin utilization
            bin_load = sum(self.item_sizes[i] for i in bin_items)
            utilization = bin_load / self.bin_capacity * 100
            
            # Store bin information
            bin_stats[j] = {
                'items': bin_items,
                'load': bin_load,
                'utilization': utilization,
                'display_index': bin_idx
            }
            
            bin_idx += 1
        
        # Part 1: Create 3D visualization if enabled
        if use_3d:
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            # Create 3D subplot for bin visualization
            ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
            
            # Set up 3D view parameters
            ax_3d.view_init(elev=30, azim=45)
            
            # Calculate reasonable dimensions for visualization
            bin_width = 1.0
            bin_depth = 1.0
            bin_height = self.bin_capacity
            bin_spacing = 1.5
            
            # Function to create a cuboid for an item
            def create_cuboid(x, y, z, width, depth, height, color):
                vertices = [
                    [x, y, z], [x+width, y, z], [x+width, y+depth, z], [x, y+depth, z],
                    [x, y, z+height], [x+width, y, z+height], 
                    [x+width, y+depth, z+height], [x, y+depth, z+height]
                ]
                
                faces = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                    [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                    [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
                    [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
                    [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                    [vertices[3], vertices[2], vertices[6], vertices[7]]   # back
                ]
                
                collection = Poly3DCollection(faces, alpha=0.8, linewidths=1, edgecolor='black')
                collection.set_facecolor(color)
                return collection
            
            # Draw each bin and its items
            for j, stats in bin_stats.items():
                bin_x = stats['display_index'] * bin_spacing
                
                # Draw bin outline as wireframe
                bin_edges = [
                    [bin_x, 0, 0], [bin_x+bin_width, 0, 0],
                    [bin_x+bin_width, bin_depth, 0], [bin_x, bin_depth, 0],
                    [bin_x, 0, bin_height], [bin_x+bin_width, 0, bin_height],
                    [bin_x+bin_width, bin_depth, bin_height], [bin_x, bin_depth, bin_height]
                ]
                
                # Draw bin edges
                for start, end in [
                    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
                    (4, 5), (5, 6), (6, 7), (7, 4),  # top
                    (0, 4), (1, 5), (2, 6), (3, 7)   # sides
                ]:
                    ax_3d.plot3D(
                        [bin_edges[start][0], bin_edges[end][0]],
                        [bin_edges[start][1], bin_edges[end][1]],
                        [bin_edges[start][2], bin_edges[end][2]],
                        color='gray', linestyle='--', alpha=0.5
                    )
                
                # Sort items by size for better packing visualization
                sorted_items = sorted(stats['items'], key=lambda i: self.item_sizes[i], reverse=True)
                
                # Simple packing algorithm for visualization
                # This is just for visual representation, not an optimal packing
                current_height = 0
                row_width = 0
                row_items = []
                
                for i in sorted_items:
                    item_size = self.item_sizes[i]
                    # Normalize item size to be a cube proportion
                    item_vol = item_size / self.bin_capacity
                    item_dim = item_vol ** (1/3)  # Cube root for 3D scaling
                    
                    # Scale dimensions to fit bin
                    item_width = bin_width * item_dim * 0.9  # 90% of bin width
                    item_depth = bin_depth * item_dim * 0.9  # 90% of bin depth
                    item_height = item_size * 0.8  # Keep height proportional to size
                    
                    # If items would exceed bin width, start a new row
                    if row_width + item_width > bin_width:
                        current_height += max(self.item_sizes[i] for i in row_items) * 0.8
                        row_width = 0
                        row_items = []
                    
                    # Position item
                    item_x = bin_x + row_width
                    item_y = 0.1  # Small offset from front of bin
                    item_z = current_height
                    
                    # Add item to 3D plot
                    item_color = colors[i]
                    cuboid = create_cuboid(
                        item_x, item_y, item_z, 
                        item_width, item_depth, item_height, 
                        item_color
                    )
                    ax_3d.add_collection3d(cuboid)
                    
                    # Add item label
                    ax_3d.text(
                        item_x + item_width/2, 
                        item_y + item_depth/2, 
                        item_z + item_height/2, 
                        f"{i}", 
                        color='black', 
                        ha='center', va='center', 
                        fontweight='bold'
                    )
                    
                    # Update row information
                    row_width += item_width
                    row_items.append(i)
                
                # Add bin label
                ax_3d.text(
                    bin_x + bin_width/2, 
                    bin_depth/2, 
                    bin_height * 1.1, 
                    f"Bin {j+1}\n{stats['utilization']:.1f}%", 
                    color='black', 
                    ha='center', va='bottom'
                )
            
            # Set up 3D plot limits and labels
            max_x = used_bins * bin_spacing + bin_width
            ax_3d.set_xlim(0, max_x)
            ax_3d.set_ylim(0, bin_depth * 1.5)
            ax_3d.set_zlim(0, bin_height * 1.2)
            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Y')
            ax_3d.set_zlabel('Size')
            ax_3d.set_title('3D Bin Packing Visualization')
        
        # Part 2: Create 2D bin visualizations
        for j, stats in bin_stats.items():
            if use_3d:
                # Use gridspec for layout
                ax = fig.add_subplot(gs[1, stats['display_index'] + 1])
            else:
                # Use the predefined subplot axes
                ax = axes[0, stats['display_index']]
            
            # Get items in this bin
            bin_items = stats['items']
            
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
            ax.text(0.5, self.bin_capacity * 1.02, 
                   f"{stats['utilization']:.1f}%", 
                   ha='center', va='bottom', fontsize=8)
        
        # Add information panel if using 3D view
        if use_3d:
            # Add information text in the remaining gridspec cell
            ax_info = fig.add_subplot(gs[1, 0])
            ax_info.axis('off')  # Hide axes
            
            # Calculate overall statistics
            total_items = sum(len(stats['items']) for stats in bin_stats.values())
            avg_utilization = np.mean([stats['utilization'] for stats in bin_stats.values()])
            
            # Create info text
            status = "Optimal" if is_optimal else "Candidate"
            info_text = (
                f"Status: {status}\n\n"
                f"Bins Used: {used_bins}/{self.max_bins}\n"
                f"Items Packed: {total_items}/{self.n_items}\n"
                f"Avg Utilization: {avg_utilization:.1f}%\n"
                f"Bin Capacity: {self.bin_capacity}\n"
            )
            
            # Add exploration info if provided
            if animated:
                info_text += (
                    f"\nStep: {step}\n"
                    f"Nodes Explored: {nodes_explored}\n"
                    f"Time: {elapsed_time:.2f}s\n"
                )
                
            # Draw text box
            ax_info.text(0.5, 0.5, info_text, 
                       ha='center', va='center', 
                       fontsize=10, fontweight='normal',
                       bbox=dict(facecolor='white', alpha=0.8, 
                               boxstyle='round,pad=0.7', edgecolor='gray'),
                       transform=ax_info.transAxes)
        
        # Add overall title
        status = "Optimal" if is_optimal else "Candidate"
        plt.suptitle(title, fontsize=14, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        counter = kwargs.get('counter', step if animated else 1)
        status_str = "optimal" if is_optimal else "candidate"
        filename = f"plots/{self.name}_solution_{counter:03d}_{status_str}.png"
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