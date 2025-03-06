"""
Priority queue strategies for branch-and-bound optimization.

This module provides various node prioritization strategies for the branch-and-bound algorithm.
Each strategy defines how nodes should be ordered in the priority queue.
"""

import math
import numpy as np
from typing import Any, Callable, Optional, Tuple


class NodePrioritizer:
    """
    Base class for node prioritization strategies.
    
    Defines how nodes are compared and ordered in the priority queue.
    For maximization problems, higher values have higher priority.
    """
    
    def get_priority_key(self, node: Any) -> Tuple:
        """
        Get the priority key for a node.
        
        The priority key is used for comparison in the heap.
        For maximization, we negate the values so that higher values have higher priority.
        
        Args:
            node: The node to get a priority key for
            
        Returns:
            tuple: A tuple that can be compared for ordering
        """
        raise NotImplementedError("Subclasses must implement get_priority_key")
        
    def compare(self, node1: Any, node2: Any) -> bool:
        """
        Compare two nodes to determine priority order.
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            bool: True if node1 has higher priority than node2
        """
        return self.get_priority_key(node1) > self.get_priority_key(node2)


class BestBoundPrioritizer(NodePrioritizer):
    """
    Prioritize nodes with the highest bound (best-bound strategy).
    
    This strategy explores the most promising regions of the search space first,
    which can lead to better pruning and faster convergence.
    """
    
    def get_priority_key(self, node: Any) -> Tuple:
        """
        Get priority based on the LP relaxation objective value.
        
        Args:
            node: The node to prioritize
            
        Returns:
            tuple: (objective_value,) as the priority key
        """
        # For maximization, higher objective values have higher priority
        return (node.value,)


class DepthFirstPrioritizer(NodePrioritizer):
    """
    Prioritize nodes with the greatest depth (depth-first strategy).
    
    This strategy tends to find feasible solutions quickly but may explore
    large parts of the tree before finding the optimal solution.
    """
    
    def get_priority_key(self, node: Any) -> Tuple:
        """
        Get priority based on depth in the tree.
        
        Args:
            node: The node to prioritize
            
        Returns:
            tuple: (depth, objective_value) as the priority key
        """
        # Primary key: depth (higher depth has priority)
        # Secondary key: objective value (higher value has priority)
        return (node.depth, node.value)


class BreadthFirstPrioritizer(NodePrioritizer):
    """
    Prioritize nodes with the smallest depth (breadth-first strategy).
    
    This strategy thoroughly explores all possibilities at each level
    before moving deeper in the tree.
    """
    
    def get_priority_key(self, node: Any) -> Tuple:
        """
        Get priority based on depth in the tree (reversed).
        
        Args:
            node: The node to prioritize
            
        Returns:
            tuple: (-depth, objective_value) as the priority key
        """
        # Primary key: negative depth (lower depth has priority)
        # Secondary key: objective value (higher value has priority)
        return (-node.depth, node.value)


class HybridPrioritizer(NodePrioritizer):
    """
    Combined strategy using both bound and depth.
    
    This strategy balances between exploration and exploitation.
    """
    
    def __init__(self, alpha: float = 0.5, max_depth: int = 100, max_val: float = 1000.0):
        """
        Initialize with weight for the bound component.
        
        Args:
            alpha: Weight for bound (0 means pure depth-first, 1 means pure best-bound)
            max_depth: Estimated maximum depth for normalization
            max_val: Estimated maximum objective value for normalization
        """
        self.alpha = alpha
        self.max_depth = max_depth
        self.max_val = max_val
        
    def get_priority_key(self, node: Any) -> Tuple:
        """
        Get priority based on weighted combination of depth and bound.
        
        Args:
            node: The node to prioritize
            
        Returns:
            tuple: (combined_score,) as the priority key
        """
        # Normalize depth and objective value to [0,1]
        norm_depth = node.depth / self.max_depth if self.max_depth > 0 else 0
        norm_value = node.value / self.max_val if self.max_val > 0 else 0
        
        # Combine with weighted sum - alpha controls the balance
        # For exploration vs. exploitation
        combined_score = (1 - self.alpha) * norm_depth + self.alpha * norm_value
        
        return (combined_score,)


class DecayingBestBoundPrioritizer(NodePrioritizer):
    """
    Best-bound strategy with depth-based exponential decay.
    
    This strategy gives higher priority to shallower nodes with good bounds,
    which can help balance between finding good solutions quickly and
    proving optimality.
    """
    
    def __init__(self, beta: float = 0.05):
        """
        Initialize with decay rate.
        
        Args:
            beta: Decay factor controlling the influence of depth (0.01-0.1 typical)
        """
        self.beta = beta
        
    def get_priority_key(self, node: Any) -> Tuple:
        """
        Get priority based on objective value decayed by depth.
        
        Args:
            node: The node to prioritize
            
        Returns:
            tuple: (decayed_value,) as the priority key
        """
        # Apply exponential decay based on depth
        # Deeper nodes have lower priority, all else being equal
        decayed_value = node.value * math.exp(-self.beta * node.depth)
        
        return (decayed_value,)


class EstimatedValuePrioritizer(NodePrioritizer):
    """
    Prioritize based on estimation of integer solution value.
    
    This strategy tries to predict the best integer solution
    that can be reached from each node.
    """
    
    def __init__(self, pseudo_costs=None):
        """
        Initialize with pseudocosts for fractional variables.
        
        Args:
            pseudo_costs: Dictionary mapping variables to pseudo-costs
        """
        self.pseudo_costs = pseudo_costs or {}
        
    def get_priority_key(self, node: Any) -> Tuple:
        """
        Get priority based on estimated integer solution value.
        
        Args:
            node: The node to prioritize
            
        Returns:
            tuple: (estimated_value,) as the priority key
        """
        # Start with the LP relaxation value
        estimated_value = node.value
        
        # Apply penalty for each fractional variable
        if node.relaxed_soln is not None:
            for idx in node.indices_frac:
                x_val = node.relaxed_soln[idx]
                frac_part = abs(x_val - round(x_val))
                
                # Use 1.0 as default pseudocost if not available
                pseudo_cost = self.pseudo_costs.get(idx, 1.0)
                
                # Apply penalty proportional to fractionality
                estimated_value -= frac_part * pseudo_cost
                
        return (estimated_value,)