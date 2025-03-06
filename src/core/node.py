"""
Node classes for branch-and-bound optimization.

This module contains core node representations for branch-and-bound trees.
Nodes store problem state, bounds, and solution information.
"""

import numpy as np
import math
from typing import Optional, List, Dict, Any, Tuple, Set


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
        
        For a max-heap priority queue (we're maximizing), we need to return 
        whether this node has higher priority than the other node.
        
        Args:
            other (Node): Node to compare against
            
        Returns:
            bool: True if this node should be processed before the other node
        """
        # Default comparison using objective value (for maximization problems)
        # Higher bounds are prioritized, so we reverse the comparison
        return self.value > other.value
    
    def get_state_features(self) -> Dict[str, Any]:
        """
        Extract features from this node for state representation in RL.
        
        Returns:
            Dict[str, Any]: Dictionary of node features
        """
        features = {
            'depth': self.depth,
            'objective_value': self.value,
            'parent_value': self.parent.value if self.parent else None,
            'branch_var': self.branch_var,
            'branch_val': self.branch_val,
            'branch_direction': self.branch_direction,
            'num_int': self.num_int,
            'num_frac': self.num_frac,
            'gap': self.optimality_gap
        }
        return features


class BranchingCandidate:
    """
    Represents a candidate variable for branching.
    
    Stores information about the variable and potential child nodes
    to support sophisticated branching strategies.
    """
    
    def __init__(self, var_idx: int, var_value: float):
        """
        Initialize a branching candidate.
        
        Args:
            var_idx (int): Index of the variable
            var_value (float): Current value of the variable
        """
        self.var_idx = var_idx
        self.var_value = var_value
        self.score = 0.0
        self.down_obj = None  # Objective after down branch
        self.up_obj = None    # Objective after up branch
        self.pseudocost_down = 0.0
        self.pseudocost_up = 0.0
        
    def compute_score(self, scoring_method: str = 'product') -> float:
        """
        Compute a score for this candidate based on potential child bounds.
        
        Args:
            scoring_method (str): Method to combine up/down bounds ('product', 'sum', etc.)
            
        Returns:
            float: Score value (higher is better for branching)
        """
        if self.down_obj is None or self.up_obj is None:
            return 0.0
            
        if scoring_method == 'product':
            return max(1e-6, abs(self.down_obj)) * max(1e-6, abs(self.up_obj))
        elif scoring_method == 'sum':
            return abs(self.down_obj) + abs(self.up_obj)
        else:
            return min(abs(self.down_obj), abs(self.up_obj))  # Default: min strategy