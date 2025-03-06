"""
Branching strategies for branch-and-bound optimization.

This module provides various variable selection strategies for the branch-and-bound algorithm.
Each strategy defines how to choose the next variable to branch on.
"""

import numpy as np
import random
from typing import Any, Dict, List, Optional, Tuple

from src.core.node import BranchingCandidate


class BranchingStrategy:
    """
    Base class for branching strategies.
    
    Defines how to select variables for branching in the branch-and-bound algorithm.
    """
    
    def select_branching_variable(self, node: Any) -> int:
        """
        Select a variable to branch on.
        
        Args:
            node: The node to select a branching variable for
            
        Returns:
            int: Index of the selected variable
        """
        raise NotImplementedError("Subclasses must implement select_branching_variable")


class MostFractionalBranching(BranchingStrategy):
    """
    Select the variable with value closest to 0.5.
    
    This is a simple branching rule that aims to create balanced branches.
    """
    
    def select_branching_variable(self, node: Any) -> int:
        """
        Select the most fractional variable.
        
        Args:
            node: The node to select a branching variable for
            
        Returns:
            int: Index of the most fractional variable
        """
        if not node.indices_frac:
            raise ValueError("No fractional variables to branch on")
            
        # Find the variable with fractional part closest to 0.5
        best_var = -1
        best_distance = float('inf')
        
        for idx in node.indices_frac:
            value = node.relaxed_soln[idx]
            frac_part = abs(value - round(value))
            distance = abs(0.5 - frac_part)
            
            if distance < best_distance:
                best_distance = distance
                best_var = idx
                
        return best_var


class PseudoCostBranching(BranchingStrategy):
    """
    Select variable based on pseudo-costs.
    
    Pseudo-costs estimate the objective improvement per unit change
    in a variable's value, based on historical performance.
    """
    
    def __init__(self, reliability_threshold: int = 8):
        """
        Initialize with reliability threshold for strong branching.
        
        Args:
            reliability_threshold: Minimum number of observations needed
                                  for a pseudo-cost to be considered reliable
        """
        self.pseudo_costs_up = {}    # Maps variable index to historical up-branch costs
        self.pseudo_costs_down = {}  # Maps variable index to historical down-branch costs
        self.reliability = {}        # Maps variable index to count of observations
        self.reliability_threshold = reliability_threshold
        
    def select_branching_variable(self, node: Any) -> int:
        """
        Select variable with highest estimated score based on pseudo-costs.
        
        Args:
            node: The node to select a branching variable for
            
        Returns:
            int: Index of the selected variable
        """
        if not node.indices_frac:
            raise ValueError("No fractional variables to branch on")
        
        # Evaluate the score for each candidate
        candidates = []
        
        for idx in node.indices_frac:
            value = node.relaxed_soln[idx]
            frac_part = abs(value - round(value))
            
            # If this variable has reliable pseudo-costs, use them
            if self.reliability.get(idx, 0) >= self.reliability_threshold:
                up_cost = self.pseudo_costs_up.get(idx, 1.0) * (1.0 - frac_part)
                down_cost = self.pseudo_costs_down.get(idx, 1.0) * frac_part
                
                # Score is the product of estimated improvements (higher is better)
                score = max(1e-6, up_cost) * max(1e-6, down_cost)
                candidates.append((idx, score))
            else:
                # For unreliable variables, we'll try strong branching later
                candidates.append((idx, -1))
                
        # If we have any reliable candidates, choose the best one
        reliable_candidates = [(idx, score) for idx, score in candidates if score >= 0]
        if reliable_candidates:
            return max(reliable_candidates, key=lambda x: x[1])[0]
        
        # Otherwise, just pick the most fractional variable
        return MostFractionalBranching().select_branching_variable(node)
                
    def update_pseudo_costs(self, var_idx: int, parent_bound: float, 
                           up_bound: float, down_bound: float):
        """
        Update pseudo-costs based on observed branching results.
        
        Args:
            var_idx: Variable index
            parent_bound: Parent node bound
            up_bound: Upper branch bound
            down_bound: Lower branch bound
        """
        # Initialize if not present
        if var_idx not in self.pseudo_costs_up:
            self.pseudo_costs_up[var_idx] = 0.0
            self.pseudo_costs_down[var_idx] = 0.0
            self.reliability[var_idx] = 0
            
        # Calculate improvements
        up_improvement = max(0, parent_bound - up_bound)
        down_improvement = max(0, parent_bound - down_bound)
        
        # Update pseudo-costs with exponential smoothing (alpha=0.3)
        alpha = 0.3
        self.pseudo_costs_up[var_idx] = alpha * up_improvement + (1 - alpha) * self.pseudo_costs_up[var_idx]
        self.pseudo_costs_down[var_idx] = alpha * down_improvement + (1 - alpha) * self.pseudo_costs_down[var_idx]
        
        # Increment reliability counter
        self.reliability[var_idx] = self.reliability.get(var_idx, 0) + 1


class StrongBranching(BranchingStrategy):
    """
    Select variable by evaluating the impact of branching.
    
    This strategy solves the LP relaxations for potential child nodes
    to estimate the impact of branching on each variable.
    """
    
    def __init__(self, lp_solver, max_candidates: int = 10):
        """
        Initialize with LP solver function and maximum candidates to evaluate.
        
        Args:
            lp_solver: Function to solve LP relaxations
            max_candidates: Maximum number of variables to evaluate with strong branching
        """
        self.lp_solver = lp_solver
        self.max_candidates = max_candidates
        
    def select_branching_variable(self, node: Any) -> int:
        """
        Select variable by solving LP relaxations for child nodes.
        
        Args:
            node: The node to select a branching variable for
            
        Returns:
            int: Index of the selected variable
        """
        if not node.indices_frac:
            raise ValueError("No fractional variables to branch on")
            
        # Limit the number of candidates for computational efficiency
        candidates = node.indices_frac.copy()
        if len(candidates) > self.max_candidates:
            # Pre-filter by fractionality
            fractionality = []
            for idx in candidates:
                value = node.relaxed_soln[idx]
                frac_part = abs(value - round(value))
                fractionality.append((idx, frac_part))
            
            # Sort by fractionality (closest to 0.5 first)
            fractionality.sort(key=lambda x: abs(0.5 - x[1]))
            candidates = [idx for idx, _ in fractionality[:self.max_candidates]]
        
        # Evaluate each candidate by solving the child node LP relaxations
        best_score = -float('inf')
        best_var = -1
        
        for idx in candidates:
            value = node.relaxed_soln[idx]
            frac_part = value - np.floor(value)
            
            # Create the branching constraints for down branch (x <= floor(value))
            down_A_ub = np.vstack([node.A_ub, np.zeros(node.A_ub.shape[1])])
            down_A_ub[-1, idx] = 1
            down_b_ub = np.append(node.b_ub, np.floor(value))
            
            # Create the branching constraints for up branch (x >= ceil(value))
            up_A_ub = np.vstack([node.A_ub, np.zeros(node.A_ub.shape[1])])
            up_A_ub[-1, idx] = -1
            up_b_ub = np.append(node.b_ub, -np.ceil(value))
            
            # Solve LP relaxations for child nodes
            c = np.ones(node.A_ub.shape[1])  # Placeholder objective, actual comes from solver
            down_result = self.lp_solver(c, down_A_ub, down_b_ub)
            up_result = self.lp_solver(c, up_A_ub, up_b_ub)
            
            # Calculate score based on child node bounds
            # Default is product of degradations (min rule)
            if down_result.success and up_result.success:
                down_bound = down_result.fun
                up_bound = up_result.fun
                
                # Calculate degradation from parent bound
                down_degradation = max(0, node.value - down_bound)
                up_degradation = max(0, node.value - up_bound)
                
                # Higher score is better for branching
                score = down_degradation * up_degradation
                
                if score > best_score:
                    best_score = score
                    best_var = idx
            elif down_result.success:
                # Only down branch feasible, use its value
                best_var = idx
                break
            elif up_result.success:
                # Only up branch feasible, use its value
                best_var = idx
                break
        
        # If no variables evaluated successfully, fall back to most fractional
        if best_var == -1:
            best_var = MostFractionalBranching().select_branching_variable(node)
        
        return best_var


class ReliabilityBranching(BranchingStrategy):
    """
    Combines pseudo-cost and strong branching strategies.
    
    Uses strong branching for variables with unreliable pseudo-costs
    and switches to pseudo-cost branching when enough data is gathered.
    """
    
    def __init__(self, lp_solver, reliability_threshold: int = 8, max_candidates: int = 10):
        """
        Initialize with LP solver and reliability parameters.
        
        Args:
            lp_solver: Function to solve LP relaxations
            reliability_threshold: Minimum observation count for reliable pseudo-costs
            max_candidates: Maximum candidates to evaluate with strong branching
        """
        self.pseudo_cost = PseudoCostBranching(reliability_threshold)
        self.strong_branch = StrongBranching(lp_solver, max_candidates)
        
    def select_branching_variable(self, node: Any) -> int:
        """
        Select variable using reliability branching.
        
        Args:
            node: The node to select a branching variable for
            
        Returns:
            int: Index of the selected variable
        """
        if not node.indices_frac:
            raise ValueError("No fractional variables to branch on")
            
        # Identify unreliable variables
        unreliable = []
        for idx in node.indices_frac:
            if self.pseudo_cost.reliability.get(idx, 0) < self.pseudo_cost.reliability_threshold:
                unreliable.append(idx)
                
        # If all variables are reliable, use pseudo-cost branching
        if not unreliable:
            return self.pseudo_cost.select_branching_variable(node)
            
        # Otherwise, use strong branching for unreliable variables
        # and update their pseudo-costs
        
        # First, restrict node's fractional variables to unreliable ones
        original_indices = node.indices_frac
        node.indices_frac = unreliable
        
        # Use strong branching to select among unreliable variables
        selected_var = self.strong_branch.select_branching_variable(node)
        
        # Restore original indices
        node.indices_frac = original_indices
        
        return selected_var
        
    def update_pseudo_costs(self, var_idx: int, parent_bound: float, 
                           up_bound: float, down_bound: float):
        """
        Update pseudo-costs based on observed branching results.
        
        Delegates to the pseudo_cost component.
        
        Args:
            var_idx: Variable index
            parent_bound: Parent node bound
            up_bound: Upper branch bound
            down_bound: Lower branch bound
        """
        self.pseudo_cost.update_pseudo_costs(var_idx, parent_bound, up_bound, down_bound)