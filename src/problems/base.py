"""
Base class for optimization problems.

This module defines the abstract base class that all optimization problems must implement,
providing a standardized interface for working with different problem types.
"""

import abc
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional, Set


class OptimizationProblem(abc.ABC):
    """
    Abstract base class for all optimization problems.
    
    This class defines the interface that all optimization problems must implement
    to be compatible with the branch-and-bound solver and RL framework. Each problem
    type should subclass this and implement the required methods.
    """
    
    @abc.abstractmethod
    def to_ilp(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert problem to ILP formulation.
        
        Returns:
            Tuple containing:
            - c: Objective coefficients
            - A_eq: Equality constraint matrix (may be empty)
            - b_eq: Equality constraint right-hand side (may be empty)
            - A_ub: Inequality constraint matrix (may be empty)
            - b_ub: Inequality constraint right-hand side (may be empty)
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def validate_solution(self, solution: np.ndarray) -> Tuple[bool, float]:
        """
        Check if a solution is valid and compute its objective value.
        
        Args:
            solution: Solution vector to validate
            
        Returns:
            Tuple containing:
            - is_valid: Boolean indicating if the solution is valid
            - objective_value: The objective value of the solution
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def visualize_instance(self, title: str = None, **kwargs) -> str:
        """
        Visualize the problem instance.
        
        Args:
            title: Optional title for the visualization
            **kwargs: Additional visualization parameters
            
        Returns:
            str: Path to the saved visualization file
        """
        raise NotImplementedError
   
    @abc.abstractmethod
    def visualize_solution(self, solution: np.ndarray, is_optimal: bool = False, **kwargs) -> str:
        """
        Visualize a solution to the problem.
        
        Args:
            solution: Solution vector to visualize
            is_optimal: Whether the solution is optimal
            **kwargs: Additional visualization parameters
            
        Returns:
            str: Path to the saved visualization file
        """
        raise NotImplementedError
    
    @classmethod
    @abc.abstractmethod
    def generate_random_instance(cls, **params) -> 'OptimizationProblem':
        """
        Generate a random instance of the problem.
        
        Args:
            **params: Parameters controlling the instance generation
            
        Returns:
            OptimizationProblem: A new instance of the problem
        """
        raise NotImplementedError
    
    @classmethod
    @abc.abstractmethod
    def generate_benchmark_suite(cls, difficulty_levels: List[str]) -> Dict[str, List['OptimizationProblem']]:
        """
        Generate a suite of benchmark instances at different difficulty levels.
        
        Args:
            difficulty_levels: List of difficulty levels to generate (e.g., ['easy', 'medium', 'hard'])
            
        Returns:
            Dict[str, List[OptimizationProblem]]: Dictionary mapping difficulty levels to lists of problem instances
        """
        raise NotImplementedError
        
    @abc.abstractmethod
    def get_constraint_generator(self) -> Optional[callable]:
        """
        Return a function that generates additional constraints during branch-and-bound.
        
        This method is used for problems like TSP that require lazy constraint
        generation during the solution process (e.g., subtour elimination constraints).
        
        Returns:
            Optional[callable]: A function that takes a solution vector and returns 
            a list of (constraint, rhs) tuples, or None if no constraints needed
        """
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Get the name of the problem instance.
        
        Returns:
            str: A human-readable name for this problem instance
        """
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def size(self) -> Dict[str, int]:
        """
        Get the size metrics for this problem.
        
        Returns:
            Dict[str, int]: Dictionary of size metrics (e.g., num_variables, num_constraints)
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def size_metrics(self) -> Dict[str, int]:
        """
        Get detailed size metrics for benchmarking purposes.
        
        This should include problem-specific metrics like number of cities for TSP,
        number of items for Knapsack, etc.
        
        Returns:
            Dict[str, int]: Dictionary of size metrics
        """
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def difficulty(self) -> str:
        """
        Get the difficulty level of this problem instance.
        
        Returns:
            str: Difficulty level (e.g., 'easy', 'medium', 'hard')
        """
        raise NotImplementedError
        
    @abc.abstractmethod
    def get_specific_metrics(self) -> Dict[str, Any]:
        """
        Get problem-specific metrics for benchmarking.
        
        These metrics should capture structural properties of the problem instance
        that may correlate with difficulty, such as:
        - For TSP: symmetry, clustering coefficient, etc.
        - For Knapsack: correlation between weights and values, etc.
        - For Assignment: cost distribution, etc.
        
        Returns:
            Dict[str, Any]: Dictionary of problem-specific metrics
        """
        raise NotImplementedError
        
    def solve_lp_relaxation(self) -> float:
        """
        Solve the LP relaxation of the problem.
        
        Returns:
            float: The objective value of the LP relaxation
        """
        from ..simplex import solve_lp
        
        # Convert to ILP
        ilp_model = self.to_ilp()
        
        # Solve LP relaxation
        result = solve_lp(
            ilp_model, 
            integrality_constraints=False,
            early_stop_gap=0.0
        )
        
        return result.get('objective_value', float('inf'))