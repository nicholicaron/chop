"""
Problem module containing optimization problem implementations.

This module provides a standardized interface for different optimization problems
and generators to create problem instances of varying difficulty.
"""

from .base import OptimizationProblem
from .tsp import TSP, create_predefined_instances as tsp_instances
from .knapsack import Knapsack, create_predefined_instances as knapsack_instances
from .assignment import Assignment, create_predefined_instances as assignment_instances
from .bin_packing import BinPacking, create_predefined_instances as bin_packing_instances
from .set_cover import SetCover, create_predefined_instances as set_cover_instances


# Create a dictionary of all predefined instances for easy access
def get_predefined_instances():
    """
    Get a dictionary of all predefined problem instances.
    
    Returns:
        Dict: Dictionary containing predefined instances for each problem type
    """
    instances = {
        'tsp': tsp_instances(),
        'knapsack': knapsack_instances(),
        'assignment': assignment_instances(),
        'bin_packing': bin_packing_instances(),
        'set_cover': set_cover_instances()
    }
    return instances