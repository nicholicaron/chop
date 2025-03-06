"""
Example script demonstrating the use of the refactored Branch-and-Bound solver
on Traveling Salesman Problem instances.
"""

import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import BranchAndBoundSolver
from src.strategies import (
    MostFractionalBranching, 
    BestBoundPrioritizer, 
    DecayingBestBoundPrioritizer
)
from src.utils import BnBLogger, Timer
from src.tsp import TSPInstance


def solution_callback(solution, is_optimal, tsp_instance, problem_name):
    """Callback function for visualizing solutions during branch and bound."""
    tsp_instance.plot_solution(solution, is_optimal, problem_name)


def main(visualize):
    # Create logs and plots directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    # Create logger
    logger = BnBLogger(log_file='logs/tsp_example.log')
    
    # Configure solver
    prioritizer = DecayingBestBoundPrioritizer(beta=0.05)
    branching_strategy = MostFractionalBranching()
    
    # Create solver instance
    solver = BranchAndBoundSolver(
        prioritizer=prioritizer,
        branching_strategy=branching_strategy,
        logger=logger,
        early_stop_gap=1e-4,
        use_cuts=True,
        cut_probability=0.3
    )

    # Create TSP instances
    # Example 1: 3 cities in a triangle
    coords_3 = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0)
    }
    tsp_3 = TSPInstance(3, coords_3)
    tsp_3.plot_instance("Triangle TSP")
    
    # Example 2: 4 cities in a square
    coords_4 = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 1),
        3: (1, 0)
    }
    tsp_4 = TSPInstance(4, coords_4)
    tsp_4.plot_instance("Square TSP")
    
    # Example 3: 5 cities in a star pattern
    coords_5 = {
        0: (0, 0),    # center
        1: (1, 1),    # top right
        2: (-1, 1),   # top left
        3: (-1, -1),  # bottom left
        4: (1, -1)    # bottom right
    }
    tsp_5 = TSPInstance(5, coords_5)
    tsp_5.plot_instance("Star TSP")
    
    # Solve each instance
    for instance, name in [(tsp_3, "Triangle"), (tsp_4, "Square"), (tsp_5, "Star")]:
        print(f"\nSolving {name} TSP instance with {instance.n_cities} cities")
        
        # Get all constraint matrices
        c, A_eq, b_eq, A_ub, b_ub = instance.to_ilp()
        
        # Create callback closure
        callback_fn = lambda solution, is_optimal, problem_name: solution_callback(
            solution, is_optimal, instance, problem_name
        )
        
        # Solve the instance
        with Timer() as timer:
            result = solver.solve(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                problem_name=f"{name}_TSP",
                visualize=visualize,
                callback=callback_fn,
                tsp_instance=instance
            )
        
        # Unpack the result
        solution = result[0]  # Optimal solution vector
        value = result[1]    # Optimal objective value (negative due to maximization)
        num_nodes_explored = result[2]
        
        # Calculate the actual tour distance (positive value)
        actual_distance = 0
        edges = []
        if solution is not None:
            # Find edges in the solution
            for i in range(instance.n_cities):
                for j in range(i+1, instance.n_cities):
                    idx = instance._get_variable_index(i, j)
                    if abs(solution[idx] - 1.0) < 1e-6:
                        edges.append((i, j))
                        actual_distance += instance.distances[(i, j)]
        
        print(f"\nResults for {name} TSP:")
        print(f"  Time: {timer.elapsed:.3f}s")
        print(f"  Nodes explored: {num_nodes_explored}")
        print(f"  Optimal tour length: {actual_distance:.3f}")
        print(f"  Optimal tour edges: {edges}")
        
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TSP Solver with Branch and Bound")
    parser.add_argument("--visualize", action="store_true", help="Generate and save plots")
    args = parser.parse_args()
    
    print("Starting TSP solver...")
    main(args.visualize)