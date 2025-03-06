"""
Example script demonstrating TSP solving with the Branch-and-Bound solver.

This script shows how to:
1. Create TSP instances with different configurations
2. Solve them using the Branch-and-Bound solver
3. Visualize the solutions and the branch-and-bound tree
"""

import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.solver import BranchAndBoundSolver
from src.strategies.branching import MostFractionalBranching
from src.strategies.priority_queue import BestBoundPrioritizer, DecayingBestBoundPrioritizer
from src.utils.logging import BnBLogger, Timer
from src.problems import TSP


def solve_tsp_instance(tsp_instance, solver, visualize=True):
    """
    Solve a TSP instance using the Branch-and-Bound solver.
    
    Args:
        tsp_instance: The TSP instance to solve
        solver: The BranchAndBoundSolver instance
        visualize: Whether to generate visualizations
        
    Returns:
        Tuple of (solution, objective_value, node_count, elapsed_time)
    """
    # Convert instance to ILP form
    c, A_eq, b_eq, A_ub, b_ub = tsp_instance.to_ilp()
    
    # Create a callback function for solution visualization
    def callback_fn(solution, is_optimal, problem_name):
        tsp_instance.visualize_solution(solution, is_optimal)
    
    # Solve the instance
    with Timer() as timer:
        result = solver.solve(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            problem_name=tsp_instance.name,
            visualize=visualize,
            callback=callback_fn,
            tsp_instance=tsp_instance  # For subtour elimination
        )
    
    # Unpack the result
    solution, obj_value, node_count, _ = result
    
    # Calculate tour details if solution exists
    if solution is not None:
        # Validate the solution
        is_valid, actual_obj = tsp_instance.validate_solution(solution)
        
        # Find the edges in the solution
        edges = []
        for i in range(tsp_instance.n_cities):
            for j in range(i+1, tsp_instance.n_cities):
                idx = tsp_instance._get_variable_index(i, j)
                if abs(solution[idx] - 1.0) < 1e-6:
                    edges.append((i, j))
        
        print(f"\nResults for {tsp_instance.name}:")
        print(f"  Time: {timer.elapsed:.3f}s")
        print(f"  Nodes explored: {node_count}")
        print(f"  Tour length: {-actual_obj:.3f}")  # Negate because we maximize negative distances
        print(f"  Tour edges: {edges}")
        print(f"  Solution validation: {'Valid' if is_valid else 'Invalid'}")
    else:
        print(f"\nNo solution found for {tsp_instance.name}")
    
    return solution, obj_value, node_count, timer.elapsed


def main(visualize):
    """Main function demonstrating TSP solving with the new problem framework."""
    
    # Create directories for logs and plots
    os.makedirs('logs', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Create logger
    logger = BnBLogger(log_file='logs/tsp_example.log')
    
    # Configure solver components
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
    
    # Create predefined TSP instances with the new framework
    
    # Example 1: 3 cities in a triangle
    coords_3 = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0)
    }
    tsp_3 = TSP(3, coords_3, name="Triangle_TSP", difficulty="easy")
    tsp_3.visualize_instance()
    
    # Example 2: 4 cities in a square
    coords_4 = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 1),
        3: (1, 0)
    }
    tsp_4 = TSP(4, coords_4, name="Square_TSP", difficulty="easy")
    tsp_4.visualize_instance()
    
    # Example 3: 5 cities in a star pattern
    coords_5 = {
        0: (0, 0),    # center
        1: (1, 1),    # top right
        2: (-1, 1),   # top left
        3: (-1, -1),  # bottom left
        4: (1, -1)    # bottom right
    }
    tsp_5 = TSP(5, coords_5, name="Star_TSP", difficulty="easy")
    tsp_5.visualize_instance()
    
    # Generate a random instance
    random_tsp = TSP.generate_random_instance(
        n_cities=7,
        seed=42,
        name="Random_TSP_7",
        difficulty="medium"
    )
    random_tsp.visualize_instance()
    
    # Solve each instance
    instances = [tsp_3, tsp_4, tsp_5, random_tsp]
    
    for instance in instances:
        print(f"\nSolving {instance.name} with {instance.n_cities} cities")
        solve_tsp_instance(instance, solver, visualize)
        print("\n" + "="*50 + "\n")
    
    print("\nAll TSP instances solved. Check the 'plots' directory for visualizations.")
    
    # Generate a benchmark suite (only easy difficulty for demonstration)
    print("\nGenerating a benchmark suite of TSP instances...")
    benchmark_suite = TSP.generate_benchmark_suite(['easy'])
    
    # Print information about the benchmark suite
    for difficulty, instances in benchmark_suite.items():
        print(f"\n{difficulty.capitalize()} TSP instances:")
        for instance in instances:
            print(f"  - {instance.name}: {instance.n_cities} cities")
            
            # Visualize the first instance of each difficulty
            if instances.index(instance) == 0:
                instance.visualize_instance()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TSP Solver with Branch and Bound")
    parser.add_argument("--visualize", action="store_true", help="Generate and save plots")
    args = parser.parse_args()
    
    print("Starting TSP solver with the new problem framework...")
    main(args.visualize)