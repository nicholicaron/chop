"""
Example script demonstrating Knapsack Problem solving with the Branch-and-Bound solver.

This script shows how to:
1. Create Knapsack instances with different configurations
2. Solve them using the Branch-and-Bound solver
3. Visualize the solutions and the branch-and-bound tree
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(".")  # Add the project root to the path

from src.problems import Knapsack
from src.core.solver import BranchAndBoundSolver
from src.utils.logging import BnBLogger, Timer


def solve_knapsack_instance(knapsack_instance, solver, visualize=True):
    """
    Solve a Knapsack instance using the Branch-and-Bound solver.
    
    Args:
        knapsack_instance: The Knapsack instance to solve
        solver: The BranchAndBoundSolver instance
        visualize: Whether to generate visualizations
        
    Returns:
        Tuple of (solution, objective_value, node_count, elapsed_time)
    """
    # Convert instance to ILP form
    c, A_eq, b_eq, A_ub, b_ub = knapsack_instance.to_ilp()
    
    # Create a callback function for solution visualization
    def callback_fn(solution, is_optimal, problem_name):
        knapsack_instance.visualize_solution(solution, is_optimal)
    
    # Solve the instance
    with Timer() as timer:
        result = solver.solve(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            problem_name=knapsack_instance.name,
            visualize=visualize,
            callback=callback_fn
        )
    
    # Unpack the result
    solution, obj_value, node_count, _ = result
    
    # Calculate solution details if solution exists
    if solution is not None:
        # Validate the solution
        is_valid, actual_obj = knapsack_instance.validate_solution(solution)
        
        # Find the selected items
        selected_items = [i for i, x in enumerate(solution) if abs(x - 1.0) < 1e-6]
        total_weight = sum(knapsack_instance.weights[i] for i in selected_items)
        
        print(f"\nResults for {knapsack_instance.name}:")
        print(f"  Time: {timer.elapsed:.3f}s")
        print(f"  Nodes explored: {node_count}")
        print(f"  Total value: {actual_obj:.2f}")
        print(f"  Total weight: {total_weight:.2f} / {knapsack_instance.capacity:.2f}")
        print(f"  Selected items: {selected_items}")
        print(f"  Solution validation: {'Valid' if is_valid else 'Invalid'}")
    else:
        print(f"\nNo solution found for {knapsack_instance.name}")
    
    return solution, obj_value, node_count, timer.elapsed


def main():
    """Main function demonstrating Knapsack solving with the problem framework."""
    
    # Create directories for logs and plots
    os.makedirs('logs', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Create logger
    logger = BnBLogger(log_file='logs/knapsack_example.log')
    
    # Create solver instance with default settings
    solver = BranchAndBoundSolver(
        logger=logger,
        max_nodes=1000,
        early_stop_gap=1e-6,
        use_cuts=True
    )
    
    # Load predefined Knapsack instances
    print("Loading predefined Knapsack instances...")
    from src.problems import knapsack_instances
    predefined = knapsack_instances()
    
    # Visualize predefined instances
    for name, instance in predefined.items():
        print(f"Visualizing {instance.name}...")
        instance.visualize_instance()
    
    # Generate a random instance
    random_knapsack = Knapsack.generate_random_instance(
        n_items=15,
        seed=42,
        capacity_factor=0.5,
        name="Random_Knapsack_15",
        difficulty="medium"
    )
    random_knapsack.visualize_instance()
    
    # Solve each instance
    instances = list(predefined.values()) + [random_knapsack]
    
    for instance in instances:
        print(f"\nSolving {instance.name} with {instance.n_items} items")
        print(f"Capacity: {instance.capacity:.2f}")
        
        solve_knapsack_instance(instance, solver)
        print("\n" + "="*50 + "\n")
    
    print("\nAll Knapsack instances solved. Check the 'plots' directory for visualizations.")
    
    # Generate a benchmark suite with different difficulty levels
    print("\nGenerating benchmark suite of Knapsack instances...")
    benchmark_suite = Knapsack.generate_benchmark_suite(['easy', 'medium'])
    
    # Print information about the benchmark suite
    for difficulty, instances in benchmark_suite.items():
        print(f"\n{difficulty.capitalize()} Knapsack instances:")
        for instance in instances:
            print(f"  - {instance.name}: {instance.n_items} items, capacity: {instance.capacity:.2f}")
            
            # Visualize one instance per difficulty level as an example
            if instances.index(instance) == 0:
                instance.visualize_instance()
                
                # Solve the first instance of each difficulty level as an example
                print(f"\nSolving {instance.name} as an example...")
                solve_knapsack_instance(instance, solver)


if __name__ == "__main__":
    print("Starting Knapsack solver with the problem framework...")
    main()