"""
Example script demonstrating the use of the problem generation framework.

This script shows how to:
1. Create and visualize predefined problem instances
2. Generate random instances with controlled parameters
3. Create benchmark suites at different difficulty levels
4. Convert problems to ILP form and solve them using the Branch-and-Bound solver
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(".")  # Add the project root to the path

from src.problems import OptimizationProblem, TSP, Knapsack, get_predefined_instances
from src.core.solver import BranchAndBoundSolver


def visualize_predefined_instances():
    """Demonstrate the predefined problem instances."""
    
    # Get all predefined instances
    all_instances = get_predefined_instances()
    
    print("Predefined instances available:")
    for problem_type, instances in all_instances.items():
        print(f"\n{problem_type.upper()} Instances:")
        for name, instance in instances.items():
            print(f"  - {name}: {instance.name} (difficulty: {instance.difficulty})")
            # Visualize the instance
            img_path = instance.visualize_instance()
            print(f"    Visualization saved to: {img_path}")
            
            # Print size metrics
            size_info = instance.size
            print(f"    Size metrics: {size_info}")


def generate_and_solve_instance(problem_class, instance_params, solver_params=None):
    """Generate and solve a random instance of the given problem class."""
    
    # Generate a random instance
    instance = problem_class.generate_random_instance(**instance_params)
    
    print(f"\nGenerated instance: {instance.name}")
    print(f"Size: {instance.size}")
    
    # Visualize the instance
    img_path = instance.visualize_instance()
    print(f"Visualization saved to: {img_path}")
    
    # Convert to ILP form
    c, A_eq, b_eq, A_ub, b_ub = instance.to_ilp()
    print("\nILP Formulation:")
    print(f"Variables: {len(c)}")
    print(f"Objective: {c}")
    print(f"Equality constraints: {A_eq.shape if A_eq.size > 0 else '0'}")
    print(f"Inequality constraints: {A_ub.shape if A_ub.size > 0 else '0'}")
    
    # Initialize solver with default or provided parameters
    solver_config = solver_params or {}
    solver = BranchAndBoundSolver(**solver_config)
    
    # Define callback function for visualization
    def solution_callback(solution, is_optimal, problem_name):
        visualize_solution(instance, solution, is_optimal, problem_name)
    
    # Solve the instance
    print("\nSolving...")
    solution, obj_value, nodes, optimal_node = solver.solve(
        c=c, 
        A_ub=A_ub, 
        b_ub=b_ub, 
        A_eq=A_eq, 
        b_eq=b_eq,
        problem_name=instance.name,
        visualize=True,
        callback=solution_callback
    )
    
    # Print results
    print(f"\nResult:")
    if solution is None:
        print("No solution found.")
    else:
        print(f"Objective value: {obj_value}")
        print(f"Nodes explored: {nodes}")
        
        # Validate the solution
        is_valid, actual_obj = instance.validate_solution(solution)
        print(f"Solution validation: {'Valid' if is_valid else 'Invalid'}")
        print(f"Actual objective value: {actual_obj}")
    
    return instance, solution, obj_value


def visualize_solution(instance, solution, is_optimal, problem_name):
    """Visualize a solution for the given problem instance."""
    if solution is not None:
        img_path = instance.visualize_solution(solution, is_optimal)
        print(f"Solution visualization saved to: {img_path}")


def generate_benchmark_suite():
    """Generate benchmark suites for different problem types."""
    
    # Dictionary to store all generated instances
    benchmarks = {}
    
    # Generate TSP benchmark suite
    print("\nGenerating TSP benchmark suite...")
    tsp_suite = TSP.generate_benchmark_suite(['easy', 'medium'])  # Skip 'hard' for quick demo
    benchmarks['tsp'] = tsp_suite
    
    # Generate Knapsack benchmark suite
    print("\nGenerating Knapsack benchmark suite...")
    knapsack_suite = Knapsack.generate_benchmark_suite(['easy', 'medium'])  # Skip 'hard'
    benchmarks['knapsack'] = knapsack_suite
    
    # Display information about generated instances
    for problem_type, suite in benchmarks.items():
        print(f"\n{problem_type.upper()} Benchmark Suite:")
        for difficulty, instances in suite.items():
            print(f"  {difficulty.capitalize()} level - {len(instances)} instances:")
            for instance in instances:
                print(f"    - {instance.name}, size: {instance.size}")
                
                # Visualize one instance per difficulty level as an example
                if instances.index(instance) == 0:
                    img_path = instance.visualize_instance()
                    print(f"      Visualization saved to: {img_path}")
    
    return benchmarks


def main():
    """Main function demonstrating the problem generation framework."""
    
    # Ensure plot directory exists
    os.makedirs('plots', exist_ok=True)
    
    # Display available predefined instances
    visualize_predefined_instances()
    
    # Generate and solve a small TSP instance
    print("\n\n=== GENERATING AND SOLVING A TSP INSTANCE ===")
    tsp_params = {
        'n_cities': 6,
        'seed': 42,
        'name': 'Demo_TSP',
        'difficulty': 'easy'
    }
    tsp_instance, tsp_solution, tsp_obj = generate_and_solve_instance(TSP, tsp_params)
    
    # Generate and solve a small Knapsack instance
    print("\n\n=== GENERATING AND SOLVING A KNAPSACK INSTANCE ===")
    knapsack_params = {
        'n_items': 8,
        'seed': 42,
        'capacity_factor': 0.6,
        'name': 'Demo_Knapsack',
        'difficulty': 'easy'
    }
    knapsack_instance, knapsack_solution, knapsack_obj = generate_and_solve_instance(
        Knapsack, knapsack_params
    )
    
    # Generate benchmark suites
    print("\n\n=== GENERATING BENCHMARK SUITES ===")
    benchmark_suites = generate_benchmark_suite()
    
    print("\nDemonstration completed. All visualizations have been saved to the 'plots' directory.")


if __name__ == "__main__":
    main()