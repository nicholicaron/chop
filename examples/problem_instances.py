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

from src.problems import (
    OptimizationProblem, TSP, Knapsack, Assignment, BinPacking, SetCover, 
    get_predefined_instances
)
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
    
    # Determine problem type (maximization or minimization)
    if problem_class in [Assignment, BinPacking, SetCover]:
        obj_type = "Minimize"
    else:
        obj_type = "Maximize"
        
    print(f"Objective function: {obj_type}")
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
    
    # Handle objective function direction (our solver maximizes by default)
    # Note: BinPacking and SetCover already have negated costs in their implementations
    objective = c
    if problem_class == Assignment:
        # For assignment problems, we minimize costs but solver maximizes
        objective = -c
    
    solution, obj_value, nodes, optimal_node = solver.solve(
        c=objective, 
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
        # Adjust objective value display for minimization problems
        display_obj = obj_value
        if problem_class == Assignment:
            display_obj = -obj_value  # Convert back to cost (positive)
            
        print(f"Objective value: {display_obj}")
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
    
    # Generate benchmark suites for each problem type
    print("\nGenerating benchmark suites...")
    
    # Only use 'easy' difficulty to keep the demo quick
    difficulty = ['easy']
    
    # Generate TSP benchmark suite
    print("  - TSP benchmark suite...")
    benchmarks['tsp'] = TSP.generate_benchmark_suite(difficulty)
    
    # Generate Knapsack benchmark suite
    print("  - Knapsack benchmark suite...")
    benchmarks['knapsack'] = Knapsack.generate_benchmark_suite(difficulty)
    
    # Generate Assignment benchmark suite
    print("  - Assignment benchmark suite...")
    benchmarks['assignment'] = Assignment.generate_benchmark_suite(difficulty)
    
    # Generate Bin Packing benchmark suite
    print("  - Bin Packing benchmark suite...")
    benchmarks['bin_packing'] = BinPacking.generate_benchmark_suite(difficulty)
    
    # Generate Set Cover benchmark suite
    print("  - Set Cover benchmark suite...")
    benchmarks['set_cover'] = SetCover.generate_benchmark_suite(difficulty)
    
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
    
    # Generate and solve a small instance of each problem type
    
    # TSP instance
    print("\n\n=== GENERATING AND SOLVING A TSP INSTANCE ===")
    tsp_params = {
        'n_cities': 6,
        'seed': 42,
        'name': 'Demo_TSP',
        'difficulty': 'easy'
    }
    tsp_instance, tsp_solution, tsp_obj = generate_and_solve_instance(TSP, tsp_params)
    
    # Knapsack instance
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
    
    # Assignment instance
    print("\n\n=== GENERATING AND SOLVING AN ASSIGNMENT INSTANCE ===")
    assignment_params = {
        'n_agents': 5,
        'cost_distribution': 'uniform',
        'seed': 42,
        'name': 'Demo_Assignment',
        'difficulty': 'easy'
    }
    assignment_instance, assignment_solution, assignment_obj = generate_and_solve_instance(
        Assignment, assignment_params
    )
    
    # Bin Packing instance
    print("\n\n=== GENERATING AND SOLVING A BIN PACKING INSTANCE ===")
    bin_packing_params = {
        'n_items': 10,
        'bin_capacity': 50.0,
        'min_size': 10.0,
        'max_size': 30.0,
        'seed': 42,
        'name': 'Demo_BinPacking',
        'difficulty': 'easy'
    }
    bin_packing_instance, bin_packing_solution, bin_packing_obj = generate_and_solve_instance(
        BinPacking, bin_packing_params
    )
    
    # Set Cover instance
    print("\n\n=== GENERATING AND SOLVING A SET COVER INSTANCE ===")
    set_cover_params = {
        'n_elements': 8,
        'n_sets': 5,
        'density': 0.4,
        'seed': 42,
        'name': 'Demo_SetCover',
        'difficulty': 'easy'
    }
    set_cover_instance, set_cover_solution, set_cover_obj = generate_and_solve_instance(
        SetCover, set_cover_params
    )
    
    # Generate benchmark suites
    print("\n\n=== GENERATING BENCHMARK SUITES ===")
    benchmark_suites = generate_benchmark_suite()
    
    print("\nDemonstration completed. All visualizations have been saved to the 'plots' directory.")
    print("For more details on each problem type, check the individual example scripts:")
    print("  - tsp_example.py")
    print("  - knapsack_example.py")
    print("  - assignment_example.py")
    print("  - bin_packing_example.py")
    print("  - set_cover_example.py")


if __name__ == "__main__":
    main()