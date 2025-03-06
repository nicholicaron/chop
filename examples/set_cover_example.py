"""
Example script demonstrating Set Cover Problem solving with the Branch-and-Bound solver.

This script shows how to:
1. Create Set Cover Problem instances with different configurations
2. Solve them using the Branch-and-Bound solver
3. Visualize the solutions showing which sets are selected to cover all elements
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(".")  # Add the project root to the path

from src.problems import SetCover
from src.core.solver import BranchAndBoundSolver
from src.utils.logging import BnBLogger, Timer


def solve_set_cover_instance(set_cover_instance, solver, visualize=True):
    """
    Solve a Set Cover Problem instance using the Branch-and-Bound solver.
    
    Args:
        set_cover_instance: The SetCover instance to solve
        solver: The BranchAndBoundSolver instance
        visualize: Whether to generate visualizations
        
    Returns:
        Tuple of (solution, objective_value, node_count, elapsed_time)
    """
    # Convert instance to ILP form
    c, A_eq, b_eq, A_ub, b_ub = set_cover_instance.to_ilp()
    
    # Create a callback function for solution visualization
    def callback_fn(solution, is_optimal, problem_name):
        set_cover_instance.visualize_solution(solution, is_optimal)
    
    # Solve the instance
    with Timer() as timer:
        result = solver.solve(
            c=c,  # Note: c already has negated costs for minimization
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            problem_name=set_cover_instance.name,
            visualize=visualize,
            callback=callback_fn
        )
    
    # Unpack the result
    solution, obj_value, node_count, _ = result
    
    # Calculate solution details if solution exists
    if solution is not None:
        # Validate the solution
        is_valid, actual_obj = set_cover_instance.validate_solution(solution)
        
        # Calculate statistics
        selected_sets = np.where(solution > 0.5)[0]
        coverage_redundancy = 0
        
        # Calculate coverage redundancy (how many extra times elements are covered)
        for i in range(set_cover_instance.n_elements):
            covers = sum(set_cover_instance.coverage_matrix[i, j] for j in selected_sets)
            redundancy = max(0, covers - 1)  # Extra covers beyond the required one
            coverage_redundancy += redundancy
        
        print(f"\nResults for {set_cover_instance.name}:")
        print(f"  Time: {timer.elapsed:.3f}s")
        print(f"  Nodes explored: {node_count}")
        print(f"  Sets selected: {len(selected_sets)} of {set_cover_instance.n_sets}")
        print(f"  Total cost: {actual_obj:.2f}")
        print(f"  Coverage redundancy: {coverage_redundancy}")
        print(f"  Solution validation: {'Valid' if is_valid else 'Invalid'}")
        
        # Print specific sets selected
        print(f"  Selected sets: {sorted(selected_sets.tolist())}")
    else:
        print(f"\nNo solution found for {set_cover_instance.name}")
    
    return solution, obj_value, node_count, timer.elapsed


def main():
    """Main function demonstrating Set Cover Problem solving with the problem framework."""
    
    # Create directories for logs and plots
    os.makedirs('logs', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Create logger
    logger = BnBLogger(log_file='logs/set_cover_example.log')
    
    # Create solver instance with default settings
    solver = BranchAndBoundSolver(
        logger=logger,
        max_nodes=1000,
        early_stop_gap=1e-6,
        use_cuts=True
    )
    
    # Load predefined Set Cover instances
    print("Loading predefined Set Cover instances...")
    from src.problems import set_cover_instances
    predefined = set_cover_instances()
    
    # Visualize predefined instances
    for name, instance in predefined.items():
        print(f"Visualizing {instance.name}...")
        instance.visualize_instance()
    
    # Generate a random instance
    random_set_cover = SetCover.generate_random_instance(
        n_elements=12,
        n_sets=8,
        density=0.3,
        min_cost=1.0,
        max_cost=10.0,
        ensure_feasible=True,
        seed=42,
        name="Random_SetCover_12x8",
        difficulty="medium"
    )
    random_set_cover.visualize_instance()
    
    # Solve each instance
    instances = list(predefined.values()) + [random_set_cover]
    
    for instance in instances:
        print(f"\nSolving {instance.name} with {instance.n_elements} elements and {instance.n_sets} sets")
        
        solve_set_cover_instance(instance, solver)
        print("\n" + "="*50 + "\n")
    
    print("\nAll Set Cover instances solved. Check the 'plots' directory for visualizations.")
    
    # Generate a benchmark suite with different difficulty levels
    print("\nGenerating benchmark suite of Set Cover instances...")
    benchmark_suite = SetCover.generate_benchmark_suite(['easy'])  # Just 'easy' for quick demo
    
    # Print information about the benchmark suite
    for difficulty, instances in benchmark_suite.items():
        print(f"\n{difficulty.capitalize()} Set Cover instances:")
        for instance in instances:
            print(f"  - {instance.name}: {instance.n_elements} elements, {instance.n_sets} sets")
            
            # Visualize the first instance of each size as an example
            n_elements = instance.n_elements
            if n_elements not in [prev.n_elements for prev in instances[:instances.index(instance)]]:
                instance.visualize_instance()
                
                # Solve the first instance of each size as an example
                print(f"\nSolving {instance.name} as an example...")
                solve_set_cover_instance(instance, solver)
    
    # Advanced example: Create a structured Set Cover instance
    print("\n=== Advanced Example: Structured Set Cover ===")
    
    # Create a structured instance where we know the optimal solution
    # 12 elements divided into 4 groups of 3, and 7 sets:
    # - 4 sets that each cover one group exactly (sets 0-3) - cost 3 each
    # - 2 sets that each cover two groups (sets 4-5) - cost 5 each
    # - 1 set that covers all elements (set 6) - cost 10
    n_elements = 12
    n_sets = 7
    
    # Initialize empty coverage matrix
    coverage_matrix = np.zeros((n_elements, n_sets), dtype=int)
    
    # Set 0 covers elements 0-2
    coverage_matrix[0:3, 0] = 1
    
    # Set 1 covers elements 3-5
    coverage_matrix[3:6, 1] = 1
    
    # Set 2 covers elements 6-8
    coverage_matrix[6:9, 2] = 1
    
    # Set 3 covers elements 9-11
    coverage_matrix[9:12, 3] = 1
    
    # Set 4 covers elements 0-5 (groups 1 and 2)
    coverage_matrix[0:6, 4] = 1
    
    # Set 5 covers elements 6-11 (groups 3 and 4)
    coverage_matrix[6:12, 5] = 1
    
    # Set 6 covers all elements
    coverage_matrix[:, 6] = 1
    
    # Set costs - individual group sets cost 3 each, combined sets cost 5 each, universal set costs 10
    set_costs = np.array([3.0, 3.0, 3.0, 3.0, 5.0, 5.0, 10.0])
    
    structured_set_cover = SetCover(
        coverage_matrix=coverage_matrix,
        set_costs=set_costs,
        name="Structured_SetCover",
        difficulty="medium"
    )
    
    structured_set_cover.visualize_instance(title="Structured Set Cover")
    
    # Solve the instance
    print(f"\nSolving Structured Set Cover instance with {structured_set_cover.n_elements} elements and {structured_set_cover.n_sets} sets")
    print("Optimal solution should be either the 4 individual sets (total cost: 12) or the 2 combined sets (total cost: 10)")
    solve_set_cover_instance(structured_set_cover, solver)


if __name__ == "__main__":
    print("Starting Set Cover Problem solver with the problem framework...")
    main()