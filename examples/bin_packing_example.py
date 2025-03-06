"""
Example script demonstrating Bin Packing Problem solving with the Branch-and-Bound solver.

This script shows how to:
1. Create Bin Packing Problem instances with different configurations
2. Solve them using the Branch-and-Bound solver
3. Visualize the solutions showing how items are packed into bins
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(".")  # Add the project root to the path

from src.problems import BinPacking
from src.core.solver import BranchAndBoundSolver
from src.utils.logging import BnBLogger, Timer


def solve_bin_packing_instance(bin_packing_instance, solver, visualize=True):
    """
    Solve a Bin Packing Problem instance using the Branch-and-Bound solver.
    
    Args:
        bin_packing_instance: The BinPacking instance to solve
        solver: The BranchAndBoundSolver instance
        visualize: Whether to generate visualizations
        
    Returns:
        Tuple of (solution, objective_value, node_count, elapsed_time)
    """
    # Convert instance to ILP form
    c, A_eq, b_eq, A_ub, b_ub = bin_packing_instance.to_ilp()
    
    # Create a callback function for solution visualization
    def callback_fn(solution, is_optimal, problem_name):
        bin_packing_instance.visualize_solution(solution, is_optimal)
    
    # Solve the instance
    with Timer() as timer:
        result = solver.solve(
            c=c,  # Note: c is already negated for bin minimization
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            problem_name=bin_packing_instance.name,
            visualize=visualize,
            callback=callback_fn
        )
    
    # Unpack the result
    solution, obj_value, node_count, _ = result
    
    # Calculate solution details if solution exists
    if solution is not None:
        # Validate the solution
        is_valid, actual_obj = bin_packing_instance.validate_solution(solution)
        
        # Extract solution details
        n_items = bin_packing_instance.n_items
        max_bins = bin_packing_instance.max_bins
        x_vars = solution[:n_items * max_bins].reshape(n_items, max_bins)
        y_vars = solution[n_items * max_bins:]
        
        # Count used bins
        bins_used = int(np.sum(y_vars > 0.5))
        
        # Calculate bin utilization
        bin_loads = np.zeros(max_bins)
        for j in range(max_bins):
            bin_loads[j] = sum(bin_packing_instance.item_sizes[i] * x_vars[i, j] 
                             for i in range(n_items))
        
        used_bin_loads = [bin_loads[j] for j in range(max_bins) if y_vars[j] > 0.5]
        avg_utilization = np.mean(used_bin_loads) / bin_packing_instance.bin_capacity * 100
        
        print(f"\nResults for {bin_packing_instance.name}:")
        print(f"  Time: {timer.elapsed:.3f}s")
        print(f"  Nodes explored: {node_count}")
        print(f"  Bins used: {bins_used} of {max_bins}")
        print(f"  Average bin utilization: {avg_utilization:.1f}%")
        print(f"  Solution validation: {'Valid' if is_valid else 'Invalid'}")
    else:
        print(f"\nNo solution found for {bin_packing_instance.name}")
    
    return solution, obj_value, node_count, timer.elapsed


def main():
    """Main function demonstrating Bin Packing Problem solving with the problem framework."""
    
    # Create directories for logs and plots
    os.makedirs('logs', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Create logger
    logger = BnBLogger(log_file='logs/bin_packing_example.log')
    
    # Create solver instance with default settings
    solver = BranchAndBoundSolver(
        logger=logger,
        max_nodes=1000,
        early_stop_gap=1e-6,
        use_cuts=True
    )
    
    # Load predefined Bin Packing instances
    print("Loading predefined Bin Packing instances...")
    from src.problems import bin_packing_instances
    predefined = bin_packing_instances()
    
    # Visualize predefined instances
    for name, instance in predefined.items():
        print(f"Visualizing {instance.name}...")
        instance.visualize_instance()
    
    # Generate a random instance
    random_bin_packing = BinPacking.generate_random_instance(
        n_items=15,
        bin_capacity=50.0,
        min_size=5.0,
        max_size=25.0,
        size_distribution='uniform',
        seed=42,
        name="Random_BinPacking_15",
        difficulty="medium"
    )
    random_bin_packing.visualize_instance()
    
    # Solve each instance
    instances = list(predefined.values()) + [random_bin_packing]
    
    for instance in instances:
        print(f"\nSolving {instance.name} with {instance.n_items} items and bin capacity {instance.bin_capacity}")
        
        solve_bin_packing_instance(instance, solver)
        print("\n" + "="*50 + "\n")
    
    print("\nAll Bin Packing instances solved. Check the 'plots' directory for visualizations.")
    
    # Generate a benchmark suite with different difficulty levels
    print("\nGenerating benchmark suite of Bin Packing instances...")
    benchmark_suite = BinPacking.generate_benchmark_suite(['easy'])  # Just 'easy' for quick demo
    
    # Print information about the benchmark suite
    for difficulty, instances in benchmark_suite.items():
        print(f"\n{difficulty.capitalize()} Bin Packing instances:")
        for instance in instances:
            print(f"  - {instance.name}: {instance.n_items} items, capacity: {instance.bin_capacity}")
            
            # Visualize the first instance of each size as an example
            n_items = instance.n_items
            if n_items not in [prev.n_items for prev in instances[:instances.index(instance)]]:
                instance.visualize_instance()
                
                # Solve the first instance of each size as an example
                print(f"\nSolving {instance.name} as an example...")
                solve_bin_packing_instance(instance, solver)
    
    # Advanced example: Create a bin packing instance with items that fit perfectly
    print("\n=== Advanced Example: Perfect Bin Packing ===")
    
    # Create item sizes that perfectly fit into bins of capacity 100
    item_sizes = np.array([
        50, 50,           # Bin 1: 50 + 50 = 100
        30, 30, 40,       # Bin 2: 30 + 30 + 40 = 100
        25, 25, 25, 25,   # Bin 3: 25 + 25 + 25 + 25 = 100
        10, 10, 20, 20, 40, # Bin 4: 10 + 10 + 20 + 20 + 40 = 100
        60, 40            # Bin 5: 60 + 40 = 100
    ])
    
    perfect_packing = BinPacking(
        item_sizes=item_sizes,
        bin_capacity=100.0,
        max_bins=10,
        name="Perfect_BinPacking",
        difficulty="medium"
    )
    
    perfect_packing.visualize_instance(title="Perfect Bin Packing (100% Utilization Possible)")
    
    # Solve the instance
    print(f"\nSolving Perfect Bin Packing instance with {perfect_packing.n_items} items")
    solve_bin_packing_instance(perfect_packing, solver)


if __name__ == "__main__":
    print("Starting Bin Packing Problem solver with the problem framework...")
    main()