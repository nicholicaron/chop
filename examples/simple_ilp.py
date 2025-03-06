"""
Example script demonstrating the use of the Branch-and-Bound solver
on simple ILP problems.

This script shows how to:
1. Create and solve simple ILP problems directly without problem class definitions
2. Compare different solver strategies
3. Visualize the branch-and-bound tree and solutions
"""

import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.solver import BranchAndBoundSolver
from src.strategies.branching import MostFractionalBranching, PseudoCostBranching
from src.strategies.priority_queue import BestBoundPrioritizer, DecayingBestBoundPrioritizer
from src.utils.logging import BnBLogger, Timer


def solve_and_print_results(solver, c, A_ub, b_ub, A_eq=None, b_eq=None, 
                          problem_name="unnamed", visualize=False):
    """
    Helper function to solve ILP and display results.
    
    Args:
        solver: Instance of BranchAndBoundSolver
        c: Objective coefficients
        A_ub: Inequality constraint matrix
        b_ub: Inequality RHS vector
        A_eq: Equality constraint matrix (optional)
        b_eq: Equality RHS vector (optional)
        problem_name: Name for the problem
        visualize: Whether to generate visualization
        
    Returns:
        Tuple: (solution, objective_value, nodes_explored, optimal_node)
    """
    with Timer() as timer:
        result = solver.solve(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            problem_name=problem_name,
            visualize=visualize
        )
    
    # Unpack the result
    solution, value, num_nodes_explored, optimal_node = result
    
    print(f"\nResults for {problem_name}:")
    print(f"Time: {timer.elapsed:.3f}s")
    
    if solution is not None:
        print(f"Optimal solution: {solution}")
        print(f"Optimal objective value: {value}")
        print(f"Number of nodes explored: {num_nodes_explored}")
        if optimal_node is not None:
            print(f"Optimal node ID: {optimal_node.id}")
    else:
        print("No solution found.")
    
    print("\n" + "="*50 + "\n")
    
    return result, timer.elapsed


def visualize_2d_problem(c, A_ub, b_ub, solution, name="ILP_Problem"):
    """
    Visualize a 2D ILP problem with its solution.
    
    Args:
        c: Objective coefficients [c1, c2]
        A_ub: Inequality constraint matrix
        b_ub: Inequality constraint RHS
        solution: Solution vector [x1, x2]
        name: Problem name for file saving
    """
    # Only works for 2D problems
    if len(c) != 2:
        print("Visualization only works for 2D problems.")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Define the plot boundaries
    x_max = max(10, int(solution[0] * 1.5) if solution is not None else 10)
    y_max = max(10, int(solution[1] * 1.5) if solution is not None else 10)
    x = np.linspace(0, x_max, 1000)
    y = np.linspace(0, y_max, 1000)
    X, Y = np.meshgrid(x, y)
    
    # Plot the feasible region
    inequality_region = np.ones_like(X, dtype=bool)
    
    for i in range(len(b_ub)):
        if A_ub[i, 0] == 0 and A_ub[i, 1] == 0:
            continue  # Skip if both coefficients are zero
            
        if A_ub[i, 0] == 0:  # Constraint on y only
            constraint = Y <= b_ub[i] / A_ub[i, 1] if A_ub[i, 1] > 0 else Y >= b_ub[i] / A_ub[i, 1]
        elif A_ub[i, 1] == 0:  # Constraint on x only
            constraint = X <= b_ub[i] / A_ub[i, 0] if A_ub[i, 0] > 0 else X >= b_ub[i] / A_ub[i, 0]
        else:  # Constraint on both x and y
            if A_ub[i, 1] > 0:
                constraint = Y <= (b_ub[i] - A_ub[i, 0] * X) / A_ub[i, 1]
            else:
                constraint = Y >= (b_ub[i] - A_ub[i, 0] * X) / A_ub[i, 1]
        
        inequality_region = np.logical_and(inequality_region, constraint)
    
    # Add non-negativity constraints
    inequality_region = np.logical_and(inequality_region, X >= 0)
    inequality_region = np.logical_and(inequality_region, Y >= 0)
    
    # Plot the feasible region
    plt.imshow(inequality_region, extent=(0, x_max, 0, y_max), 
              origin='lower', cmap='Blues', alpha=0.3)
    
    # Plot the constraint lines
    for i in range(len(b_ub)):
        if A_ub[i, 0] == 0 and A_ub[i, 1] == 0:
            continue  # Skip if both coefficients are zero
            
        if A_ub[i, 0] == 0:  # Vertical line
            y_val = b_ub[i] / A_ub[i, 1]
            plt.axhline(y=y_val, color='r', linestyle='--', 
                       label=f"{A_ub[i, 1]}y <= {b_ub[i]}")
        elif A_ub[i, 1] == 0:  # Horizontal line
            x_val = b_ub[i] / A_ub[i, 0]
            plt.axvline(x=x_val, color='r', linestyle='--', 
                       label=f"{A_ub[i, 0]}x <= {b_ub[i]}")
        else:
            # General line: a*x + b*y = c => y = (c - a*x) / b
            x_vals = np.linspace(0, x_max, 100)
            y_vals = (b_ub[i] - A_ub[i, 0] * x_vals) / A_ub[i, 1]
            plt.plot(x_vals, y_vals, 'r--', 
                    label=f"{A_ub[i, 0]}x + {A_ub[i, 1]}y = {b_ub[i]}")
    
    # Highlight integer points in the feasible region
    int_x, int_y = np.meshgrid(range(int(x_max) + 1), range(int(y_max) + 1))
    int_points = np.vstack([int_x.flatten(), int_y.flatten()]).T
    
    for point in int_points:
        x, y = point
        # Check if point satisfies all constraints
        satisfies_all = True
        for i in range(len(b_ub)):
            if np.dot(A_ub[i], [x, y]) > b_ub[i]:
                satisfies_all = False
                break
        
        if satisfies_all:
            obj_val = c[0] * x + c[1] * y
            plt.plot(x, y, 'o', color='gray', markersize=5)
    
    # Plot the objective function contour
    Z = c[0] * X + c[1] * Y
    contour = plt.contour(X, Y, Z, 10, colors='green', alpha=0.6)
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Highlight the solution
    if solution is not None:
        plt.plot(solution[0], solution[1], 'r*', markersize=15, label="Optimal Solution")
        plt.annotate(f"({solution[0]}, {solution[1]})\nObj: {np.dot(c, solution)}", 
                    (solution[0], solution[1]), xytext=(10, -20),
                    textcoords="offset points", fontsize=12,
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    # Add plot details
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'ILP Problem: Maximize {c[0]}x + {c[1]}y')
    plt.legend(loc='best')
    
    # Save the plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{name}_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to plots/{name}_visualization.png")


def create_solver(prioritizer=None, branching_strategy=None, use_cuts=True, 
                log_file=None, **kwargs):
    """
    Create a Branch and Bound solver with the specified configuration.
    
    Args:
        prioritizer: Node prioritization strategy
        branching_strategy: Variable branching strategy
        use_cuts: Whether to use Gomory cuts
        log_file: Path to log file
        **kwargs: Additional solver parameters
        
    Returns:
        BranchAndBoundSolver: Configured solver
    """
    # Set default strategies if not provided
    if prioritizer is None:
        prioritizer = DecayingBestBoundPrioritizer(beta=0.05)
    if branching_strategy is None:
        branching_strategy = MostFractionalBranching()
    
    # Create logger
    logger = BnBLogger(log_file=log_file)
    
    # Configure solver
    return BranchAndBoundSolver(
        prioritizer=prioritizer,
        branching_strategy=branching_strategy,
        logger=logger,
        use_cuts=use_cuts,
        **kwargs
    )


def main(visualize):
    """Main function demonstrating various ILP examples."""
    
    # Create directories for logs and plots
    os.makedirs('logs', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    print("Starting ILP examples with the Branch-and-Bound solver...")
    
    # Create default solver
    solver = create_solver(
        log_file='logs/simple_ilp.log',
        early_stop_gap=1e-4,
        use_cuts=True,
        cut_probability=0.3
    )

    # Example 1: Simple 2-variable problem (good for visualization)
    print("\n=== Example 1: Simple 2-variable Problem ===")
    c1 = np.array([3, 4])  # Maximize 3x + 4y
    A_ub1 = np.array([
        [1, 2],   # x + 2y <= 10
        [3, 1],   # 3x + y <= 15
    ])
    b_ub1 = np.array([10, 15])
    
    result1, time1 = solve_and_print_results(
        solver=solver,
        c=c1,
        A_ub=A_ub1,
        b_ub=b_ub1,
        problem_name="Example 1 (2 var, 2 cons)",
        visualize=visualize
    )
    
    # Visualize the 2D problem
    if visualize and result1[0] is not None:
        visualize_2d_problem(c1, A_ub1, b_ub1, result1[0], "Example1")

    # Example 2: 5 variables, 3 constraints
    print("\n=== Example 2: 5-variable Problem ===")
    c2 = np.array([3, 2, 5, 4, 1])
    A_ub2 = np.array([
        [2, 1, 3, 2, 1],
        [1, 2, 1, 1, 3],
        [1, 1, 2, 3, 1]
    ])
    b_ub2 = np.array([10, 8, 15])
    
    result2, time2 = solve_and_print_results(
        solver=solver,
        c=c2,
        A_ub=A_ub2,
        b_ub=b_ub2,
        problem_name="Example 2 (5 var, 3 cons)",
        visualize=visualize
    )

    # Example 3: 8 variables, 5 constraints
    print("\n=== Example 3: 8-variable Problem ===")
    c3 = np.array([5, 7, 3, 2, 6, 4, 8, 1])
    A_ub3 = np.array([
        [3, 2, 1, 4, 2, 5, 1, 3],
        [1, 3, 2, 1, 4, 3, 2, 1],
        [2, 1, 4, 3, 1, 2, 3, 2],
        [4, 3, 2, 1, 3, 1, 2, 4],
        [1, 2, 3, 4, 2, 1, 3, 2]
    ])
    b_ub3 = np.array([20, 25, 30, 22, 18])
    
    result3, time3 = solve_and_print_results(
        solver=solver,
        c=c3,
        A_ub=A_ub3,
        b_ub=b_ub3,
        problem_name="Example 3 (8 var, 5 cons)",
        visualize=visualize
    )

    # Compare different strategies on the 8-variable problem
    print("\n=== Strategy Comparison on Example 3 ===")
    
    strategies = [
        ("BestBound", BestBoundPrioritizer(), MostFractionalBranching()),
        ("DecayingBestBound", DecayingBestBoundPrioritizer(beta=0.05), MostFractionalBranching()),
        ("PseudoCost", BestBoundPrioritizer(), PseudoCostBranching())
    ]
    
    comparison_results = []
    
    for name, prioritizer, branching in strategies:
        print(f"\nEvaluating strategy: {name}")
        
        # Create solver with this strategy
        strategy_solver = create_solver(
            prioritizer=prioritizer,
            branching_strategy=branching,
            log_file=f'logs/simple_ilp_{name}.log',
            early_stop_gap=1e-4,
            use_cuts=True,
            cut_probability=0.3
        )
        
        # Solve the problem
        with Timer() as timer:
            result = strategy_solver.solve(
                c=c3,
                A_ub=A_ub3,
                b_ub=b_ub3,
                problem_name=f"Example 3 - {name}",
                visualize=False
            )
        
        solution, obj_value, nodes, _ = result
        
        print(f"  Time: {timer.elapsed:.3f}s")
        print(f"  Nodes explored: {nodes}")
        print(f"  Objective value: {obj_value}")
        
        comparison_results.append({
            "strategy": name,
            "time": timer.elapsed,
            "nodes": nodes,
            "objective": obj_value
        })
    
    # Compare cutting planes vs. no cutting planes
    print("\n=== Cutting Planes Comparison on Example 3 ===")
    
    # Create solver without cuts
    no_cuts_solver = create_solver(
        log_file='logs/simple_ilp_no_cuts.log',
        early_stop_gap=1e-4,
        use_cuts=False
    )
    
    # Solve with no cuts
    with Timer() as timer:
        result = no_cuts_solver.solve(
            c=c3,
            A_ub=A_ub3,
            b_ub=b_ub3,
            problem_name="Example 3 - No Cuts",
            visualize=False
        )
    
    no_cuts_solution, no_cuts_obj, no_cuts_nodes, _ = result
    
    print(f"No Cuts Strategy:")
    print(f"  Time: {timer.elapsed:.3f}s")
    print(f"  Nodes explored: {no_cuts_nodes}")
    print(f"  Objective value: {no_cuts_obj}")
    
    # Summarize comparison
    print("\n=== Strategy Comparison Summary ===")
    print(f"{'Strategy':<20} {'Time (s)':<10} {'Nodes':<10} {'Objective':<10}")
    print("-" * 50)
    
    for res in comparison_results:
        print(f"{res['strategy']:<20} {res['time']:<10.3f} {res['nodes']:<10} {res['objective']:<10}")
    
    print(f"{'No Cuts':<20} {timer.elapsed:<10.3f} {no_cuts_nodes:<10} {no_cuts_obj:<10}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ILP Solver with Branch and Bound")
    parser.add_argument("--visualize", action="store_true", help="Generate and save plots")
    args = parser.parse_args()
    
    main(args.visualize)