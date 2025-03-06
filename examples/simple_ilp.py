"""
Example script demonstrating the use of the refactored Branch-and-Bound solver
on simple ILP problems.
"""

import numpy as np
import sys
import os
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import BranchAndBoundSolver
from src.strategies import (
    MostFractionalBranching, 
    PseudoCostBranching,
    BestBoundPrioritizer, 
    DecayingBestBoundPrioritizer
)
from src.utils import BnBLogger, Timer


def solve_and_print_results(solver, c, A_ub, b_ub, A_eq=None, b_eq=None, 
                          problem_name="unnamed", visualize=False):
    """
    Helper function to solve ILP and display results.
    
    Args:
        solver (BranchAndBoundSolver): Instance of solver
        c (np.ndarray): Objective coefficients
        A_ub (np.ndarray): Inequality constraint matrix
        b_ub (np.ndarray): Inequality RHS vector
        A_eq (np.ndarray, optional): Equality constraint matrix
        b_eq (np.ndarray, optional): Equality RHS vector
        problem_name (str): Name for the problem
        visualize (bool): Whether to generate visualization
    """
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
    solution = result[0]  # Optimal solution vector
    value = result[1]    # Optimal objective value
    num_nodes_explored = result[2] if len(result) > 2 else None
    optimal_node = result[3] if len(result) > 3 else None
    
    print(f"\nResults for {problem_name}:")
    print(f"Optimal solution: {solution}")
    print(f"Optimal objective value: {value}")
    
    if num_nodes_explored is not None:
        print(f"Number of nodes explored: {num_nodes_explored}")
    
    if optimal_node is not None:
        print(f"Optimal node ID: {optimal_node.id}")
    
    print("\n" + "="*50 + "\n")


def main(visualize):
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    # Create logger
    logger = BnBLogger(log_file='logs/simple_ilp.log')
    
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

    # Example 1: 2 variables, 2 constraints
    c1 = np.array([1, 1])
    A_ub1 = np.array([[-1, 1], [8, 2]])
    b_ub1 = np.array([2, 19])
    solve_and_print_results(
        solver=solver,
        c=c1,
        A_ub=A_ub1,
        b_ub=b_ub1,
        problem_name="Example 1 (2 var, 2 cons)",
        visualize=visualize
    )

    # Example 2: 5 variables, 3 constraints
    c2 = np.array([3, 2, 5, 4, 1])
    A_ub2 = np.array([
        [2, 1, 3, 2, 1],
        [1, 2, 1, 1, 3],
        [1, 1, 2, 3, 1]
    ])
    b_ub2 = np.array([10, 8, 15])
    solve_and_print_results(
        solver=solver,
        c=c2,
        A_ub=A_ub2,
        b_ub=b_ub2,
        problem_name="Example 2 (5 var, 3 cons)",
        visualize=visualize
    )

    # Example 3: 8 variables, 5 constraints
    c3 = np.array([5, 7, 3, 2, 6, 4, 8, 1])
    A_ub3 = np.array([
        [3, 2, 1, 4, 2, 5, 1, 3],
        [1, 3, 2, 1, 4, 3, 2, 1],
        [2, 1, 4, 3, 1, 2, 3, 2],
        [4, 3, 2, 1, 3, 1, 2, 4],
        [1, 2, 3, 4, 2, 1, 3, 2]
    ])
    b_ub3 = np.array([20, 25, 30, 22, 18])
    solve_and_print_results(
        solver=solver,
        c=c3,
        A_ub=A_ub3,
        b_ub=b_ub3,
        problem_name="Example 3 (8 var, 5 cons)",
        visualize=visualize
    )

    # Compare different strategies
    print("\nComparing different strategies for Example 3...")
    
    strategies = [
        ("BestBound", BestBoundPrioritizer(), MostFractionalBranching()),
        ("DecayingBestBound", DecayingBestBoundPrioritizer(beta=0.05), MostFractionalBranching()),
        ("PseudoCost", BestBoundPrioritizer(), PseudoCostBranching())
    ]
    
    for name, prioritizer, branching in strategies:
        print(f"\nStrategy: {name}")
        
        strategy_logger = BnBLogger(log_file=f'logs/simple_ilp_{name}.log')
        
        solver = BranchAndBoundSolver(
            prioritizer=prioritizer,
            branching_strategy=branching,
            logger=strategy_logger,
            early_stop_gap=1e-4,
            use_cuts=True,
            cut_probability=0.3
        )
        
        with Timer() as timer:
            result = solver.solve(
                c=c3,
                A_ub=A_ub3,
                b_ub=b_ub3,
                problem_name=f"Example 3 - {name}",
                visualize=False
            )
        
        print(f"  Time: {timer.elapsed:.3f}s")
        print(f"  Nodes: {result[2]}")
        print(f"  Objective: {result[1]}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ILP Solver with Branch and Bound")
    parser.add_argument("--visualize", action="store_true", help="Generate and save plots")
    args = parser.parse_args()
    
    print("Starting branch and bound solver...")
    main(args.visualize)