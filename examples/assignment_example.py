"""
Example script demonstrating Assignment Problem solving with the Branch-and-Bound solver.

This script shows how to:
1. Create Assignment Problem instances with different configurations
2. Solve them using the Branch-and-Bound solver
3. Visualize the solutions as bipartite graphs
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(".")  # Add the project root to the path

from src.problems import Assignment
from src.core.solver import BranchAndBoundSolver
from src.utils.logging import BnBLogger, Timer


def solve_assignment_instance(assignment_instance, solver, visualize=True):
    """
    Solve an Assignment Problem instance using the Branch-and-Bound solver.
    
    Args:
        assignment_instance: The Assignment instance to solve
        solver: The BranchAndBoundSolver instance
        visualize: Whether to generate visualizations
        
    Returns:
        Tuple of (solution, objective_value, node_count, elapsed_time)
    """
    # Convert instance to ILP form
    c, A_eq, b_eq, A_ub, b_ub = assignment_instance.to_ilp()
    
    # Create a callback function for solution visualization
    def callback_fn(solution, is_optimal, problem_name):
        assignment_instance.visualize_solution(solution, is_optimal)
    
    # Solve the instance
    with Timer() as timer:
        result = solver.solve(
            c=-c,  # Negate costs since we minimize costs but solver maximizes
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            problem_name=assignment_instance.name,
            visualize=visualize,
            callback=callback_fn
        )
    
    # Unpack the result
    solution, obj_value, node_count, _ = result
    
    # Calculate solution details if solution exists
    if solution is not None:
        # Validate the solution
        is_valid, actual_obj = assignment_instance.validate_solution(solution)
        
        # Find the assignments
        solution_matrix = solution.reshape(assignment_instance.n_agents, assignment_instance.n_tasks)
        assignments = []
        for i in range(assignment_instance.n_agents):
            for j in range(assignment_instance.n_tasks):
                if solution_matrix[i, j] > 0.5:
                    assignments.append((i, j, assignment_instance.cost_matrix[i, j]))
        
        print(f"\nResults for {assignment_instance.name}:")
        print(f"  Time: {timer.elapsed:.3f}s")
        print(f"  Nodes explored: {node_count}")
        print(f"  Total cost: {actual_obj:.2f}")
        print(f"  Assignments:")
        for i, j, cost in assignments:
            print(f"    Agent {i} → Task {j}: Cost {cost:.2f}")
        print(f"  Solution validation: {'Valid' if is_valid else 'Invalid'}")
    else:
        print(f"\nNo solution found for {assignment_instance.name}")
    
    return solution, -obj_value, node_count, timer.elapsed  # Negate objective back to cost


def main():
    """Main function demonstrating Assignment Problem solving with the problem framework."""
    
    # Create directories for logs and plots
    os.makedirs('logs', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Create logger
    logger = BnBLogger(log_file='logs/assignment_example.log')
    
    # Create solver instance with default settings
    solver = BranchAndBoundSolver(
        logger=logger,
        max_nodes=1000,
        early_stop_gap=1e-6,
        use_cuts=True
    )
    
    # Load predefined Assignment instances
    print("Loading predefined Assignment instances...")
    from src.problems import assignment_instances
    predefined = assignment_instances()
    
    # Visualize predefined instances
    for name, instance in predefined.items():
        print(f"Visualizing {instance.name}...")
        instance.visualize_instance()
    
    # Generate a random instance
    random_assignment = Assignment.generate_random_instance(
        n_agents=8,
        cost_distribution='normal',
        seed=42,
        name="Random_Assignment_8x8",
        difficulty="medium"
    )
    random_assignment.visualize_instance()
    
    # Solve each instance
    instances = list(predefined.values()) + [random_assignment]
    
    for instance in instances:
        print(f"\nSolving {instance.name} with {instance.n_agents} agents and {instance.n_tasks} tasks")
        
        solve_assignment_instance(instance, solver)
        print("\n" + "="*50 + "\n")
    
    print("\nAll Assignment instances solved. Check the 'plots' directory for visualizations.")
    
    # Generate a benchmark suite with different difficulty levels
    print("\nGenerating benchmark suite of Assignment instances...")
    benchmark_suite = Assignment.generate_benchmark_suite(['easy', 'medium'])
    
    # Print information about the benchmark suite
    for difficulty, instances in benchmark_suite.items():
        print(f"\n{difficulty.capitalize()} Assignment instances:")
        for instance in instances:
            print(f"  - {instance.name}: {instance.n_agents}x{instance.n_tasks} matrix")
            
            # Visualize one instance per difficulty level as an example
            if instances.index(instance) == 0:
                instance.visualize_instance()
                
                # Solve the first instance of each difficulty level as an example
                print(f"\nSolving {instance.name} as an example...")
                solve_assignment_instance(instance, solver)
    
    # Advanced example: Create an instance with structured costs
    # In this example, agents have specific skills for certain tasks
    print("\n=== Advanced Example: Assignment with Structured Costs ===")
    n = 6  # 6 agents and 6 tasks
    
    # Initialize cost matrix with high base costs
    costs = np.ones((n, n)) * 50
    
    # Assign low costs for certain agent-task pairs (agent skills)
    for i in range(n):
        # Each agent is skilled in 2 tasks
        skill_tasks = [i, (i+1) % n]  # Agent i is skilled in task i and i+1
        for j in skill_tasks:
            costs[i, j] = np.random.uniform(5, 15)  # Low cost for skills
    
    # Create the instance
    skilled_assignment = Assignment(costs, name="Skilled_Assignment", difficulty="medium")
    skilled_assignment.visualize_instance(title="Assignment with Agent Skills")
    
    # Solve the instance
    print(f"\nSolving Skilled Assignment with {n} agents and {n} tasks")
    solve_assignment_instance(skilled_assignment, solver)


if __name__ == "__main__":
    print("Starting Assignment Problem solver with the problem framework...")
    main()