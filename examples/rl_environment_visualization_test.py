"""
Test script for the enhanced visualizations in the RL environment.

This script demonstrates the enhanced visualization capabilities for
different problem types in the Branch-and-Bound RL environment.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import os

from src.problems.knapsack import Knapsack, create_predefined_instances as create_knapsack_instances
from src.problems.tsp import TSP, create_predefined_instances as create_tsp_instances
from src.problems.bin_packing import BinPacking, create_predefined_instances as create_bin_packing_instances
from src.environments.branch_and_bound_env import BranchAndBoundEnv


def test_knapsack_visualization():
    """Test the enhanced Knapsack visualization."""
    print("Testing Knapsack visualization...")
    
    # Create a knapsack instance
    instances = create_knapsack_instances()
    problem = instances["medium"]
    
    # Create a sample solution (selecting items)
    solution = np.zeros(problem.n_items)
    solution[[0, 2, 4, 7]] = 1  # Select some items
    
    # Try both normal and RL environment-style visualization
    problem.visualize_instance(title="Knapsack Test Instance")
    
    # Regular visualization
    path = problem.visualize_solution(
        solution, 
        is_optimal=False, 
        title="Knapsack Test Solution"
    )
    print(f"Regular visualization saved to: {path}")
    
    # RL environment-style visualization
    path = problem.visualize_solution(
        solution, 
        is_optimal=False,
        step=42,
        nodes_explored=123,
        elapsed_time=15.7,
        best_obj_value=sum(problem.values * solution),
        animated=True,
        title="Knapsack RL Environment Solution"
    )
    print(f"RL environment visualization saved to: {path}")


def test_tsp_visualization():
    """Test the enhanced TSP visualization."""
    print("Testing TSP visualization...")
    
    # Create a TSP instance
    instances = create_tsp_instances()
    problem = instances["star"]
    
    # Create a sample solution (selecting edges)
    n_vars = problem.n_cities * (problem.n_cities - 1) // 2
    solution = np.zeros(n_vars)
    
    # Set edges for a tour 0-1-2-3-4-0
    def set_edge(i, j, val=1):
        idx = problem._get_variable_index(min(i, j), max(i, j))
        solution[idx] = val
        
    set_edge(0, 1)
    set_edge(1, 2)
    set_edge(2, 3)
    set_edge(3, 4)
    set_edge(4, 0)
    
    # Try both normal and RL environment-style visualization
    problem.visualize_instance(title="TSP Test Instance")
    
    # Regular visualization
    path = problem.visualize_solution(
        solution, 
        is_optimal=False, 
        title="TSP Test Solution"
    )
    print(f"Regular visualization saved to: {path}")
    
    # RL environment-style visualization with subtours
    subtour_solution = solution.copy()
    # Replace edge 4-0 with 3-0 to create subtours
    set_edge(4, 0, 0)
    set_edge(3, 0, 1)
    set_edge(2, 3, 0)
    set_edge(2, 4, 1)
    
    path = problem.visualize_solution(
        subtour_solution, 
        is_optimal=False,
        step=27,
        nodes_explored=85,
        elapsed_time=8.3,
        best_obj_value=-10.5,  # Negative because we maximize -distance
        animated=True,
        title="TSP RL Environment Solution (with subtours)"
    )
    print(f"RL environment visualization with subtours saved to: {path}")


def test_bin_packing_visualization():
    """Test the enhanced Bin Packing visualization."""
    print("Testing Bin Packing visualization...")
    
    # Create a Bin Packing instance
    instances = create_bin_packing_instances()
    problem = instances["medium"]
    
    # Create a sample solution
    n_items = problem.n_items
    max_bins = problem.max_bins
    n_vars = n_items * max_bins + max_bins
    solution = np.zeros(n_vars)
    
    # Extract x_ij and y_j variables
    x_vars = solution[:n_items * max_bins].reshape(n_items, max_bins)
    y_vars = solution[n_items * max_bins:]
    
    # Assign items to bins in a simple way
    # Group 0, 2, 5, 8 in bin 0
    # Group 1, 3, 7 in bin 1
    # Group 4, 6, 9 in bin 2
    assignments = {
        0: [0, 2, 5, 8],
        1: [1, 3, 7],
        2: [4, 6, 9]
    }
    
    # Set variables accordingly
    for bin_idx, items in assignments.items():
        # Mark bin as used
        y_vars[bin_idx] = 1
        
        # Assign items to this bin
        for item in items:
            x_vars[item, bin_idx] = 1
    
    # Flatten back to 1D array
    solution = np.concatenate([x_vars.flatten(), y_vars])
    
    # Try both normal and RL environment-style visualization
    problem.visualize_instance(title="Bin Packing Test Instance")
    
    # Regular visualization
    path = problem.visualize_solution(
        solution, 
        is_optimal=False, 
        title="Bin Packing Test Solution"
    )
    print(f"Regular visualization saved to: {path}")
    
    # RL environment-style visualization
    path = problem.visualize_solution(
        solution, 
        is_optimal=False,
        step=33,
        nodes_explored=97,
        elapsed_time=12.4,
        best_obj_value=3,  # Number of bins used
        animated=True,
        title="Bin Packing RL Environment Solution"
    )
    print(f"RL environment visualization saved to: {path}")


def test_rl_environment_rendering():
    """Test the RL environment rendering with different problem types."""
    print("Testing RL environment rendering...")
    
    # Create problem instances
    knapsack = create_knapsack_instances()["medium"]
    tsp = create_tsp_instances()["star"]
    bin_packing = create_bin_packing_instances()["medium"]
    
    # Create problem generators that return the fixed instances
    def knapsack_generator():
        return knapsack
        
    def tsp_generator():
        return tsp
        
    def bin_packing_generator():
        return bin_packing
    
    # Test with each problem type
    for name, generator in [
        ("Knapsack", knapsack_generator),
        ("TSP", tsp_generator),
        ("BinPacking", bin_packing_generator)
    ]:
        print(f"\nTesting RL environment with {name}...")
        
        # Create the environment
        env = BranchAndBoundEnv(
            problem_generator=generator,
            max_steps=10,
            time_limit=60.0,
            reward_type='improvement',
            observation_type='vector',
            render_mode='human',
            verbose=True
        )
        
        # Reset the environment
        observation, info = env.reset(seed=42)
        print(f"Environment reset with {name} problem")
        print(f"Problem info: {info}")
        
        # Render the initial state
        env.render()
        
        # Take a few random steps
        for i in range(5):
            action = np.random.uniform(-1.0, 1.0, size=(1,))
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: reward = {reward}, terminated = {terminated}, truncated = {truncated}")
            
            # Render after each step
            env.render()
            
            if terminated or truncated:
                break
        
        # Close the environment
        env.close()


def main():
    """Main function to run the visualization tests."""
    # Make sure plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    # Test each visualization type
    test_knapsack_visualization()
    test_tsp_visualization()
    test_bin_packing_visualization()
    
    # Test the RL environment rendering
    test_rl_environment_rendering()
    
    print("\nAll visualization tests completed!")
    print("Check the 'plots' directory for output images.")


if __name__ == "__main__":
    main()