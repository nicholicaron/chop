"""
Example showing the Branch-and-Bound Reinforcement Learning environment.

This example demonstrates how to create and use the BranchAndBoundEnv for
training RL agents to optimize the priority queue ordering in the
Branch-and-Bound algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time

from src.problems.knapsack import KnapsackProblem
from src.environments.branch_and_bound_env import BranchAndBoundEnv
from src.strategies.priority_queue import BestBoundPrioritizer, DepthFirstPrioritizer


def random_agent(observation):
    """
    Simple random agent that returns a random perturbation.
    
    Args:
        observation: Current state observation
        
    Returns:
        numpy.ndarray: Random action
    """
    return np.random.uniform(-1.0, 1.0, size=(1,))


def best_bound_agent(observation):
    """
    Agent that mimics the best-bound strategy.
    
    Args:
        observation: Current state observation
        
    Returns:
        numpy.ndarray: Action encouraging best-bound behavior (zero perturbation)
    """
    return np.array([0.0])


def depth_first_agent(observation):
    """
    Agent that encourages depth-first behavior.
    
    For vector observations, it biases towards deeper nodes.
    
    Args:
        observation: Current state observation
        
    Returns:
        numpy.ndarray: Action encouraging depth-first behavior
    """
    # Positive values bias towards higher depths
    return np.array([1.0])


def compare_agents(problem_generator, num_episodes=5, max_steps=100):
    """
    Compare different agents on the same set of problems.
    
    Args:
        problem_generator: Function that generates problem instances
        num_episodes: Number of episodes to run for each agent
        max_steps: Maximum steps per episode
        
    Returns:
        dict: Results for each agent
    """
    agents = {
        "Random": random_agent,
        "BestBound": best_bound_agent,
        "DepthFirst": depth_first_agent
    }
    
    # For storing results
    results = {
        agent_name: {
            "steps_to_solve": [],
            "objective_values": [],
            "times": []
        } for agent_name in agents
    }
    
    # Set the same seeds for fair comparison
    seeds = [42 + i for i in range(num_episodes)]
    
    # Run each agent on the same problem instances
    for agent_name, agent_fn in agents.items():
        print(f"\nEvaluating {agent_name} agent...")
        
        for episode in range(num_episodes):
            # Create environment with the same seed
            env = BranchAndBoundEnv(
                problem_generator=problem_generator,
                max_steps=max_steps,
                reward_type='improvement',
                observation_type='vector'
            )
            
            # Reset environment with seed
            observation, info = env.reset(seed=seeds[episode])
            
            # Run episode
            start_time = time()
            done = False
            truncated = False
            total_reward = 0
            steps = 0
            
            while not (done or truncated):
                # Get action from agent
                action = agent_fn(observation)
                
                # Take step in environment
                observation, reward, done, truncated, info = env.step(action)
                
                # Update counters
                total_reward += reward
                steps += 1
            
            # Record results
            episode_time = time() - start_time
            results[agent_name]["steps_to_solve"].append(steps)
            results[agent_name]["objective_values"].append(info["current_best_obj"])
            results[agent_name]["times"].append(episode_time)
            
            print(f"  Episode {episode + 1}: {steps} steps, obj_value={info['current_best_obj']:.2f}, time={episode_time:.3f}s")
    
    # Calculate averages
    for agent_name in agents:
        avg_steps = np.mean(results[agent_name]["steps_to_solve"])
        avg_obj = np.mean(results[agent_name]["objective_values"])
        avg_time = np.mean(results[agent_name]["times"])
        
        print(f"\n{agent_name} Summary:")
        print(f"  Avg Steps: {avg_steps:.2f}")
        print(f"  Avg Objective: {avg_obj:.2f}")
        print(f"  Avg Time: {avg_time:.3f}s")
    
    return results


def plot_results(results):
    """
    Plot comparison results.
    
    Args:
        results: Results dictionary from compare_agents
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot steps
    for i, (metric, title, ylabel) in enumerate([
        ("steps_to_solve", "Steps to Solve", "Steps"),
        ("objective_values", "Objective Values", "Value"),
        ("times", "Solution Times", "Time (s)")
    ]):
        ax = axes[i]
        
        # Get data
        data = []
        labels = []
        
        for agent_name in results:
            data.append(results[agent_name][metric])
            labels.append(agent_name)
        
        # Create boxplot
        ax.boxplot(data, labels=labels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        
        # Add individual points
        for j, d in enumerate(data):
            # Add jitter to x position
            x = np.random.normal(j + 1, 0.05, size=len(d))
            ax.scatter(x, d, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("plots/agent_comparison.png", dpi=300)
    plt.show()


def knapsack_generator():
    """Generate a random knapsack problem instance."""
    return KnapsackProblem.generate_random_instance(num_items=20, difficulty='medium')


def main():
    """Main function to run the example."""
    print("Branch-and-Bound RL Environment Example")
    print("======================================")
    
    # Create problem generator
    problem_generator = knapsack_generator
    
    # Compare different agent strategies
    results = compare_agents(problem_generator, num_episodes=5, max_steps=100)
    
    # Plot the results
    plot_results(results)


if __name__ == "__main__":
    main()