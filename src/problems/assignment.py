"""
Assignment Problem implementation.

This module provides an Assignment Problem class that implements the OptimizationProblem interface,
with utilities for creating, visualizing, and solving Assignment problem instances.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import os
from typing import Dict, List, Tuple, Set, Optional, Any, Union, Callable

from .base import OptimizationProblem


class Assignment(OptimizationProblem):
    """
    Assignment Problem implementation.
    
    In the Assignment problem, we have n agents and n tasks, and each agent-task
    pair has an associated cost. The goal is to assign each agent to exactly one
    task and each task to exactly one agent, minimizing the total cost.
    
    Attributes:
        n_agents (int): Number of agents
        n_tasks (int): Number of tasks
        cost_matrix (np.ndarray): Matrix of costs for agent-task pairs
        problem_name (str): Name of this Assignment instance
        problem_difficulty (str): Difficulty level of this instance
    """
    
    def __init__(self, cost_matrix: np.ndarray, name: str = None, difficulty: str = 'medium'):
        """
        Initialize an Assignment Problem instance.
        
        Args:
            cost_matrix: Matrix where cost_matrix[i,j] is the cost of assigning agent i to task j
            name: Name for this problem instance
            difficulty: Difficulty level for this instance
        """
        self.cost_matrix = np.array(cost_matrix)
        
        # Validate input
        if len(self.cost_matrix.shape) != 2:
            raise ValueError("Cost matrix must be 2-dimensional")
        
        self.n_agents, self.n_tasks = self.cost_matrix.shape
        self._name = name or f"Assignment_{self.n_agents}x{self.n_tasks}"
        self._difficulty = difficulty
        
        # Create a directory for plots if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
    def to_ilp(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert Assignment Problem to ILP formulation.
        
        The ILP formulation uses binary variables x_ij where:
        x_ij = 1 if agent i is assigned to task j, 0 otherwise
        
        With constraints:
        - Each agent is assigned to exactly one task
        - Each task is assigned to exactly one agent
        
        Returns:
            Tuple containing:
            - c: Objective coefficients (costs to minimize, flattened cost matrix)
            - A_eq: Equality constraint matrix (assignment constraints)
            - b_eq: Equality constraint right-hand side (all 1's)
            - A_ub: Inequality constraint matrix (empty)
            - b_ub: Inequality constraint right-hand side (empty)
        """
        n_agents = self.n_agents
        n_tasks = self.n_tasks
        
        # Number of variables: n_agents * n_tasks
        num_vars = n_agents * n_tasks
        
        # Objective: minimize the sum of costs
        c = self.cost_matrix.flatten()
        
        # Equality constraints:
        # 1. Each agent is assigned to exactly one task: sum_j x_ij = 1 for all i
        # 2. Each task is assigned to exactly one agent: sum_i x_ij = 1 for all j
        
        # Initialize constraint matrix: (n_agents + n_tasks) x (n_agents * n_tasks)
        A_eq = np.zeros((n_agents + n_tasks, num_vars))
        
        # Agent constraints: sum_j x_ij = 1 for all i
        for i in range(n_agents):
            for j in range(n_tasks):
                A_eq[i, i * n_tasks + j] = 1
        
        # Task constraints: sum_i x_ij = 1 for all j
        for j in range(n_tasks):
            for i in range(n_agents):
                A_eq[n_agents + j, i * n_tasks + j] = 1
        
        # Right-hand side for equality constraints: all 1's
        b_eq = np.ones(n_agents + n_tasks)
        
        # No inequality constraints
        A_ub = np.zeros((0, num_vars))
        b_ub = np.zeros(0)
        
        return c, A_eq, b_eq, A_ub, b_ub
    
    def validate_solution(self, solution: np.ndarray) -> Tuple[bool, float]:
        """
        Check if a solution is valid and compute its objective value.
        
        A valid assignment solution must:
        1. Be binary (0 or 1 values)
        2. Assign each agent to exactly one task
        3. Assign each task to exactly one agent
        
        Args:
            solution: Binary solution vector (flattened x_ij variables)
            
        Returns:
            Tuple containing:
            - is_valid: Boolean indicating if the solution is valid
            - objective_value: The objective value (total assignment cost)
        """
        # Reshape solution to matrix form
        solution_matrix = solution.reshape(self.n_agents, self.n_tasks)
        
        # Check binary values
        TOLERANCE = 1e-4
        if not all((x < TOLERANCE) or (abs(x - 1) < TOLERANCE) for x in solution):
            return False, 0.0
        
        # Check agent constraints: each agent assigned to exactly one task
        for i in range(self.n_agents):
            if abs(np.sum(solution_matrix[i, :]) - 1.0) > TOLERANCE:
                return False, 0.0
        
        # Check task constraints: each task assigned to exactly one agent
        for j in range(self.n_tasks):
            if abs(np.sum(solution_matrix[:, j]) - 1.0) > TOLERANCE:
                return False, 0.0
        
        # Calculate objective value: sum of assignment costs
        objective_value = np.sum(solution_matrix * self.cost_matrix)
        
        return True, objective_value
    
    def visualize_instance(self, title: str = None, figsize: Tuple[int, int] = (10, 8), **kwargs) -> str:
        """
        Visualize the Assignment Problem instance as a bipartite graph.
        
        Args:
            title: Optional title for the visualization
            figsize: Figure size as (width, height)
            **kwargs: Additional visualization parameters
            
        Returns:
            str: Path to the saved visualization file
        """
        plt.figure(figsize=figsize)
        
        # Create a bipartite graph
        G = nx.DiGraph()
        
        # Add agent and task nodes
        agent_nodes = [f"Agent {i}" for i in range(self.n_agents)]
        task_nodes = [f"Task {j}" for j in range(self.n_tasks)]
        
        # Add nodes with positions
        pos = {}
        
        # Position agents on the left
        for i, agent in enumerate(agent_nodes):
            G.add_node(agent, bipartite=0)
            pos[agent] = (-1, (self.n_agents - 1) / 2 - i)
            
        # Position tasks on the right
        for j, task in enumerate(task_nodes):
            G.add_node(task, bipartite=1)
            pos[task] = (1, (self.n_tasks - 1) / 2 - j)
        
        # Add edges with costs
        edge_labels = {}
        
        # Scale costs for better visualization (if needed)
        vmin, vmax = np.min(self.cost_matrix), np.max(self.cost_matrix)
        norm_costs = (self.cost_matrix - vmin) / (vmax - vmin + 1e-10)
        
        for i in range(self.n_agents):
            for j in range(self.n_tasks):
                cost = self.cost_matrix[i, j]
                weight = 1 + 5 * norm_costs[i, j]  # Scale for visualization
                G.add_edge(agent_nodes[i], task_nodes[j], weight=cost, width=weight)
                edge_labels[(agent_nodes[i], task_nodes[j])] = f"{cost:.1f}"
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=agent_nodes, 
                              node_color='lightblue', 
                              node_size=500)
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=task_nodes, 
                              node_color='lightgreen', 
                              node_size=500)
        
        # Draw edges with varying thickness based on costs
        edges = G.edges(data=True)
        edge_widths = [e[2]['width'] for e in edges]
        edge_colors = ['red' if e[2]['weight'] > np.mean(self.cost_matrix) else 'blue' for e in edges]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, edge_color=edge_colors)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos)
        
        # Draw edge labels (costs)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        # Set title and layout
        plt_title = title or f"Assignment Problem: {self.name}"
        plt.title(plt_title)
        plt.axis('off')
        
        # Save plot
        filename = f"plots/assignment_instance_{self.name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        return filename
   
    def visualize_solution(self, solution: np.ndarray, is_optimal: bool = False, 
                         figsize: Tuple[int, int] = (10, 8), **kwargs) -> str:
        """
        Visualize a solution to the Assignment Problem.
        
        Args:
            solution: Binary solution vector (flattened assignment matrix)
            is_optimal: Whether the solution is optimal
            figsize: Figure size as (width, height)
            **kwargs: Additional visualization parameters
            
        Returns:
            str: Path to the saved visualization file
        """
        plt.figure(figsize=figsize)
        
        # Reshape solution to matrix form
        solution_matrix = solution.reshape(self.n_agents, self.n_tasks)
        
        # Create a bipartite graph
        G = nx.DiGraph()
        
        # Add agent and task nodes
        agent_nodes = [f"Agent {i}" for i in range(self.n_agents)]
        task_nodes = [f"Task {j}" for j in range(self.n_tasks)]
        
        # Add nodes with positions
        pos = {}
        
        # Position agents on the left
        for i, agent in enumerate(agent_nodes):
            G.add_node(agent, bipartite=0)
            pos[agent] = (-1, (self.n_agents - 1) / 2 - i)
            
        # Position tasks on the right
        for j, task in enumerate(task_nodes):
            G.add_node(task, bipartite=1)
            pos[task] = (1, (self.n_tasks - 1) / 2 - j)
        
        # Add all edges with costs
        for i in range(self.n_agents):
            for j in range(self.n_tasks):
                cost = self.cost_matrix[i, j]
                G.add_edge(agent_nodes[i], task_nodes[j], weight=cost, selected=solution_matrix[i, j] > 0.5)
        
        # Collect selected and non-selected edges
        selected_edges = [(u, v) for u, v, d in G.edges(data=True) if d['selected']]
        other_edges = [(u, v) for u, v, d in G.edges(data=True) if not d['selected']]
        
        # Draw the nodes
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=agent_nodes, 
                              node_color='lightblue', 
                              node_size=500)
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=task_nodes, 
                              node_color='lightgreen', 
                              node_size=500)
        
        # Draw non-selected edges (thin, faded)
        nx.draw_networkx_edges(G, pos, 
                              edgelist=other_edges, 
                              width=0.5, 
                              alpha=0.2, 
                              edge_color='gray')
        
        # Draw selected edges (thick, bright)
        nx.draw_networkx_edges(G, pos, 
                              edgelist=selected_edges, 
                              width=3, 
                              alpha=1.0, 
                              edge_color='red')
        
        # Add cost labels for selected edges
        edge_labels = {}
        for i in range(self.n_agents):
            for j in range(self.n_tasks):
                if solution_matrix[i, j] > 0.5:
                    cost = self.cost_matrix[i, j]
                    edge_labels[(agent_nodes[i], task_nodes[j])] = f"{cost:.1f}"
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos)
        
        # Calculate total cost
        total_cost = np.sum(solution_matrix * self.cost_matrix)
        
        # Set title and layout
        status = "Optimal" if is_optimal else "Candidate"
        plt.title(f"{status} Assignment - Total Cost: {total_cost:.1f}")
        plt.axis('off')
        
        # Save plot
        counter = kwargs.get('counter', 1)
        filename = f"plots/{self.name}_solution_{counter:03d}_{status.lower()}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        return filename
    
    @classmethod
    def generate_random_instance(cls, n_agents: int = 5, 
                               min_cost: float = 1.0, 
                               max_cost: float = 100.0,
                               cost_distribution: str = 'uniform',
                               seed: int = None,
                               name: str = None,
                               difficulty: str = 'medium') -> 'Assignment':
        """
        Generate a random Assignment Problem instance.
        
        Args:
            n_agents: Number of agents and tasks (square matrix)
            min_cost: Minimum cost
            max_cost: Maximum cost
            cost_distribution: Distribution of costs ('uniform', 'normal', 'exp')
            seed: Random seed for reproducibility
            name: Name for the instance
            difficulty: Difficulty level
            
        Returns:
            Assignment: A new randomly generated Assignment Problem instance
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Generate costs based on specified distribution
        if cost_distribution == 'uniform':
            costs = np.random.uniform(min_cost, max_cost, (n_agents, n_agents))
        elif cost_distribution == 'normal':
            mean_cost = (min_cost + max_cost) / 2
            std_cost = (max_cost - min_cost) / 6  # ~99.7% of values within range
            costs = np.random.normal(mean_cost, std_cost, (n_agents, n_agents))
            costs = np.clip(costs, min_cost, max_cost)
        elif cost_distribution == 'exp':
            # Exponential distribution with scale adjusted to fit range
            scale = (max_cost - min_cost) / 3  # reasonable scale
            costs = np.random.exponential(scale, (n_agents, n_agents)) + min_cost
            costs = np.minimum(costs, max_cost)  # cap at max_cost
        else:
            raise ValueError(f"Unknown cost distribution: {cost_distribution}")
        
        instance_name = name or f"Random_Assignment_{n_agents}x{n_agents}"
        
        return cls(costs, name=instance_name, difficulty=difficulty)
    
    @classmethod
    def generate_benchmark_suite(cls, difficulty_levels: List[str] = None) -> Dict[str, List['Assignment']]:
        """
        Generate a suite of benchmark Assignment instances at different difficulty levels.
        
        Args:
            difficulty_levels: List of difficulty levels to generate
                             (defaults to ['easy', 'medium', 'hard'])
            
        Returns:
            Dict[str, List[Assignment]]: Dictionary mapping difficulty levels to lists of instances
        """
        if difficulty_levels is None:
            difficulty_levels = ['easy', 'medium', 'hard']
            
        suite = {}
        
        # Define instance parameters for each difficulty level
        configs = {
            'easy': {
                'sizes': [3, 5, 7],
                'distribution': 'uniform',
                'min_cost': 1.0,
                'max_cost': 100.0
            },
            'medium': {
                'sizes': [10, 15, 20],
                'distribution': 'normal',
                'min_cost': 1.0,
                'max_cost': 1000.0
            },
            'hard': {
                'sizes': [30, 50, 75],
                'distribution': 'exp',
                'min_cost': 1.0,
                'max_cost': 10000.0
            }
        }
        
        # Generate instances for each difficulty level
        for level in difficulty_levels:
            if level not in configs:
                continue
                
            suite[level] = []
            config = configs[level]
            
            for size in config['sizes']:
                for i in range(3):  # Generate 3 instances of each size
                    # Use deterministic seed for reproducibility
                    seed = hash(f"{level}_{size}_{i}") % 10000
                    
                    name = f"{level.capitalize()}_Assignment_{size}_{i+1}"
                    
                    instance = cls.generate_random_instance(
                        n_agents=size,
                        min_cost=config['min_cost'],
                        max_cost=config['max_cost'],
                        cost_distribution=config['distribution'],
                        seed=seed,
                        name=name,
                        difficulty=level
                    )
                    suite[level].append(instance)
        
        return suite
    
    def get_constraint_generator(self) -> Optional[Callable]:
        """
        Return a function that generates additional constraints during branch-and-bound.
        
        For Assignment Problem, we don't need lazy constraints, so return None.
        
        Returns:
            None: Assignment doesn't require lazy constraint generation
        """
        return None
    
    @property
    def name(self) -> str:
        """Get the name of this Assignment instance."""
        return self._name
    
    @property
    def size(self) -> Dict[str, int]:
        """Get size metrics for this Assignment instance."""
        return {
            'agents': self.n_agents,
            'tasks': self.n_tasks,
            'variables': self.n_agents * self.n_tasks,
            'constraints': self.n_agents + self.n_tasks
        }
    
    @property
    def difficulty(self) -> str:
        """Get the difficulty level of this Assignment instance."""
        return self._difficulty


def create_predefined_instances() -> Dict[str, Assignment]:
    """
    Create predefined Assignment instances for examples and testing.
    
    Returns:
        Dict[str, Assignment]: Dictionary of named Assignment instances
    """
    instances = {}
    
    # Example 1: 3x3 assignment with simple costs
    costs_3x3 = np.array([
        [10, 20, 15],
        [25, 10, 30],
        [15, 35, 5]
    ])
    instances["small"] = Assignment(costs_3x3, name="Small_Assignment", difficulty="easy")
    
    # Example 2: 4x4 assignment
    costs_4x4 = np.array([
        [9, 22, 18, 15],
        [13, 27, 20, 11],
        [25, 11, 10, 15],
        [14, 24, 16, 13]
    ])
    instances["medium"] = Assignment(costs_4x4, name="Medium_Assignment", difficulty="medium")
    
    # Example 3: 5x5 assignment with structured costs
    # Agents have specialties (lower costs) for certain tasks
    costs_5x5 = np.array([
        [5, 30, 45, 60, 50],   # Agent 0 is good at task 0
        [40, 8, 35, 45, 70],   # Agent 1 is good at task 1
        [35, 40, 10, 55, 65],  # Agent 2 is good at task 2
        [45, 50, 35, 12, 60],  # Agent 3 is good at task 3
        [50, 55, 60, 65, 15]   # Agent 4 is good at task 4
    ])
    instances["structured"] = Assignment(costs_5x5, name="Structured_Assignment", difficulty="medium")
    
    return instances