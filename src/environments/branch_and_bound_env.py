"""
Reinforcement Learning environment for Branch-and-Bound optimization.

This module provides a Gymnasium-compatible environment for training
reinforcement learning agents to optimize the ordering of the priority queue
in Branch-and-Bound solvers for faster convergence to optimal solutions.
"""

import numpy as np
import time
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any, Callable
import torch
from torch_geometric.data import Data

from src.core.solver import BranchAndBoundSolver
from src.core.node import Node
from src.core.priority_queue import PriorityQueue
from src.strategies.priority_queue import NodePrioritizer
from src.problems.base import OptimizationProblem
from src.utils.logging import BnBLogger


class RLPrioritizer(NodePrioritizer):
    """
    A node prioritizer that uses RL agent decisions.
    
    This prioritizer assigns priorities to nodes based on 
    values provided by a reinforcement learning agent.
    """
    
    def __init__(self):
        """Initialize with empty priorities dictionary."""
        self.priorities = {}  # Maps node IDs to priority values
    
    def update_priorities(self, node_priorities: Dict[str, float]):
        """
        Update the priorities based on agent decisions.
        
        Args:
            node_priorities: Dictionary mapping node IDs to priority values
        """
        self.priorities.update(node_priorities)
    
    def get_priority_key(self, node: Node) -> Tuple:
        """
        Get the priority key for a node based on agent-assigned values.
        
        Args:
            node: The node to prioritize
            
        Returns:
            tuple: (priority_value,) as the priority key
        """
        # Default to the node's own value if no priority has been assigned
        priority = self.priorities.get(node.id, node.value)
        return (priority,)


class BranchAndBoundEnv(gym.Env):
    """
    A Gymnasium environment for training agents to optimize Branch-and-Bound.
    
    This environment wraps the Branch-and-Bound solver and exposes its
    state as observations, allowing an agent to make decisions that affect
    the priority queue ordering and thus the search trajectory.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array', 'tree']}
    
    def __init__(self, 
                 problem_generator: Callable[[], OptimizationProblem],
                 max_steps: int = 1000,
                 time_limit: float = 60.0,
                 reward_type: str = 'time_based',
                 observation_type: str = 'graph',
                 render_mode: str = None,
                 verbose: bool = False):
        """
        Initialize the environment with a problem generator.
        
        Args:
            problem_generator: A callable that returns a new problem instance
            max_steps: Maximum number of steps before truncation
            time_limit: Maximum time in seconds before truncation
            reward_type: Type of reward function ('time_based' or 'improvement')
            observation_type: Type of state representation ('graph' or 'vector')
            render_mode: Mode for rendering ('human', 'rgb_array', or 'tree')
            verbose: Whether to print detailed information
        """
        super().__init__()
        
        # Store configuration
        self.problem_generator = problem_generator
        self.max_steps = max_steps
        self.time_limit = time_limit
        self.reward_type = reward_type
        self.observation_type = observation_type
        self.render_mode = render_mode
        self.verbose = verbose
        
        # Environment state
        self.problem = None
        self.solver = None
        self.rl_prioritizer = RLPrioritizer()
        self.steps_taken = 0
        self.start_time = None
        self.best_obj_value = -np.inf
        self.best_solution = None
        self.objective_values = []
        
        # For analysis and visualization
        self.logger = BnBLogger()
        self.history = {
            "rewards": [],
            "obj_values": [],
            "time_elapsed": [],
            "nodes_explored": []
        }
        
        # Define observation and action spaces
        if observation_type == 'vector':
            # Vector representation with fixed-size upper bounds
            # This is a placeholder, actual dimensions would depend on the problem
            self.observation_space = spaces.Dict({
                'tree_features': spaces.Box(low=-np.inf, high=np.inf, shape=(100,), dtype=np.float32),
                'queue_features': spaces.Box(low=-np.inf, high=np.inf, shape=(100,), dtype=np.float32),
                'global_features': spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
            })
        else:  # graph observation
            # For graph observations, we don't define a fixed space since PyTorch Geometric
            # graphs are variable-sized. This is handled specially in gymnasium.
            self.observation_space = None
        
        # Action space defines how the agent reorders the priority queue
        # Direct approach: agent provides values for each resulting subproblem
        self.action_space = spaces.Box(low=-100.0, high=100.0, shape=(1,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment with a new problem instance.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            state: The initial state observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        # Generate a new problem instance
        self.problem = self.problem_generator()
        
        # Convert to ILP form
        c, A_eq, b_eq, A_ub, b_ub = self.problem.to_ilp()
        
        # Reset environment state
        self.steps_taken = 0
        self.start_time = time.time()
        self.best_obj_value = -np.inf
        self.best_solution = None
        self.objective_values = []
        
        # Reset the RL prioritizer
        self.rl_prioritizer = RLPrioritizer()
        
        # Create a new logger
        self.logger = BnBLogger()
        
        # Initialize the solver with our prioritizer
        self.solver = BranchAndBoundSolver(
            prioritizer=self.rl_prioritizer,
            logger=self.logger,
            max_nodes=self.max_steps
        )
        
        # Initialize the root node but don't start solving yet
        self.solver.reset()
        root_node = self.solver._initialize_root_node(c, A_ub, b_ub, A_eq, b_eq)
        
        if root_node is None:
            # The problem is infeasible at the root
            raise ValueError("The problem is infeasible at the root node")
        
        # Add root node to the priority queue
        self.solver.priority_queue.push(root_node)
        
        # Create the initial state observation
        observation = self._get_observation()
        
        # Return initial state and info
        info = {
            'problem_name': self.problem.name,
            'problem_size': self.problem.size,
            'root_relaxation': self.solver.root_relaxation_value
        }
        
        return observation, info
    
    def step(self, action):
        """
        Execute one step of the environment by processing the next node
        according to the priority queue ordering provided by the action.
        
        Args:
            action: Agent's decision for priority values
            
        Returns:
            observation: The new state observation
            reward: The reward for this step
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information dictionary
        """
        # Update step counter
        self.steps_taken += 1
        
        # Apply agent's action to update priorities
        self._apply_action(action)
        
        # Process the next node from the priority queue
        terminated, truncated, node_result = self._process_next_node()
        
        # Get the new state of the environment
        observation = self._get_observation()
        
        # Calculate reward based on the chosen reward function
        reward = self._calculate_reward(node_result)
        
        # Record for history
        self.history["rewards"].append(reward)
        self.history["obj_values"].append(self.best_obj_value)
        self.history["time_elapsed"].append(time.time() - self.start_time)
        self.history["nodes_explored"].append(self.steps_taken)
        
        # Gather additional information
        info = {
            'current_best_obj': self.best_obj_value,
            'nodes_explored': self.steps_taken,
            'time_elapsed': time.time() - self.start_time,
            'queue_size': len(self.solver.priority_queue),
            'node_result': node_result
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the current state of the environment.
        
        The render method supports multiple visualization modes:
        - 'tree': Visualizes the branch-and-bound search tree
        - 'human': Renders a rich problem-specific visualization for human viewing
        - 'rgb_array': Similar to 'human' but returns an image array
        
        Returns:
            str or None: Path to the saved visualization file, or None if no render_mode
        """
        if self.render_mode is None:
            return None
        
        if self.render_mode == 'tree':
            # Visualize the branch-and-bound tree
            tree_path = f"plots/{self.problem.name.replace(' ', '_').lower()}_tree.png"
            self.solver._visualize_tree(self.problem.name)
            return tree_path
        
        elif self.render_mode == 'human' or self.render_mode == 'rgb_array':
            # Rich problem-specific visualization
            render_info = {
                'step': self.steps_taken,
                'nodes_explored': self.solver.node_counter,
                'queue_size': len(self.solver.priority_queue),
                'elapsed_time': time.time() - self.start_time,
                'best_obj_value': self.best_obj_value,
                'is_optimal': not self.solver.priority_queue,  # Queue empty means we're done
                'global_lower_bound': self.solver.global_lower_bound,
                'global_upper_bound': self.solver.global_upper_bound,
                'root_relaxation': self.solver.root_relaxation_value,
                'animated': True,  # Hint to visualization that this is part of a sequence
                'render_mode': self.render_mode
            }
            
            # If there's a solution, visualize it with enhanced information
            if self.best_solution is not None:
                title = f"{self.problem.name} - Step {self.steps_taken}"
                if render_info['is_optimal']:
                    title += " - OPTIMAL"
                else:
                    title += " - Best Solution So Far"
                
                return self.problem.visualize_solution(
                    self.best_solution, 
                    is_optimal=render_info['is_optimal'],
                    title=title,
                    **render_info
                )
            else:
                # Just visualize the problem instance if no solution yet
                return self.problem.visualize_instance(
                    title=f"{self.problem.name} - Step {self.steps_taken} - Exploring",
                    **render_info
                )
    
    def close(self):
        """Clean up resources."""
        if self.logger:
            self.logger.finish()
        
        # Any other cleanup needed can be added here
        pass
    
    def _apply_action(self, action):
        """
        Apply the agent's action to update node priorities.
        
        Args:
            action: Agent's decision for priority values
        """
        # Get current nodes from the priority queue
        current_nodes = self.solver.priority_queue.items()
        
        if not current_nodes:
            # Nothing to do if the queue is empty
            return
        
        # Example implementation: Apply a priority offset based on the action
        # In a more complex implementation, the action could be node-specific
        priorities = {}
        
        # Basic approach for demonstration: just apply the action value as a uniform
        # perturbation to the best bound prioritization
        for node in current_nodes:
            # Use node.value (LP relaxation) and add the action as an offset
            # This allows the agent to bias the search in different directions
            priorities[node.id] = node.value + float(action[0])
        
        # Update the prioritizer with new values
        self.rl_prioritizer.update_priorities(priorities)
        
        # Rebuild the priority queue with these new priorities
        # This is already handled by the PriorityQueue when items are popped
    
    def _process_next_node(self):
        """
        Process the next node from the priority queue.
        
        Returns:
            Tuple: (terminated, truncated, node_result)
                - terminated: Whether the episode is complete
                - truncated: Whether the episode was truncated
                - node_result: Dictionary with information about the processed node
        """
        # Check if we've exceeded our limits
        time_elapsed = time.time() - self.start_time
        if self.steps_taken >= self.max_steps or time_elapsed >= self.time_limit:
            return False, True, {"status": "truncated"}
        
        # Check if the queue is empty (all nodes explored)
        if not self.solver.priority_queue:
            return True, False, {"status": "completed"}
        
        # Get the next node from the priority queue
        try:
            current_node = self.solver.priority_queue.pop()
            
            # Skip if node already processed
            if current_node.id in self.solver.processed_nodes:
                return False, False, {"status": "duplicate"}
            
            # Mark as processed
            self.solver.processed_nodes.add(current_node.id)
            
            # Check if node can be pruned by bound
            if current_node.value <= self.solver.global_lower_bound:
                return False, False, {"status": "pruned_bound"}
            
            # Check for integer solution
            is_integer_solution = all(abs(x - round(x)) < self.solver.tolerance 
                                   for x in current_node.relaxed_soln)
            
            if is_integer_solution:
                # Handle integer solution, potentially updating best bound
                node_result = self._handle_integer_solution(current_node)
            else:
                # Handle fractional solution by branching
                node_result = self._handle_fractional_solution(current_node)
            
            # Update best objective value for tracking
            if self.solver.optimal_obj_value > self.best_obj_value:
                self.best_obj_value = self.solver.optimal_obj_value
                self.best_solution = self.solver.optimal_solution
                
            return False, False, node_result
            
        except IndexError:
            # Empty queue
            return True, False, {"status": "completed"}
    
    def _handle_integer_solution(self, node):
        """
        Process a node with an integer solution.
        
        Args:
            node: Node with integer solution
            
        Returns:
            Dict: Information about the result
        """
        # If there's a TSP-specific constraint generator, handle subtours
        if self.problem.get_constraint_generator() is not None:
            # This would need proper TSP integration; simplified for now
            return {"status": "integer_infeasible"}
            
        # Process the integer solution
        if node.value > self.solver.global_lower_bound:
            # New best solution found
            self.solver.global_lower_bound = node.value
            self.solver.optimal_obj_value = node.value
            self.solver.optimal_solution = node.relaxed_soln
            self.solver.optimal_node = node
            
            # Update visualization attributes
            self.solver._update_node_attributes(node, {
                'color': 'green',
                'relaxed_obj_value': node.value
            })
            
            return {
                "status": "new_best",
                "value": node.value
            }
        else:
            # Integer solution but not better than current best
            return {
                "status": "integer_feasible",
                "value": node.value
            }
    
    def _handle_fractional_solution(self, node):
        """
        Process a node with a fractional solution.
        
        Args:
            node: Node with fractional solution
            
        Returns:
            Dict: Information about the result
        """
        # Get coefficients for the ILP
        c, _, _, _, _ = self.problem.to_ilp()
        
        # Identify fractional variables
        node.indices_frac = [i for i, x in enumerate(node.relaxed_soln) 
                           if abs(x - round(x)) > self.solver.tolerance]
        node.num_frac = len(node.indices_frac)
        node.num_int = len(c) - node.num_frac
        
        # Branch on the node
        self.solver._branch(node, c)
        
        return {
            "status": "branched",
            "num_frac": node.num_frac,
            "value": node.value
        }
    
    def _calculate_reward(self, node_result):
        """
        Calculate the reward based on the chosen reward function.
        
        Args:
            node_result: Dictionary with information about the processed node
            
        Returns:
            float: The calculated reward
        """
        if self.reward_type == 'time_based':
            # Simple time-based reward: negative for each step
            return -1.0
            
        elif self.reward_type == 'improvement':
            # Improvement-based reward
            status = node_result.get("status", "")
            
            if status == "new_best":
                # Big positive reward for finding a better solution
                # Scale reward by the magnitude of improvement
                improvement = node_result["value"] - self.best_obj_value
                return 10.0 + improvement
                
            elif status == "integer_feasible":
                # Small positive reward for finding any feasible solution
                return 1.0
                
            elif status == "pruned_bound" or status == "duplicate":
                # Small negative reward for wasted work
                return -0.5
                
            elif status == "truncated" or status == "completed":
                # Episode ending rewards
                if self.solver.optimal_solution is not None:
                    # Positive reward for finishing with a solution
                    return 5.0
                else:
                    # Negative reward for failing to find any solution
                    return -10.0
            
            # Default small negative reward for each step
            return -0.1
        
        else:
            # Default reward
            return 0.0
    
    def _get_observation(self):
        """
        Get the current state observation.
        
        Returns:
            State observation (graph or vector)
        """
        if self.observation_type == 'graph':
            return self._get_graph_observation()
        else:
            return self._get_vector_observation()
    
    def _get_graph_observation(self):
        """
        Get graph representation of the current state.
        
        Returns:
            torch_geometric.data.Data: Graph representation
        """
        # Create a graph from the branch-and-bound tree
        tree = self.solver.enumeration_tree
        
        # Collect node features
        node_features = []
        node_ids = []
        
        for node_id in tree.nodes():
            node_data = tree.nodes[node_id]
            
            # Extract features for each node
            features = [
                node_data.get('depth', 0),
                node_data.get('relaxed_obj_value', 0.0),
                1.0 if node_data.get('color', '') == 'green' else 0.0,  # Is optimal solution
                1.0 if node_data.get('color', '') == 'lightblue' else 0.0,  # Is integer feasible
                1.0 if node_data.get('color', '') == 'red' else 0.0,  # Is infeasible
                1.0 if node_data.get('color', '') == 'orange' else 0.0,  # Is pruned
                node_data.get('num_int', 0),
                node_data.get('num_frac', 0),
                node_data.get('optimality_gap', float('inf')),
                1.0 if node_id in [node.id for node in self.solver.priority_queue.items()] else 0.0  # In queue
            ]
            
            node_features.append(features)
            node_ids.append(node_id)
        
        # Convert to tensors
        if node_features:
            node_features = torch.tensor(node_features, dtype=torch.float)
        else:
            # Handle empty case
            node_features = torch.zeros((0, 10), dtype=torch.float)
        
        # Create edge index from tree edges
        edge_list = list(tree.edges())
        edge_index = []
        
        for u, v in edge_list:
            # Convert node IDs to indices
            u_idx = node_ids.index(u)
            v_idx = node_ids.index(v)
            
            # Add both directions for undirected graph
            edge_index.append([u_idx, v_idx])
            edge_index.append([v_idx, u_idx])  # For undirected graph
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            # Handle empty case
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Global features (scalar)
        global_features = torch.tensor([
            self.solver.global_lower_bound,  # Best integer objective
            self.solver.global_upper_bound,  # Best possible value
            self.steps_taken,  # Number of steps
            len(self.solver.priority_queue),  # Queue size
            time.time() - self.start_time,  # Elapsed time
            float(self.solver.root_relaxation_value),  # Root relaxation value
        ], dtype=torch.float).unsqueeze(0)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            global_features=global_features,
            num_nodes=len(node_ids)
        )
        
        return data
    
    def _get_vector_observation(self):
        """
        Get vector representation of the current state.
        
        Returns:
            Dict: Vector representation
        """
        # Global features
        global_features = np.array([
            self.solver.global_lower_bound,  # Best integer objective
            self.solver.global_upper_bound,  # Best possible value
            self.steps_taken,  # Number of steps
            len(self.solver.priority_queue),  # Queue size
            time.time() - self.start_time,  # Elapsed time
            float(self.solver.root_relaxation_value),  # Root relaxation value
            self.solver.node_counter,  # Total nodes created
        ], dtype=np.float32)
        
        # Get tree features
        tree = self.solver.enumeration_tree
        tree_features = np.zeros(100, dtype=np.float32)  # Fixed size for simplicity
        
        # Basic tree statistics
        if tree.nodes:
            idx = 0
            tree_features[idx] = len(tree.nodes); idx += 1
            tree_features[idx] = len(tree.edges); idx += 1
            
            # Depth distribution (simplified)
            depths = [tree.nodes[n].get('depth', 0) for n in tree.nodes]
            if depths:
                tree_features[idx] = max(depths); idx += 1
                tree_features[idx] = min(depths); idx += 1
                tree_features[idx] = np.mean(depths); idx += 1
            
            # Number of nodes of each type
            tree_features[idx] = sum(1 for n in tree.nodes if tree.nodes[n].get('color', '') == 'green'); idx += 1
            tree_features[idx] = sum(1 for n in tree.nodes if tree.nodes[n].get('color', '') == 'lightblue'); idx += 1
            tree_features[idx] = sum(1 for n in tree.nodes if tree.nodes[n].get('color', '') == 'red'); idx += 1
            tree_features[idx] = sum(1 for n in tree.nodes if tree.nodes[n].get('color', '') == 'orange'); idx += 1
        
        # Queue features
        queue_items = self.solver.priority_queue.items()
        queue_features = np.zeros(100, dtype=np.float32)  # Fixed size for simplicity
        
        if queue_items:
            # Sort by current priority
            queue_items = sorted(queue_items, key=lambda n: self.rl_prioritizer.get_priority_key(n)[0], reverse=True)
            
            # Extract features from top K items
            K = min(10, len(queue_items))  # Top 10 items
            for i in range(K):
                node = queue_items[i]
                base_idx = i * 10  # 10 features per node
                
                if base_idx + 10 <= 100:  # Ensure we don't exceed array bounds
                    queue_features[base_idx] = node.depth
                    queue_features[base_idx + 1] = node.value
                    queue_features[base_idx + 2] = node.num_int if hasattr(node, 'num_int') else 0
                    queue_features[base_idx + 3] = node.num_frac if hasattr(node, 'num_frac') else 0
                    queue_features[base_idx + 4] = node.optimality_gap if hasattr(node, 'optimality_gap') else float('inf')
                    queue_features[base_idx + 5] = float(node.branch_var is not None)
                    # Additional features could be added here
        
        return {
            'tree_features': tree_features,
            'queue_features': queue_features,
            'global_features': global_features
        }