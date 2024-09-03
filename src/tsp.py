"""
Just playing around with the PuLP library -- feel free to disregard this file
"""

from pulp import *
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional

def create_graph_from_adjacency_matrix(adj_matrix: List[List[float]]) -> nx.Graph:
    """
    Create a networkx graph from an adjacency matrix.
    
    Args:
        adj_matrix (List[List[float]]): The adjacency matrix representing the TSP instance.
    
    Returns:
        nx.Graph: A networkx graph representing the TSP instance.
    """
    G = nx.Graph()
    n = len(adj_matrix)
    for i in range(n):
        for j in range(i+1, n):
            G.add_edge(i, j, weight=adj_matrix[i][j])
    return G

def draw_tsp_graph(G: nx.Graph, pos: Dict[int, np.ndarray], solution: Optional[Dict[str, int]] = None) -> None:
    """
    Draw the TSP graph and highlight the solution if provided.
    
    Args:
        G (nx.Graph): The networkx graph representing the TSP instance.
        pos (Dict[int, np.ndarray]): The positions of the nodes for drawing.
        solution (Optional[Dict[str, int]]): The current solution, if any.
    """
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos)
    
    edge_weights = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)
    
    if solution:
        solution_edges = [(int(k.split('_')[1]), int(k.split('_')[2])) for k, v in solution.items() if v == 1]
        nx.draw_networkx_edges(G, pos, edgelist=solution_edges, edge_color='r', width=2)
    else:
        nx.draw_networkx_edges(G, pos, width=1)

def pretty_print_tsp(adj_matrix: List[List[float]], solution: Optional[Dict[str, int]] = None) -> None:
    """
    Create a pretty print visualization of the TSP instance and current solution if provided.
    
    Args:
        adj_matrix (List[List[float]]): The adjacency matrix representing the TSP instance.
        solution (Optional[Dict[str, int]]): The current solution, if any.
    """
    G = create_graph_from_adjacency_matrix(adj_matrix)
    pos = nx.spring_layout(G)
    draw_tsp_graph(G, pos, solution)
    plt.title("TSP Instance" + (" with Current Solution" if solution else ""))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    adj_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
pretty_print_tsp(adj_matrix)
    
# Example with a solution
solution = {'x_0_1': 1, 'x_1_2': 1, 'x_2_3': 1, 'x_3_0': 1}
pretty_print_tsp(adj_matrix, solution)



def formulate_tsp(adj_matrix):
    n = len(adj_matrix)
    
    # Create the model
    model = pulp.LpProblem("TSP", pulp.LpMinimize)
    
    # Create binary variables
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(n) if i != j),
                              cat='Binary')
    
    # Create auxiliary variables for subtour elimination
    u = pulp.LpVariable.dicts("u", (i for i in range(1, n)), lowBound=0, upBound=n-1, cat='Integer')
    
    # Objective function
    model += pulp.lpSum(adj_matrix[i][j] * x[(i, j)] for i in range(n) for j in range(n) if i != j)
    
    # Add constraints
    # Each city must be entered exactly once
    for j in range(n):
        model += pulp.lpSum(x[(i, j)] for i in range(n) if i != j) == 1
    
    # Each city must be exited exactly once
    for i in range(n):
        model += pulp.lpSum(x[(i, j)] for j in range(n) if i != j) == 1
    
    # Subtour elimination (MTZ formulation)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model += u[i] - u[j] + n * x[(i, j)] <= n - 1
    
    return model

def create_lp_relaxation(ilp_model):
    # Create a deep copy of the ILP model
    lp_model = ilp_model.copy()
    
    # Change all variables to continuous
    for var in lp_model.variables():
        var.cat = pulp.LpContinuous
    
    return lp_model

# Test the functions
def test_tsp_formulation():
    # Example adjacency matrix (costs)
    adj_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    
    ilp_model = formulate_tsp(adj_matrix)
    lp_relaxation = create_lp_relaxation(ilp_model)
    
    print("ILP Formulation:")
    print(ilp_model)
    print("\nLP Relaxation:")
    print(lp_relaxation)

# Run the test
test_tsp_formulation()
