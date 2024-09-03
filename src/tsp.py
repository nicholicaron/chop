"""
Just playing around with the PuLP library -- feel free to disregard this file
"""

from pulp import *

import pulp

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