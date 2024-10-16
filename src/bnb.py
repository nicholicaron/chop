import numpy as np
from scipy.optimize import linprog
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

optimal_obj_value = np.inf
optimal_solution = None
enumeration_tree = nx.DiGraph()
node_counter = 0
optimal_node = None

def ilp_solver(obj_coeffs, constraint_coeffs, constraint_rhs, parent_node=None):
    """
    Solves an ILP minimization problem using the branch and bound algorithm.
    
    Parameters:
    (For some given ILP: z := min{c^T * x: x belongs to S which is a subset of Z^n})
    - obj_coeffs (np.array): Objective function coefficients (c).
    - constraint_coeffs (np.array): Left-hand side of inequality constraints (A).
    - constraint_rhs (np.array): Right-hand side of inequality constraints (b).
    - parent_node (str, optional): Label of the parent node in the enumeration tree.
    
    Returns:
    - np.array or None: Optimal integer solution if found, None otherwise.
    """
    global optimal_obj_value, optimal_solution, node_counter, optimal_node

    node_label = f"Node {node_counter}"
    enumeration_tree.add_node(node_label, solution="?", value=np.inf, color="lightblue")
    
    if parent_node is not None:
        enumeration_tree.add_edge(parent_node, node_label)
    
    current_node = node_label
    node_counter += 1

    # Create bounds for all variables: 0 <= x < infinity
    bounds = [(0, None) for _ in range(len(obj_coeffs))]

    result = linprog(obj_coeffs, A_ub=constraint_coeffs, b_ub=constraint_rhs, 
                     bounds=bounds, method='highs')

    if not result.success:
        enumeration_tree.nodes[current_node]['color'] = 'salmon'  # Infeasible
        return None

    enumeration_tree.nodes[current_node]['solution'] = np.round(result.x, 2)
    enumeration_tree.nodes[current_node]['value'] = result.fun

    # Prune if the lower bound is worse than the current best solution
    if result.fun >= optimal_obj_value:
        enumeration_tree.nodes[current_node]['color'] = 'salmon'  # Pruned
        return None

    non_integer_vars = [i for i, x in enumerate(result.x) if abs(x - round(x)) > 1e-6]

    if not non_integer_vars:
        # We have an integer solution
        if result.fun < optimal_obj_value:
            # This is the new best solution
            optimal_obj_value = result.fun
            optimal_solution = result.x
            optimal_node = current_node
            enumeration_tree.nodes[current_node]['color'] = 'lightgreen'  # Optimal integer solution
        else:
            enumeration_tree.nodes[current_node]['color'] = 'yellow'  # Integer but not optimal
        return result.x

    # If we reach here, we need to branch
    branch_var = non_integer_vars[0]
    new_constraint = np.zeros(len(obj_coeffs))
    new_constraint[branch_var] = 1

    # Lower bound branch
    constraint_coeffs_lower = np.vstack([constraint_coeffs, new_constraint])
    constraint_rhs_lower = np.hstack([constraint_rhs, np.floor(result.x[branch_var])])

    # Upper bound branch
    constraint_coeffs_upper = np.vstack([constraint_coeffs, -new_constraint])
    constraint_rhs_upper = np.hstack([constraint_rhs, -np.ceil(result.x[branch_var])])

    ilp_solver(obj_coeffs, constraint_coeffs_lower, constraint_rhs_lower, current_node)
    ilp_solver(obj_coeffs, constraint_coeffs_upper, constraint_rhs_upper, current_node)

    return optimal_solution

def visualize_enumeration_tree():
    """
    Visualizes the branch and bound enumeration tree.
    
    This function creates a graphical representation of the branch and bound process,
    showing the explored nodes and their relationships. Nodes are color-coded to 
    indicate their status in the solution process.
    
    Color scheme:
    - Salmon: Pruned branches
    - Lightgreen: Optimal integer solution
    - Yellow: Integer solutions (non-optimal)
    - Lightblue: Nodes with fractional solutions
    
    The function uses the global 'enumeration_tree' (NetworkX DiGraph) and 'optimal_node'.
    """
    
    if optimal_node is not None:
        enumeration_tree.nodes[optimal_node]['color'] = 'lightgreen'

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract node colors and prepare labels
    node_colors = [enumeration_tree.nodes[node]['color'] for node in enumeration_tree.nodes]
    node_labels = {node: f"{enumeration_tree.nodes[node]['solution']}\nValue: {enumeration_tree.nodes[node]['value']:.2f}" 
                   for node in enumeration_tree.nodes}
    
    # Position nodes using graphviz_layout for a top-down tree
    node_positions = graphviz_layout(enumeration_tree, prog='dot')

    # Draw the graph
    nx.draw(enumeration_tree, node_positions, with_labels=True, labels=node_labels,
            node_color=node_colors, node_size=3000, font_size=8, ax=ax,
            arrows=True, arrowsize=20)

    # Create legend
    legend_elements = [
        plt.Rectangle((0,0),1,1,fc="salmon", edgecolor='none'),
        plt.Rectangle((0,0),1,1,fc="lightgreen", edgecolor='none'),
        plt.Rectangle((0,0),1,1,fc="yellow", edgecolor='none'),
        plt.Rectangle((0,0),1,1,fc="lightblue", edgecolor='none')
    ]
    legend_labels = ['Pruned', 'Optimal Solution', 'Integer (Non-optimal)', 'Fractional']
    ax.legend(legend_elements, legend_labels, loc='best')

    plt.title("Branch and Bound Enumeration Tree")
    plt.show()