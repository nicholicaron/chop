import numpy as np
#from scipy.optimize import linprog
from simplex import linprog_simplex, SimplexResult, PivOptions
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils.convert import from_networkx
import os
import time
import heapq
import argparse
from itertools import combinations


class Node:
    def __init__(self, parent=None, branch_var=None, branch_val=None, branch_direction=None):
        self.parent = parent
        self.branch_var = branch_var
        self.branch_val = branch_val
        self.branch_direction = branch_direction
        self.solution = None
        self.value = None
        self.estimate = parent.value if parent else 0
        self.id = None
        self.depth = parent.depth + 1 if parent else 0
        self.local_upper_bound = -np.inf # relaxation objective value
        self.relaxed_soln = None
        self.num_int = 0
        self.num_frac = 0
        self.indices_frac = []
        self.active_constraints = []
        self.slack_values = []
        self.optimality_gap = np.inf
        self.parent_objective = parent.value if parent else None
        self.prune_reason = None

    def __lt__(self, other):
        return -self.estimate < -other.estimate

class ILPSolver:
    def __init__(self):
        self.optimal_obj_value = -np.inf
        self.optimal_solution = None
        self.enumeration_tree = nx.DiGraph()
        self.node_counter = 0
        self.optimal_node = None
        self.global_lower_bound = -np.inf # Greatest lower bound (integer feasible solution) found among all instances so far
        self.problem_counter = 0 
        self.root_relaxation_value = None # Objective value of root node relaxation
        self.n_cities = 0

    def solve(self, c, A_ub, b_ub, A_eq=None, b_eq=None, problem_name="default_name", visualize=False):
        self._reset()
        self._set_global_attributes(c, A_ub, b_ub)
        # Calculate number of cities
        # Assumes that the TSP is of the standard formulaticon, i.e. c contains only the binary edge variables
        # len(c) = n_cities * (n_cities - 1) / 2, so we solve for n_cities using the quadratic equation 
        self.n_cities = round((1 + np.sqrt(1 + 8 * len(c))) / 2) 

        root_node = Node()
        root_node.id = self._get_next_node_id()
        
        # Solve LP relaxation for root node
        result = self._solve_lp_relaxation(c, A_ub, b_ub)
        if not result.success:
            return SimplexResult(None, None, None, False, 2, 0, None)
        
        root_node.relaxed_soln = result.x
        root_node.value = result.fun
        root_node.local_upper_bound = root_node.value
        self.root_relaxation_value = root_node.value

        self._add_node_to_tree(root_node, c, A_ub, b_ub)
        self._update_node_attributes(root_node, {'color': 'blue'})

        priority_queue = [(root_node.value, root_node)]

        while priority_queue:
            _, current_node = heapq.heappop(priority_queue)

            if current_node.local_upper_bound <= self.global_lower_bound:
                current_node.prune_reason = 'suboptimal'
                self._update_node_attributes(current_node, {'color': 'orange', 'prune_reason': 'suboptimal'})
                continue

            is_integer_solution = all(abs(x - round(x)) < 1e-6 for x in current_node.relaxed_soln)

            if is_integer_solution:
                violated_constraints = self._find_violated_subtour_constraints(current_node.relaxed_soln)
                if not violated_constraints:
                    if current_node.value > self.global_lower_bound:
                        self.global_lower_bound = current_node.value
                        self.optimal_obj_value = current_node.value
                        self.optimal_solution = current_node.relaxed_soln
                        self.optimal_node = current_node.id
                        self._update_node_attributes(current_node, {'color': 'green'})
                    else:
                        current_node.prune_reason = 'suboptimal'
                        self._update_node_attributes(current_node, {'color': 'orange', 'prune_reason': 'suboptimal'})
                else:
                    # Add violated constraints and re-solve
                    new_A_ub = A_ub.copy()
                    new_b_ub = b_ub.copy()
                    for constraint in violated_constraints:
                        new_A_ub = np.vstack([new_A_ub, constraint[:-1]])  # LHS
                        new_b_ub = np.append(new_b_ub, constraint[-1])  # RHS
                    result = self._solve_lp_relaxation(c, new_A_ub, new_b_ub)
                    if result.success:
                        current_node.relaxed_soln = result.x
                        current_node.value = result.fun
                        current_node.local_upper_bound = current_node.value
                        heapq.heappush(priority_queue, (current_node.value, current_node))
                    continue
            else:
                # Branch on a fractional variable
                fractional_vars = [i for i, x in enumerate(current_node.relaxed_soln) if abs(x - round(x)) > 1e-6]
                branch_var = fractional_vars[0]
                branch_val = current_node.relaxed_soln[branch_var]

                for direction in ['floor', 'ceil']:
                    new_A_ub = A_ub.copy()
                    new_b_ub = b_ub.copy()
                    
                    if direction == 'floor':
                        new_constraint = np.zeros(len(c))
                        new_constraint[branch_var] = -1
                        new_A_ub = np.vstack([new_A_ub, new_constraint])
                        new_b_ub = np.append(new_b_ub, -np.floor(branch_val))
                        new_val = np.floor(branch_val)
                    else:
                        new_constraint = np.zeros(len(c))
                        new_constraint[branch_var] = 1
                        new_A_ub = np.vstack([new_A_ub, new_constraint])
                        new_b_ub = np.append(new_b_ub, np.ceil(branch_val))
                        new_val = np.ceil(branch_val)

                    new_node = Node(parent=current_node, branch_var=branch_var, branch_val=new_val, branch_direction=direction)
                    new_node.id = self._get_next_node_id()

                    result = self._solve_lp_relaxation(c, new_A_ub, new_b_ub)
                    if result.success:
                        new_node.relaxed_soln = result.x
                        new_node.value = result.fun
                        new_node.local_upper_bound = new_node.value
                        self._add_node_to_tree(new_node, c, new_A_ub, new_b_ub)
                        if new_node.local_upper_bound > self.global_lower_bound:
                            heapq.heappush(priority_queue, (new_node.value, new_node))
                    else:
                        new_node.prune_reason = 'infeasible'
                        self._update_node_attributes(new_node, {'color': 'red', 'prune_reason': 'infeasible'})

        if visualize:
            self._visualize_tree(problem_name)
        self._save_graph_to_disk(problem_name)
        return SimplexResult(self.optimal_solution, None, self.optimal_obj_value, True, 0, 0, None)
    
    def _find_violated_subtour_constraints(self, solution):
        edges = [(i, j) for i in range(self.n_cities) for j in range(i+1, self.n_cities) 
                 if solution[i*(self.n_cities-1) - i*(i+1)//2 + j - 1] > 0.5]
        G = nx.Graph(edges)
        
        violated_constraints = []
        for r in range(2, self.n_cities):
            for subset in combinations(range(self.n_cities), r):
                subgraph = G.subgraph(subset)
                if nx.is_connected(subgraph) and sum(solution[i*(self.n_cities-1) - i*(i+1)//2 + j - 1] 
                                                     for i, j in combinations(subset, 2)) > len(subset) - 1 + 1e-6:
                    constraint = [0] * (self.n_cities * (self.n_cities - 1) // 2)
                    for i, j in combinations(subset, 2):
                        if i < j:
                            idx = i*(self.n_cities-1) - i*(i+1)//2 + j - 1
                        else:
                            idx = j*(self.n_cities-1) - j*(j+1)//2 + i - 1
                        constraint[idx] = 1
                    violated_constraints.append(constraint)
        
        return violated_constraints

    def _reset(self):
        self.optimal_obj_value = -np.inf
        self.optimal_solution = None
        self.enumeration_tree = nx.DiGraph()
        self.node_counter = 0
        self.optimal_node = None
        self.global_lower_bound = -np.inf
        self.global_upper_bound = np.inf
        self.root_relaxation_value = None

    def _add_node_to_tree(self, node, c, A_ub, b_ub):
        attributes = self._get_default_node_attributes()
        attributes.update({
            'depth': node.depth,
            'branch_variable': node.branch_var,
            'branch_value': node.branch_val,
            'branch_direction': node.branch_direction,
            'local_upper_bound': node.local_upper_bound,
            'current_constraints': A_ub.tolist(),
            'current_rhs': b_ub.tolist(),
            'active_constraints': [],
            'slack_values': [],
            'optimality_gap': np.inf,
            'parent_objective': node.parent_objective,
            'children_pruned': 0,
            'prune_reason': None,
        })
        self.enumeration_tree.add_node(node.id, **attributes)

        if node.parent:
            self.enumeration_tree.add_edge(node.parent.id, node.id)

    def _calculate_node_attributes(self, node, c, A_ub, b_ub, result):
        # Calculate active constraints
        node.active_constraints = np.where(np.isclose(A_ub @ result.x, b_ub))[0].tolist()

        # Calculate slack values
        node.slack_values = (b_ub - A_ub @ result.x).tolist()

        # Calculate optimality gap
        if self.global_lower_bound > -np.inf:
            node.optimality_gap = (node.value - self.global_lower_bound) / abs(self.global_lower_bound)

        # Update number of integer and fractional variables
        node.num_int = sum(1 for x in result.x if abs(x - round(x)) < 1e-6)
        node.num_frac = len(c) - node.num_int
        node.indices_frac = [i for i, x in enumerate(result.x) if abs(x - round(x)) > 1e-6]

        self._update_node_attributes(node, {
            'relaxed_soln': node.relaxed_soln.tolist(),
            'relaxed_obj_value': node.relaxed_obj_value,
            'value': node.value,
            'active_constraints': node.active_constraints,
            'slack_values': node.slack_values,
            'optimality_gap': node.optimality_gap,
            'num_int': node.num_int,
            'num_frac': node.num_frac,
            'indices_frac': node.indices_frac,
            'parent_objective': node.parent_objective,
            'children_pruned': node.children_pruned
        })

    #def _calculate_integrality_gap(self):
    #    if self.optimal_obj_value > -np.inf and self.root_relaxation_value is not None:
    #        integrality_gap = (self.root_relaxation_value - self.optimal_obj_value) / abs(self.optimal_obj_value)
    #        self.enumeration_tree.graph['integrality_gap'] = integrality_gap

    def _get_default_node_attributes(self):
        return {
            'depth': 0,
            'branch_variable': None,
            'branch_value': None,
            'branch_direction': None,
            'fractionality': None,
            'local_lower_bound': -np.inf,
            'local_upper_bound': np.inf,
            'global_lower_bound': -np.inf,
            'global_upper_bound': np.inf,
            'current_constraints': None,
            'current_rhs': None,
            'relaxed_soln': None,
            'relaxed_obj_value': None,
            'num_int': 0,
            'num_frac': 0,
            'indices_frac': [],
            'active_constraints': [],
            'slack_values': [],
            'optimality_gap': np.inf,
            'parent_objective': None,
            'children_pruned': 0,
            'color': 'lightgray'
        }

    def _update_node_attributes(self, node, attributes):
        # Ensure we're only updating existing attributes
        current_attributes = self.enumeration_tree.nodes[node.id]
        for key, value in attributes.items():
            if key in current_attributes:
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                current_attributes[key] = value

    def _update_constraints(self, A_ub, b_ub, branch_var, branch_val, direction):
        new_row = np.zeros(A_ub.shape[1])
        new_row[branch_var] = 1 if direction == 'floor' else -1
        new_A_ub = np.vstack([A_ub, new_row])
        new_b_ub = np.append(b_ub, branch_val if direction == 'floor' else -branch_val)
        return new_A_ub, new_b_ub

    def _get_next_node_id(self):
        self.node_counter += 1
        return f"Node {self.node_counter}"

    def _solve_lp_relaxation(self, c, A_ub, b_ub, A_eq=None, b_eq=None):
        # Ensure A_eq and b_eq are numpy arrays, even if empty
        if A_eq is None:
            A_eq = np.empty((0, len(c)))
        if b_eq is None:
            b_eq = np.empty(0)

        # Create PivOptions with default values
        piv_options = PivOptions()

        # Call linprog_simplex
        result = linprog_simplex(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            max_iter=10**6,
            piv_options=piv_options
        )

        # Convert the result to our expected format
        return SimplexResult(
            x=result.x,
            lambd=result.lambd,
            fun=result.fun,
            success=result.success,
            status=result.status,
            num_iter=result.num_iter,
            tableau=result.tableau
        )

    def _set_global_attributes(self, c, A_ub, b_ub):
        self.enumeration_tree.graph['og_obj_coefs'] = c.tolist() # Original objective coefficients
        self.enumeration_tree.graph['og_constraints'] = A_ub.tolist() # Original constraint matrix
        self.enumeration_tree.graph['og_rhs'] = b_ub.tolist() # Original right-hand side

    def _visualize_tree(self, problem_name):
        fig, ax = plt.subplots(figsize=(24, 24))

        colors = []
        for node in self.enumeration_tree.nodes:
            node_data = self.enumeration_tree.nodes[node]
            if node_data.get('color') == 'blue':  # Root node
                colors.append('blue')
            elif node_data['prune_reason'] == 'infeasible':
                colors.append('red')
            elif node_data['prune_reason'] == 'suboptimal':
                colors.append('orange')
            elif node_data.get('color') == 'green': # Optimal node
                colors.append('green')
            else:
                colors.append('lightgray')
        labels = {}
        for node in self.enumeration_tree.nodes:
            node_data = self.enumeration_tree.nodes[node]
            label = f"{node}\n"
            label += f"Value: {node_data.get('relaxed_obj_value', 'N/A'):.2f}\n"

            if node_data.get('color') == 'blue':
                label += "Root Node"
            elif node_data['branch_variable'] is not None:
                label += f"Branch Var: X{node_data['branch_variable']}\n"
                label += f"Branch Val: {node_data['branch_value']:.2f}\n"
                label += f"Direction: {node_data['branch_direction']}"

            labels[node] = label

        pos = nx.spring_layout(self.enumeration_tree)
        nx.draw(self.enumeration_tree, pos, with_labels=True, labels=labels,
                node_color=colors, node_size=5000, font_size=8, ax=ax)

        legend_elements = [
            plt.Rectangle((0,0),1,1,fc="blue", edgecolor='none', label='Root'),
            plt.Rectangle((0,0),1,1,fc="red", edgecolor='none', label='Pruned (Infeasible)'),
            plt.Rectangle((0,0),1,1,fc="orange", edgecolor='none', label='Pruned (Suboptimal)'),
            plt.Rectangle((0,0),1,1,fc="green", edgecolor='none', label='Integer Solution'),
            plt.Rectangle((0,0),1,1,fc="lightgray", edgecolor='none', label='Non-integral')
        ]
        ax.legend(handles=legend_elements, loc='best')

        plt.title(f"ILP Branch and Bound Enumeration Tree (Best-First Search) - {problem_name}")
        plt.tight_layout()
        
        # Create a directory for plots if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        # Save the plot as a PNG file
        plot_filename = f"plots/{problem_name.replace(' ', '_').lower()}_tree.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"Plot saved as {plot_filename}")

    def _save_graph_to_disk(self, problem_name):
        # Ensure all node attributes are lists or basic Python types
        for node, data in self.enumeration_tree.nodes(data=True):
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    data[key] = value.tolist()
                elif isinstance(value, np.integer):
                    data[key] = int(value)
                elif isinstance(value, np.floating):
                    data[key] = float(value)

        # Convert NetworkX graph to PyG Data
        data = from_networkx(self.enumeration_tree)

        # Create a unique name for the graph
        timestamp = int(time.time() * 1000)
        graph_name = f"graph_{problem_name.replace(' ', '_').lower()}_{self.problem_counter}_{timestamp}"
        self.problem_counter += 1

        # Ensure the 'saved_graphs' directory exists
        os.makedirs('saved_graphs', exist_ok=True)

        # Save the graph
        torch.save(data, f'saved_graphs/{graph_name}.pt')
        print(f"Graph saved as {graph_name}.pt")



def solve_and_print_results(solver, c, A_ub, b_ub, problem_name, visualize=False):
    print(f"\nSolving {problem_name}:")
    solution, value = solver.solve(c, A_ub, b_ub, problem_name, visualize)
    print(f"{problem_name} Results:")
    print("Optimal Solution:", solution)
    print("Optimal Value:", value)
    return solution, value

def main(visualize):
    solver = ILPSolver()

    # Example 1: 2 variables, 2 constraints
    c1 = np.array([1, 1])
    A_ub1 = np.array([[-1, 1], [8, 2]])
    b_ub1 = np.array([2, 19])
    solve_and_print_results(solver, c1, A_ub1, b_ub1, "Example 1 (2 var, 2 cons)", visualize=visualize)

    # Example 2: 5 variables, 3 constraints
    c2 = np.array([3, 2, 5, 4, 1])
    A_ub2 = np.array([
        [2, 1, 3, 2, 1],
        [1, 2, 1, 1, 3],
        [1, 1, 2, 3, 1]
    ])
    b_ub2 = np.array([10, 8, 15])
    solve_and_print_results(solver, c2, A_ub2, b_ub2, "Example 2 (5 var, 3 cons)", visualize=visualize)

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
    solve_and_print_results(solver, c3, A_ub3, b_ub3, "Example 3 (8 var, 5 cons)", visualize=visualize)

    # Example 4: 10 variables, 7 constraints
    c4 = np.array([4, 6, 2, 3, 7, 5, 8, 1, 9, 3])
    A_ub4 = np.array([
        [2, 3, 1, 4, 2, 5, 1, 3, 2, 1],
        [1, 2, 3, 1, 4, 2, 3, 1, 2, 4],
        [3, 1, 2, 4, 1, 3, 2, 5, 1, 2],
        [4, 2, 3, 1, 5, 2, 1, 3, 4, 2],
        [1, 3, 2, 5, 1, 4, 3, 2, 1, 3],
        [2, 4, 1, 3, 2, 1, 5, 4, 3, 1],
        [3, 2, 4, 1, 3, 5, 2, 1, 4, 2]
    ])
    b_ub4 = np.array([30, 25, 35, 40, 20, 28, 32])
    solve_and_print_results(solver, c4, A_ub4, b_ub4, "Example 4 (10 var, 7 cons)", visualize=visualize)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ILP Solver with Branch and Bound")
    parser.add_argument("--visualize", action="store_true", help="Generate and save plots")
    args = parser.parse_args()

    print("Starting branch and bound solver...")
    main(args.visualize)
