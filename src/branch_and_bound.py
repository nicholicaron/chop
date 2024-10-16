import numpy as np
from scipy.optimize import linprog
#from quantecon.optimize import linprog_simplex
import networkx as nx
import matplotlib.pyplot as plt
import heapq

class Node:
    def __init__(self, bounds, parent=None, branch_var=None, branch_val=None, branch_direction=None):
        self.bounds = bounds
        self.parent = parent
        self.branch_var = branch_var
        self.branch_val = branch_val
        self.branch_direction = branch_direction
        self.solution = None
        self.value = None
        self.estimate = parent.value if parent else 0
        self.id = None
        self.depth = parent.depth + 1 if parent else 0
        self.local_lower_bound = -np.inf
        self.local_upper_bound = np.inf
        self.relaxed_soln = None
        self.relaxed_obj_value = None
        self.num_int = 0
        self.num_frac = 0
        self.indices_frac = []
        self.active_constraints = []
        self.slack_values = []
        self.optimality_gap = np.inf
        self.reduced_costs = []
        self.parent_objective = parent.value if parent else None
        self.children_pruned = 0
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
        self.global_lower_bound = -np.inf
        self.global_upper_bound = np.inf
        self.problem_counter = 0
        self.root_relaxation_value = None

    def solve(self, c, A_ub, b_ub, visualize=False):
        self._reset()
        self._set_global_attributes(c, A_ub, b_ub)

        root_bounds = [(0, None) for _ in range(len(c))]
        root_node = Node(root_bounds)
        root_node.id = self._get_next_node_id()
        self._add_node_to_tree(root_node, c, A_ub, b_ub)
        self._update_node_attributes(root_node, {'color': 'blue'})

        priority_queue = [(0, root_node)]

        while priority_queue:
            _, current_node = heapq.heappop(priority_queue)

            result = self._solve_lp_relaxation(c, A_ub, b_ub, current_node.bounds)

            if not result.success:
                current_node.prune_reason = 'infeasible'
                self._update_node_attributes(current_node, {'color': 'red', 'prune_reason': 'infeasible'})
                if current_node.parent:
                    current_node.parent.children_pruned += 1
                continue

            current_node.relaxed_soln = result.x
            current_node.relaxed_obj_value = -result.fun  # Note the negation due to maximization
            current_node.value = current_node.relaxed_obj_value
            current_node.estimate = current_node.value

            if self.root_relaxation_value is None:
                self.root_relaxation_value = current_node.value
                self.global_upper_bound = self.root_relaxation_value

            self._calculate_node_attributes(current_node, c, A_ub, b_ub, result)

            if current_node.value <= self.global_lower_bound:
                current_node.prune_reason = 'suboptimal'
                self._update_node_attributes(current_node, {'color': 'orange', 'prune_reason': 'suboptimal'})
                if current_node.parent:
                    current_node.parent.children_pruned += 1
                continue

            non_integer_vars = current_node.indices_frac

            if not non_integer_vars:
                self._update_node_attributes(current_node, {'color': 'green'})
                if current_node.value > self.global_lower_bound:
                    self.global_lower_bound = current_node.value
                    self.optimal_obj_value = current_node.value
                    self.optimal_solution = result.x
                    self.optimal_node = current_node.id
            else:
                branch_var = non_integer_vars[0]
                branch_val = result.x[branch_var]

                for direction in ['floor', 'ceil']:
                    new_bounds = current_node.bounds.copy()
                    if direction == 'floor':
                        new_bounds[branch_var] = (new_bounds[branch_var][0], np.floor(branch_val))
                        new_val = np.floor(branch_val)
                    else:
                        new_bounds[branch_var] = (np.ceil(branch_val), new_bounds[branch_var][1])
                        new_val = np.ceil(branch_val)

                    new_node = Node(new_bounds, current_node, branch_var, new_val, direction)
                    new_node.id = self._get_next_node_id()

                    new_A_ub, new_b_ub = self._update_constraints(A_ub, b_ub, branch_var, new_val, direction)

                    self._add_node_to_tree(new_node, c, new_A_ub, new_b_ub)
                    heapq.heappush(priority_queue, (-new_node.estimate, new_node))

        self._calculate_integrality_gap()
        if visualize:
          self._visualize_tree()
        self._save_graph_to_disk()
        return self.optimal_solution, self.optimal_obj_value

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
            'local_lower_bound': node.local_lower_bound,
            'local_upper_bound': node.local_upper_bound,
            'current_constraints': A_ub.tolist(),
            'current_rhs': b_ub.tolist(),
            'active_constraints': [],
            'slack_values': [],
            'optimality_gap': np.inf,
            'reduced_costs': [],
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
        node.indices_frac = [i for i, x in enumerate(current_node.x) if abs(x - round(x)) > 1e-6]

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

    def _calculate_integrality_gap(self):
        if self.optimal_obj_value > -np.inf and self.root_relaxation_value is not None:
            integrality_gap = (self.root_relaxation_value - self.optimal_obj_value) / abs(self.optimal_obj_value)
            self.enumeration_tree.graph['integrality_gap'] = integrality_gap

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

    def _solve_lp_relaxation(self, c, A_ub, b_ub, bounds):
        return linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    def _set_global_attributes(self, c, A_ub, b_ub):
        self.enumeration_tree.graph['og_obj_coefs'] = c.tolist() # Convert numpy array to list
        self.enumeration_tree.graph['og_constraints'] = A_ub.tolist() # Convert numpy array to list
        self.enumeration_tree.graph['og_rhs'] = b_ub.tolist() # Convert numpy array to list

    def _visualize_tree(self):
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

        plt.title("ILP Branch and Bound Enumeration Tree (Best-First Search)")
        plt.tight_layout()
        plt.show()

    def _save_graph_to_disk(self):
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
        graph_name = f"graph_{self.problem_counter}_{timestamp}"
        self.problem_counter += 1

        # Ensure the 'saved_graphs' directory exists
        os.makedirs('saved_graphs', exist_ok=True)

        # Save the graph
        torch.save(data, f'saved_graphs/{graph_name}.pt')
        print(f"Graph saved as {graph_name}.pt")



def solve_and_print_results(solver, c, A_ub, b_ub, name, visualize=False):
    print(f"\nSolving {name}:")
    solution, value = solver.solve(c, A_ub, b_ub, visualize)
    print(f"{name} Results:")
    print("Optimal Solution:", solution)
    print("Optimal Value:", value)

def main():
    solver = ILPSolver()

    # Example 1: 2 variables, 2 constraints
    c1 = np.array([1, 1])
    A_ub1 = np.array([[-1, 1], [8, 2]])
    b_ub1 = np.array([2, 19])
    solve_and_print_results(solver, c1, A_ub1, b_ub1, "Example 1 (2 variables, 2 constraints)", visualize=True)

    # Example 2: 5 variables, 3 constraints
    c2 = np.array([3, 2, 5, 4, 1])
    A_ub2 = np.array([
        [2, 1, 3, 2, 1],
        [1, 2, 1, 1, 3],
        [1, 1, 2, 3, 1]
    ])
    b_ub2 = np.array([10, 8, 15])
    solve_and_print_results(solver, c2, A_ub2, b_ub2, "Example 2 (5 variables, 3 constraints)", visualize=True)

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
    solve_and_print_results(solver, c3, A_ub3, b_ub3, "Example 3 (8 variables, 5 constraints)", visualize=True)

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
    solve_and_print_results(solver, c4, A_ub4, b_ub4, "Example 4 (10 variables, 7 constraints)", visualize=True)

