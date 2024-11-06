import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import random
from typing import Tuple, List, Dict, Set
import sys
import os
sys.path.append('.')
from branch_and_bound import solve_and_print_results, ILPSolver
import requests, tarfile, os, gzip, shutil
from tqdm.auto import tqdm
from tsplib95.loaders import load_problem, load_solution


class TSPInstance:
    """
    Represents a Traveling Salesman Problem instance.
    
    Attributes:
        n_cities (int): Number of cities
        coordinates (dict): Dictionary mapping city index to (x,y) coordinates
        distances (dict): Dictionary mapping city pairs to distances
        graph (nx.Graph): NetworkX graph representation
        plot_counter (int): Counter for naming plot files
    """
    
    def __init__(self, n_cities: int, coordinates: Dict[int, Tuple[float, float]] = None):
        self.n_cities = n_cities
        if coordinates is None:
            self.coordinates = {i: (random.uniform(0, 100), random.uniform(0, 100)) 
                              for i in range(n_cities)}
        else:
            self.coordinates = coordinates
            
        self.distances = {}
        for i, j in combinations(range(n_cities), 2):
            x1, y1 = self.coordinates[i]
            x2, y2 = self.coordinates[j]
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            self.distances[(i, j)] = dist
            self.distances[(j, i)] = dist
            
        self.graph = nx.Graph()
        for i in range(n_cities):
            self.graph.add_node(i, pos=self.coordinates[i])
        for (i, j), dist in self.distances.items():
            if i < j:
                self.graph.add_edge(i, j, weight=dist)
        
        self.plot_counter = 0
        os.makedirs('plots', exist_ok=True)


    def to_ilp(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert TSP to ILP formulation with proper degree constraints.
        
        Returns:
            Tuple containing:
            - c: Objective coefficients (maximizing negative distances)
            - A_eq: Equality constraint matrix (degree constraints)
            - b_eq: Equality constraint RHS
            - A_ub: Inequality constraint matrix (initial empty)
            - b_ub: Inequality constraint RHS (initial empty)
        """
        n = self.n_cities
        num_vars = n * (n - 1) // 2
        
        # Objective: Minimize distances (maximize negative distances)
        c = np.zeros(num_vars)
        for i, j in combinations(range(n), 2):
            idx = self._get_variable_index(i, j)
            c[idx] = -self.distances[(i, j)]
        
        # Degree constraints: Each city must have exactly 2 edges
        A_eq = []
        b_eq = []
        
        for i in range(n):
            row = np.zeros(num_vars)
            for j in range(n):
                if i != j:
                    idx = self._get_variable_index(min(i, j), max(i, j))
                    row[idx] = 1
            A_eq.append(row)
            b_eq.append(2)  # Exactly 2 edges per vertex
        
        # Initialize empty inequality constraints (will be added during B&B)
        A_ub = np.zeros((0, num_vars))
        b_ub = np.zeros(0)
        
        return np.array(c), np.array(A_eq), np.array(b_eq), np.array(A_ub), np.array(b_ub)

    
    def find_subtours(self, solution: np.ndarray) -> List[Set[int]]:
        """
        Find all subtours in the current solution with improved edge detection.
        
        Args:
            solution: Binary solution vector
            
        Returns:
            List of sets, where each set contains the cities in a subtour
        """
        # Create a graph from the solution
        G = nx.Graph()
        for i in range(self.n_cities):
            G.add_node(i)
        
        # Use a more lenient tolerance for binary values
        EDGE_TOLERANCE = 1e-4
        
        # Debug information
        print("\nAnalyzing solution for subtours:")
        print(f"Solution vector: {solution}")
        
        edge_count = 0
        for i, j in combinations(range(self.n_cities), 2):
            idx = self._get_variable_index(i, j)
            # Print debug info for values close to 1
            if solution[idx] > 0.5:  # Check any significant values
                print(f"Edge ({i},{j}) has value {solution[idx]}")
            
            # More lenient check for edges
            if solution[idx] > 1 - EDGE_TOLERANCE:
                G.add_edge(i, j)
                edge_count += 1
        
        print(f"Total edges found: {edge_count}")
        
        # Find and print connected components (subtours)
        subtours = list(nx.connected_components(G))
        print(f"Found {len(subtours)} subtours:")
        for idx, subtour in enumerate(subtours):
            print(f"Subtour {idx + 1}: {subtour}")
        
        # Validate degree constraints
        self._validate_degrees(G)
        
        return subtours

    def generate_subtour_constraint(self, subtour: Set[int]) -> Tuple[np.ndarray, float]:
        """
        Generate subtour elimination constraint for a given subtour.
        
        For a subtour S, the constraint is:
        sum(x[i,j] for i,j in S) <= |S| - 1
        
        This ensures that any subset of k cities cannot have k edges
        between them, preventing isolated subtours.
        
        Args:
            subtour: Set of cities forming a subtour
            
        Returns:
            Tuple of (constraint_coefficients, rhs)
        """
        n = self.n_cities
        constraint = np.zeros(n * (n - 1) // 2)
        
        # For each pair of cities in the subtour
        for i, j in combinations(subtour, 2):
            idx = self._get_variable_index(min(i, j), max(i, j))
            constraint[idx] = 1
        
        # RHS: |S| - 1 ensures we can't have a complete subtour
        rhs = len(subtour) - 1
        
        return constraint, rhs
    
    def _validate_degrees(self, G: nx.Graph) -> None:
        """
        Validate that the graph satisfies degree constraints.
        
        Args:
            G: NetworkX graph of the current solution
        """
        print("\nValidating degree constraints:")
        for node in G.nodes():
            degree = G.degree(node)
            print(f"City {node} has degree {degree}")
            if degree != 2 and G.number_of_edges() > 0:
                print(f"Warning: City {node} has irregular degree {degree}")
    
    def _get_variable_index(self, i: int, j: int) -> int:
        """
        Get the index of the decision variable for edge (i,j) in the ILP formulation.
    
        For a TSP with n cities, we create n*(n-1)/2 binary variables, one for each
        possible undirected edge. This method converts a city pair (i,j) to the
        corresponding variable index in our flattened representation.
    
        Args:
            i (int): First city index
            j (int): Second city index
        
        Returns:
            int: Index in the flattened variable array
        
        Example:
            For a 4-city problem, the mapping would be:
            (0,1) -> 0
            (0,2) -> 1
            (0,3) -> 2
            (1,2) -> 3
            (1,3) -> 4
            (2,3) -> 5
        """
        # Ensure i < j for consistent indexing
        if i > j:
            i, j = j, i
        
        # Calculate index using combinatorial formula
        # For city i, we skip all combinations of smaller cities:
        # i*(n-1) - i*(i+1)/2
        # Then add the offset for current j: + (j-1)
        return i * (self.n_cities - 1) - i * (i + 1) // 2 + j - 1

    def plot_instance(self, title: str = "TSP Instance"):
        """
        Plot the TSP instance and save to disk.
        """
        plt.figure(figsize=(10, 10))
        pos = nx.get_node_attributes(self.graph, 'pos')
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', 
                             node_size=500)
        
        # Draw edges with weights
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        edge_labels = {k: f"{v:.1f}" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels)
        nx.draw_networkx_edges(self.graph, pos)
        
        # Draw labels
        labels = {i: f"City {i}" for i in range(self.n_cities)}
        nx.draw_networkx_labels(self.graph, pos, labels)
        
        plt.title(title)
        plt.axis('equal')
        
        # Save plot
        filename = f"plots/tsp_instance_{title.lower().replace(' ', '_')}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_solution(self, solution: np.ndarray, is_optimal: bool = False, problem_name: str = "unknown"):
        """Plot TSP solution with improved edge detection and validation."""
        plt.figure(figsize=(10, 10))
        pos = nx.get_node_attributes(self.graph, 'pos')
        
        # Create a new graph for the solution
        solution_graph = nx.Graph()
        solution_graph.add_nodes_from(self.graph.nodes(data=True))
        
        # Use same tolerance as find_subtours
        EDGE_TOLERANCE = 1e-4
                
        for i, j in combinations(range(self.n_cities), 2):
            idx = self._get_variable_index(i, j)
            if solution[idx] > 1 - EDGE_TOLERANCE:
                solution_graph.add_edge(i, j)
                #total_distance += self.distances[(i, j)]

        total_distance = -sum(solution[self._get_variable_index(i, j)] * self.distances[(i, j)]
                         for i, j in combinations(range(self.n_cities), 2))
        
        # Draw nodes
        nx.draw_networkx_nodes(solution_graph, pos, node_color='lightblue', 
                              node_size=500)
        
        # Draw edges (highlighted for solution)
        nx.draw_networkx_edges(solution_graph, pos, edge_color='r', width=2)
        
        # Draw labels
        labels = {i: f"City {i}" for i in range(self.n_cities)}
        nx.draw_networkx_labels(solution_graph, pos, labels)
        
        status = "Optimal" if is_optimal else "Candidate"
        plt.title(f"{status} Solution - Total Distance: {total_distance:.2f}")
        plt.axis('equal')
        
        # Save plot with incrementing counter
        self.plot_counter += 1
        filename = f"plots/{problem_name}_solution_{self.plot_counter:03d}_{status.lower()}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

def download_and_extract_tsplib(url, directory="tsplib_95_data", delete_after_unzip=True):
    os.makedirs(directory, exist_ok=True)
    
    # Download with progress bar
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open("tsplib.tar.gz", 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                pbar.update(len(chunk))

    # Extract tar.gz
    with tarfile.open("tsplib.tar.gz", 'r:gz') as tar:
        tar.extractall(directory)

    # Decompress .gz files inside directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".gz"):
                path = os.path.join(root, file)
                with gzip.open(path, 'rb') as f_in, open(path[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(path)

    if delete_after_unzip:
        os.remove("tsplib.tar.gz")

# # Utils function: we will normalize the coordinates of the VRP instances
# def normalize_coord(coord:torch.Tensor) -> torch.Tensor:
#     x, y = coord[:, 0], coord[:, 1]
#     x_min, x_max = x.min(), x.max()
#     y_min, y_max = y.min(), y.max()
#     x_scaled = (x - x_min) / (x_max - x_min) 
#     y_scaled = (y - y_min) / (y_max - y_min)
#     coord_scaled = torch.stack([x_scaled, y_scaled], dim=1)
#     return coord_scaled 

# def tsplib_to_td(problem, normalize=True):
#     coords = torch.tensor(problem['node_coords']).float()
#     coords_norm = normalize_coord(coords) if normalize else coords
#     td = TensorDict({
#         'locs': coords_norm,
#     })
#     td = td[None] # add batch dimension, in this case just 1
#     return td


def solution_callback(solution: np.ndarray, is_optimal: bool, tsp_instance: TSPInstance, problem_name: str):
    """Callback function for visualizing solutions during branch and bound."""
    tsp_instance.plot_solution(solution, is_optimal, problem_name)

def main():

    # Create and solve simple TSP instances
    # Example 1: 3 cities in a triangle
    coords_3 = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0)
    }
    tsp_3 = TSPInstance(3, coords_3)
    tsp_3.plot_instance("Triangle TSP")
    
    # Example 2: 4 cities in a square
    coords_4 = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 1),
        3: (1, 0)
    }
    tsp_4 = TSPInstance(4, coords_4)
    tsp_4.plot_instance("Square TSP")
    
    # Example 3: 5 cities in a star pattern
    coords_5 = {
        0: (0, 0),    # center
        1: (1, 1),    # top right
        2: (-1, 1),   # top left
        3: (-1, -1),  # bottom left
        4: (1, -1)    # bottom right
    }
    tsp_5 = TSPInstance(5, coords_5)
    tsp_5.plot_instance("Star TSP")
    
    # Initialize solver
    solver = ILPSolver()
    
    # Solve each instance
    for instance, name in [(tsp_3, "Triangle"), (tsp_4, "Square"), (tsp_5, "Star")]:
        print(f"\nSolving {name} TSP instance with {instance.n_cities} cities")
        
        # Get all constraint matrices
        c, A_eq, b_eq, A_ub, b_ub = instance.to_ilp()
        
        # Create callback closure
        callback = lambda solution, is_optimal, problem_name: solution_callback(
            solution, is_optimal, instance, problem_name
        )
        
        # Solve the instance
        solve_and_print_results(
            solver=solver,
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            problem_name=f"{name}_TSP",
            callback=callback,
            visualize=True,
            tsp_instance=instance
        )


    """
    # Only download and extract if directory is empty or doesn't exist
    if not os.path.exists('tsplib_95_data') or not os.listdir('tsplib_95_data'):
        print("Downloading and extracting TSPLIB instances...")
        download_and_extract_tsplib("http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz")
    
    # Load the problems from TSPLib
    tsplib_dir = './tsplib_95_data'
    solution_files = [f for f in os.listdir(tsplib_dir) if f.endswith('.opt.tour')]
    
    problems = []
    # Load only problems with solution files
    for sol_file in solution_files:
        prob_file = sol_file.replace('.opt.tour', '.tsp')
        problem = load_problem(os.path.join(tsplib_dir, prob_file))

        # Skip problems without node coordinates
        if not len(problem.node_coords):
            continue
        
        node_coords = [v for v in problem.node_coords.values()]
        solution = load_solution(os.path.join(tsplib_dir, sol_file))
        
        problems.append({
            "name": sol_file.replace('.opt.tour', ''),
            "node_coords": node_coords,
            "solution": solution.tours[0],
            "dimension": problem.dimension
        })
    
    # Sort problems by dimension
    problems = sorted(problems, key=lambda x: x['dimension'])
    
    # Get the smallest problem
    smallest_problem = problems[0]
    print(f"\nSolving smallest TSP instance: {smallest_problem['name']} with dimension {smallest_problem['dimension']}")
    
    # Create TSP instance from the problem
    coordinates = {i: tuple(coord) for i, coord in enumerate(smallest_problem['node_coords'])}
    tsp_instance = TSPInstance(smallest_problem['dimension'], coordinates)
    
    # Initialize solver
    solver = ILPSolver()
    
    # Get all constraint matrices
    c, A_eq, b_eq, A_ub, b_ub = tsp_instance.to_ilp()
    
    # Create callback closure
    callback = lambda solution, is_optimal, problem_name: solution_callback(
        solution, is_optimal, tsp_instance, problem_name
    )
    
    # Solve the instance
    solve_and_print_results(
        solver=solver,
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        problem_name=smallest_problem['name'],
        callback=callback,
        visualize=True,
        tsp_instance=tsp_instance
    )
    """

    # Code to solve all problems
    """
    for problem in problems:
        print(f"\nSolving TSP instance: {problem['name']} with dimension {problem['dimension']}")
        
        # Create TSP instance
        coordinates = {i: tuple(coord) for i, coord in enumerate(problem['node_coords'])}
        tsp_instance = TSPInstance(problem['dimension'], coordinates)
        
        # Get constraints
        c, A_eq, b_eq, A_ub, b_ub = tsp_instance.to_ilp()
        
        # Create callback
        callback = lambda solution, is_optimal: solution_callback(
            solution, is_optimal, tsp_instance, problem['name']
        )
        
        # Solve
        solve_and_print_results(
            solver=solver,
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            problem_name=problem['name'],
            callback=callback,
            visualize=True,
            tsp_instance=tsp_instance
        )
    """
        
if __name__ == "__main__":
    main()