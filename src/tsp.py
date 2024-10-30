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
        
        # Add edges from solution
        total_distance = 0
        edge_count = 0
        for i, j in combinations(range(self.n_cities), 2):
            idx = self._get_variable_index(i, j)
            if solution[idx] > 1 - EDGE_TOLERANCE:
                solution_graph.add_edge(i, j)
                total_distance += self.distances[(i, j)]
                edge_count += 1
        
        print(f"\nPlotting solution:")
        print(f"Number of edges: {edge_count}")
        print(f"Total distance: {total_distance:.2f}")
        
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
        
        # If there are subtours, plot them separately
        subtours = self.find_subtours(solution)
        if len(subtours) > 1 or edge_count > 0:  # Only plot if we have edges or multiple subtours
            print(f"Found {len(subtours)} subtours")
            
            plt.figure(figsize=(10, 10))
            colors = ['r', 'b', 'g', 'y', 'm', 'c']
            
            for idx, subtour in enumerate(subtours):
                subtour_graph = solution_graph.subgraph(subtour)
                nx.draw_networkx_nodes(subtour_graph, pos, node_color=colors[idx % len(colors)],
                                     node_size=500)
                nx.draw_networkx_edges(subtour_graph, pos, edge_color=colors[idx % len(colors)],
                                     width=2)
                nx.draw_networkx_labels(subtour_graph, pos, 
                                      {i: f"City {i}" for i in subtour})
            
            plt.title(f"Subtours in Solution - {len(subtours)} components")
            plt.axis('equal')
            
            filename = f"plots/{problem_name}_subtours_{self.plot_counter:03d}.png"
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()

def solution_callback(solution: np.ndarray, is_optimal: bool, tsp_instance: TSPInstance, problem_name: str):
    """Callback function for visualizing solutions during branch and bound."""
    tsp_instance.plot_solution(solution, is_optimal, problem_name)

def main():
    """Generate and solve example TSP instances."""
    # Example 1: 4 cities in a square
    coords1 = {
        0: (0, 0),
        1: (0, 10),
        2: (10, 10),
        3: (10, 0)
    }
    tsp1 = TSPInstance(4, coords1)
    tsp1.plot_instance("Square TSP - 4 Cities")
    
    # Example 2: 5 cities in a pentagon
    coords2 = {
        0: (50, 0),
        1: (15.45, 47.55),
        2: (80.9, 58.78),
        3: (97.55, 15.45),
        4: (32.45, 15.45)
    }
    tsp2 = TSPInstance(5, coords2)
    tsp2.plot_instance("Pentagon TSP - 5 Cities")
    
    # Example 3: 6 random cities
    tsp3 = TSPInstance(6)
    tsp3.plot_instance("Random TSP - 6 Cities")
    
    # Solve each instance
    solver = ILPSolver()
    
    for i, tsp in enumerate([tsp1, tsp2, tsp3], 1):
        problem_name = f"TSP_{i}"
        print(f"\nSolving {problem_name}")
        
        # Get all constraint matrices including equality constraints
        c, A_eq, b_eq, A_ub, b_ub = tsp.to_ilp()
        
        # Create a callback closure that includes the TSP instance and problem name
        callback = lambda solution, is_optimal, problem_name=problem_name: solution_callback(
            solution, is_optimal, tsp, problem_name)
        
        # Solve with both equality and inequality constraints
        solve_and_print_results(
            solver=solver,
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            problem_name=problem_name,
            callback=callback,
            visualize=True,
            tsp_instance=tsp
        )
        
if __name__ == "__main__":
    main()