import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from branch_and_bound import solve_and_print_results, ILPSolver
import os
import time
import random
import networkx as nx

def generate_random_tsp(n, x_range=(0, 100), y_range=(0, 100)):
    coordinates = [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(n)]
    edges = set()
    for i in range(n):
        degree = random.randint(2, 5)
        possible_neighbors = list(set(range(n)) - {i} - set(j for j, _ in edges if i == _) - set(_ for _, j in edges if i == j))
        neighbors = random.sample(possible_neighbors, min(degree, len(possible_neighbors)))
        for j in neighbors:
            if i < j:
                edges.add((i, j))
            else:
                edges.add((j, i))
    return coordinates, edges

def calculate_distances(coordinates, edges):
    n = len(coordinates)
    distances = {}
    for i, j in edges:
        dist = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
        distances[i, j] = distances[j, i] = dist
    return distances

def visualize_points(coordinates, edges, distances, title="TSP Instance", filename=None):
    x, y = zip(*coordinates)
    plt.figure(figsize=(12, 12))
    
    # Plot edges
    for i, j in edges:
        plt.plot([x[i], x[j]], [y[i], y[j]], 'gray', alpha=0.5)
        mid_x = (x[i] + x[j]) / 2
        mid_y = (y[i] + y[j]) / 2
        plt.annotate(f"{distances[i, j]:.1f}", (mid_x, mid_y), alpha=0.5, fontsize=8)

    # Plot points
    plt.scatter(x, y, c='blue', zorder=5)
    for i, (x, y) in enumerate(coordinates):
        plt.annotate(f"{i}", (x, y), xytext=(5, 5), textcoords='offset points', zorder=5)
    
    plt.title(title)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True)
    
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()
        print(f"TSP instance plot saved as {filename}")

def visualize_tour(coordinates, tour, edges, title, filename=None):
    G = nx.Graph()
    G.add_nodes_from(range(len(coordinates)))
    G.add_edges_from(edges)

    pos = {i: coord for i, coord in enumerate(coordinates)}
    
    plt.figure(figsize=(12, 8))
    
    # Draw all edges in light gray
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1)
    
    # Draw the tour edges in red
    tour_edges = list(zip(tour, tour[1:] + [tour[0]]))
    nx.draw_networkx_edges(G, pos, edgelist=tour_edges, edge_color='red', width=2)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
    
    # Add node labels
    labels = {i: f"{i}" for i in range(len(coordinates))}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    plt.title(title)
    plt.axis('off')
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Tour visualization saved as {filename}")
    else:
        plt.show()
    plt.close()

def create_tsp_ilp(distances, edges):
    n = max(max(i, j) for i, j in edges) + 1
    edge_to_index = {(min(i, j), max(i, j)): idx for idx, (i, j) in enumerate(edges)}
    
    # Objective function coefficients (we'll minimize distances)
    c = [distances[i, j] for i, j in edges]

    # Constraints matrix and right-hand side
    A_ub = []
    b_ub = []

    # Degree constraints
    for i in range(n):
        row = [0] * len(edges)
        for j in range(n):
            if (min(i, j), max(i, j)) in edge_to_index:
                row[edge_to_index[min(i, j), max(i, j)]] = 1
        A_ub.append(row)
        b_ub.append(2)
        A_ub.append([-x for x in row])
        b_ub.append(-2)

    # Subtour elimination constraints (for small instances)
    if n <= 5:
        for r in range(2, n):
            for subset in combinations(range(n), r):
                row = [0] * len(edges)
                for i, j in combinations(subset, 2):
                    if (min(i, j), max(i, j)) in edge_to_index:
                        row[edge_to_index[min(i, j), max(i, j)]] = 1
                A_ub.append(row)
                b_ub.append(len(subset) - 1)

    # Convert to numpy arrays
    c = np.array(c)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    return c, A_ub, b_ub, edge_to_index

def extract_tour(solution, edge_to_index, n):
    edges = [(i, j) for (i, j), idx in edge_to_index.items() if solution[idx] > 0.5]
    if not edges:
        return None
    
    tour = [edges[0][0]]
    while len(tour) < n:
        for edge in edges:
            if edge[0] == tour[-1] and edge[1] not in tour:
                tour.append(edge[1])
                break
            elif edge[1] == tour[-1] and edge[0] not in tour:
                tour.append(edge[0])
                break
        else:
            # If we can't find the next city, the tour is incomplete
            return None
    return tour

def solve_tsp(n, problem_name="TSP", visualize=True):
    timestamp = int(time.time() * 1000)
    coordinates, edges = generate_random_tsp(n)
    distances = calculate_distances(coordinates, edges)
    
    if visualize:
        instance_filename = f"plots/{problem_name}_instance_{timestamp}.png"
        visualize_points(coordinates, edges, distances, f"{problem_name} Instance", filename=instance_filename)

    c, A_ub, b_ub, edge_to_index = create_tsp_ilp(distances, edges)
    
    solver = ILPSolver()
    solution, objective_value, _, _ = solver.solve(c, A_ub, b_ub, problem_name=problem_name, visualize=visualize)

    if solution is None:
        print("Failed to find a valid solution.")
        return None, None

    tour = extract_tour(solution, edge_to_index, n)
    
    if tour is None:
        print("Failed to extract a valid tour from the solution.")
        return None, None

    if visualize:
        solution_filename = f"plots/{problem_name}_solution_{timestamp}.png"
        visualize_tour(coordinates, tour, edges, f"{problem_name} Solution (Total Distance: {objective_value:.2f})", filename=solution_filename)

    return tour, objective_value

def main():
    n = 5  # Number of cities
    print(f"Solving TSP with {n} cities...")
    tour, distance = solve_tsp(n, f"TSP_{n}_Cities")
    if tour and distance:
        print(f"Optimal tour: {tour}")
        print(f"Total distance: {distance:.2f}")
    else:
        print("Failed to solve the TSP instance.")

if __name__ == "__main__":
    main()
