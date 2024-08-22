import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml
import heapq  # for priority queue in Dijkstra's algorithm

# Load the PGM image
image_path = 'turtlebot_map.pgm'
image = Image.open(image_path)

map_array = np.array(image)

# Load the YAML file
yaml_path = 'turtlebot_map.yaml'
with open(yaml_path, 'r') as file:
    map_info = yaml.safe_load(file)

# Adjust the threshold to detect white areas more flexibly
free_space_mask_strict_adjusted = (map_array >= 254)
free_space_coords_strict_adjusted = np.column_stack(np.where(free_space_mask_strict_adjusted))

# Define the GNG class with a minimum edge length parameter
class GNG:
    def __init__(self, input_data, max_nodes=500, max_age=30, learning_rate=0.05, alpha=0.33, beta=0.0005, min_edge_length=0.5):
        self.input_data = input_data.astype(np.float64)
        self.max_nodes = max_nodes
        self.max_age = max_age
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.min_edge_length = min_edge_length  # Minimum edge length
        self.nodes = []
        self.edges = {}
        self.errors = []
    
    def initialize(self):
        indices = np.random.choice(np.arange(self.input_data.shape[0]), 2, replace=False)
        self.nodes = [self.input_data[indices[0]], self.input_data[indices[1]]]
        self.edges = {0: {1: 0}, 1: {0: 0}}
        self.errors = [0.0, 0.0]
    
    def find_nearest_nodes(self, point):
        distances = np.linalg.norm(self.nodes - point, axis=1)
        nearest_indices = np.argsort(distances)[:2]
        return nearest_indices, distances[nearest_indices]
    
    def add_node(self):
        max_error_index = np.argmax(self.errors)
        max_error_node = self.nodes[max_error_index]
        max_error_neighbor_index = min(self.edges[max_error_index], key=self.edges[max_error_index].get)
        max_error_neighbor = self.nodes[max_error_neighbor_index]
        
        new_node = (max_error_node + max_error_neighbor) / 2
        self.nodes.append(new_node)
        new_index = len(self.nodes) - 1
        
        # Check if the new edge length meets the minimum requirement
        if np.linalg.norm(max_error_node - new_node) >= self.min_edge_length and np.linalg.norm(max_error_neighbor - new_node) >= self.min_edge_length:
            self.edges[max_error_index].pop(max_error_neighbor_index, None)
            self.edges[max_error_neighbor_index].pop(max_error_index, None)
            self.edges.setdefault(max_error_index, {})[new_index] = 0
            self.edges.setdefault(max_error_neighbor_index, {})[new_index] = 0
            self.edges[new_index] = {max_error_index: 0, max_error_neighbor_index: 0}
        
        self.errors.append(self.errors[max_error_index] * self.alpha)
        self.errors[max_error_index] *= self.alpha
        self.errors[max_error_neighbor_index] *= self.alpha
    
    def remove_old_edges(self):
        nodes_to_remove = []
        for i in range(len(self.nodes)):
            if i in self.edges:
                edges_to_remove = [j for j in list(self.edges[i]) if self.edges[i][j] > self.max_age]
                for j in edges_to_remove:
                    self.edges[i].pop(j, None)
                    if j in self.edges:
                        self.edges[j].pop(i, None)
                if not self.edges[i]:
                    nodes_to_remove.append(i)
        
        for i in sorted(nodes_to_remove, reverse=True):
            self.nodes.pop(i)
            self.edges.pop(i, None)
            self.errors.pop(i)
            for j in self.edges:
                if i in self.edges[j]:
                    self.edges[j].pop(i, None)
    
    def fit(self, n_iterations=2000):
        self.initialize()
        for iteration in range(n_iterations):
            point = self.input_data[np.random.choice(np.arange(self.input_data.shape[0]))]
            nearest_indices, distances = self.find_nearest_nodes(point)
            s1, s2 = nearest_indices
            
            # Ensure s1 and s2 are valid indices before proceeding
            if s1 >= len(self.nodes) or s2 >= len(self.nodes):
                continue
            
            # Move the nearest node and its neighbors closer to the point
            self.nodes[s1] += self.learning_rate * (point - self.nodes[s1])
            for neighbor in list(self.edges.get(s1, [])):
                if neighbor < len(self.nodes):  # Check if neighbor is a valid index
                    self.nodes[neighbor] += self.learning_rate * self.beta * (point - self.nodes[neighbor])
            
            # Increment the ages of all edges connected to s1
            for neighbor in list(self.edges.get(s1, [])):
                if neighbor < len(self.edges) and s1 < len(self.edges[neighbor]):  # Validate indices
                    self.edges[s1][neighbor] += 1
                    self.edges[neighbor][s1] += 1
            
            # Reset the age of the edge between s1 and s2, or create it if it doesn't exist
            edge_length = np.linalg.norm(self.nodes[s1] - self.nodes[s2])
            if edge_length >= self.min_edge_length:
                self.edges.setdefault(s1, {})[s2] = 0
                self.edges.setdefault(s2, {})[s1] = 0
            
            self.errors[s1] += distances[0] ** 2
            
            self.remove_old_edges()
            
            if len(self.nodes) < self.max_nodes and iteration % 100 == 0:
                self.add_node()
            
            self.errors = [error * (1 - self.beta) for error in self.errors]
        
        return self.nodes, self.edges

def dijkstra_shortest_path(nodes, edges, start_idx, goal_idx):
    queue = [(0, start_idx)]
    distances = {start_idx: 0}
    previous_nodes = {start_idx: None}

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_node == goal_idx:
            break

        if current_distance > distances.get(current_node, float('inf')):
            continue

        for neighbor, edge_cost in edges.get(current_node, {}).items():
            distance = current_distance + np.linalg.norm(nodes[current_node] - nodes[neighbor])
            if distance < distances.get(neighbor, float('inf')):
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    path = []
    current_node = goal_idx
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]
    
    return path

# Apply the GNG to the white (free space) areas identified
gng_strict_adjusted = GNG(input_data=free_space_coords_strict_adjusted, max_nodes=4000, max_age=100, min_edge_length=0.01)
nodes_strict_adjusted, edges_strict_adjusted = gng_strict_adjusted.fit(n_iterations=50000)

# Add start and goal positions to the nodes
start_pos = np.array([141,200 ])
goal_pos = np.array([225, 207])

# Find the nearest nodes to the start and goal positions
start_nearest_idx = np.argmin(np.linalg.norm(nodes_strict_adjusted - start_pos, axis=1))
goal_nearest_idx = np.argmin(np.linalg.norm(nodes_strict_adjusted - goal_pos, axis=1))

# Add the start and goal nodes to the GNG nodes and edges
nodes_strict_adjusted = np.vstack([nodes_strict_adjusted, start_pos, goal_pos])
start_idx = len(nodes_strict_adjusted) - 2
goal_idx = len(nodes_strict_adjusted) - 1

edges_strict_adjusted[start_idx] = {start_nearest_idx: 0}
edges_strict_adjusted[goal_idx] = {goal_nearest_idx: 0}
edges_strict_adjusted[start_nearest_idx][start_idx] = 0
edges_strict_adjusted[goal_nearest_idx][goal_idx] = 0

# Compute the shortest path using Dijkstra's algorithm
path = dijkstra_shortest_path(nodes_strict_adjusted, edges_strict_adjusted, start_idx, goal_idx)

# Visualize the GNG network and the shortest path
plt.figure(figsize=(8, 8))
plt.imshow(map_array, cmap='gray')
for node in nodes_strict_adjusted:
    plt.plot(node[1], node[0], 'ro', markersize=3)
for i, neighbors in edges_strict_adjusted.items():
    for j in neighbors:
               plt.plot([nodes_strict_adjusted[i][1], nodes_strict_adjusted[j][1]], 
                 [nodes_strict_adjusted[i][0], nodes_strict_adjusted[j][0]], 'r-', linewidth=0.5, label='Edges' if i==0 and j== list(neighbors.keys())[0] else "")

# Highlight the shortest path with a different color (e.g., blue)
for k in range(len(path) - 1):
    plt.plot([nodes_strict_adjusted[path[k]][1], nodes_strict_adjusted[path[k + 1]][1]],
             [nodes_strict_adjusted[path[k]][0], nodes_strict_adjusted[path[k + 1]][0]], 'b-', linewidth=3, label='Shortest Path' if k == 0 else "")

# Mark the start and goal positions
plt.plot(start_pos[1], start_pos[0], '^g', markersize=8, label='Start')  # Start position in green
plt.plot(goal_pos[1], goal_pos[0], 'ms', markersize=8, label='Goal')    # Goal position in magenta
plt.gca().invert_yaxis()
plt.legend(loc='upper right')
plt.title('GNG Topological Map and Shortest Path using Dijkstra')
plt.show()