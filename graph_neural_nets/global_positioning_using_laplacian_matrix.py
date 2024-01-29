"""
Simple demo of how the Laplacian matrix can be used to encode the position of a node in a graph
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Build a simple graph
nodes = np.array([0, 1, 2, 3, 4, 5])
adjacency = np.array([[0, 1, 1, 0, 0, 0], 
                      [1, 0, 1, 0, 0, 0], 
                      [1, 1, 0, 1, 0, 0], 
                      [0, 0, 1, 0, 1, 1], 
                      [0, 0, 0, 1, 0, 1], 
                      [0, 0, 0, 1, 1, 0]])

# Compute the Laplacian matrix
laplacian = np.diag(np.sum(adjacency, axis=1)) - adjacency
print(f"Laplacian matrix:\n{laplacian}")

# Plot the graph with nodes and undirected edges
G = nx.DiGraph(np.array(adjacency))
nx.draw(G, with_labels=True, arrows=False)
plt.show()