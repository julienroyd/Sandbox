"""
Simple demo of a M-steps random walk positional encoding
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Build a simple graph
nodes = np.array([0, 1, 2, 3, 4, 5])
adjacency = np.array([[0, 1, 1, 0, 0, 0], 
                      [1, 0, 1, 0, 0, 0], 
                      [1, 1, 0, 1, 0, 0], 
                      [0, 0, 1, 0, 1, 1], 
                      [0, 0, 0, 1, 0, 1], 
                      [0, 0, 0, 1, 1, 0]])

# Perform the random walk for each node
N = 1000
M = 10
m_step_random_walk = np.zeros((len(nodes), len(nodes), N))

for init_node in nodes:
    for n in range(N):
        print(f"Starting node: {init_node}")
        cur_node = init_node
        for m in range(M):
            # Get the neighbors of the current node
            neighbors = np.where(adjacency[cur_node] == 1)[0]
            # Select a random neighbor
            random_neighbor = np.random.choice(neighbors)
            # Update the random walk matrix
            m_step_random_walk[init_node, random_neighbor, n] += 1
            # Update the current node
            cur_node = random_neighbor

m_step_random_walk = np.mean(m_step_random_walk, axis=2)
print(f"Random walk matrix after {M} steps:\n{m_step_random_walk}")

# Normalize the random walk matrix
m_step_random_walk = m_step_random_walk / M
print(f"Normalized random walk matrix:\n{m_step_random_walk}")

# Matplotlib heatmap of the random walk matrix
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(m_step_random_walk, annot=True, ax=ax)
ax.set_title(f"Random walk matrix after {M} steps")
ax.set_xlabel("Destination node")
ax.set_ylabel("Starting node")
plt.show()

