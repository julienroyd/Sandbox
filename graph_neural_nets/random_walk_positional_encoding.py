"""
Simple demo of a M-steps random walk positional encoding
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Build a simple graph
nodes = np.array([0, 1, 2, 3, 4, 5])
A = np.array([[0, 1, 1, 0, 0, 0],
              [1, 0, 1, 0, 0, 0],
              [1, 1, 0, 1, 0, 0],
              [0, 0, 1, 0, 1, 1],
              [0, 0, 0, 1, 0, 1],
              [0, 0, 0, 1, 1, 0]])

MATRIX_MODE = True

if not MATRIX_MODE:
    # Perform the random walk for each node
    N = 10_000
    M = 10
    m_step_random_walk = np.zeros((len(nodes), len(nodes), N))

    for init_node in nodes:
        for n in range(N):
            print(f"Starting node: {init_node}")
            cur_node = init_node
            for m in range(M):
                # Get the neighbors of the current node
                neighbors = np.where(A[cur_node] == 1)[0]
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

else:
    """
    Matrix computation of the random walk
    """
    # The degree matrix is the diagonal matrix of the sum of the adjacency matrix (number of neighbors for each node)
    D = np.diag(np.sum(A, axis=1))

    # The transition matrix is the inverse of the degree matrix times the adjacency matrix
    # Since the degree matrix is a diagonal matrix, its inverse is simply the inverse of the elements of the diagonal
    T = np.linalg.inv(D) @ A
    print(f"Transition matrix:\n{T}")

    # Perform the random walk for each node using power of M of the transition matrix
    M = 10
    m_step_random_walk = np.linalg.matrix_power(T, M)
    print(f"Random walk matrix after {M} steps:\n{m_step_random_walk}")

# Plot the random walk matrix
sns.heatmap(m_step_random_walk, annot=True, cmap="Blues")
plt.xlabel("Destination node")
plt.ylabel("Starting node")
plt.title(f"Random walk matrix after {M} steps")
plt.show()