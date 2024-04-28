import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def mdp_fig2d():
    """ Figure 2 d) of
    ''The Value Function Polytope in Reinforcement Learning''
    by Dadashi et al. (2019) https://arxiv.org/abs/1901.11524
    """
    P = np.array([[[0.70, 0.30], [0.20, 0.80]],  # shape: (A, S, S')
                  [[0.99, 0.01], [0.99, 0.01]]])
    R = np.array(([[-0.45, -0.1],
                   [0.5, 0.5]]))
    return P, R, 0.9


def solve_mdp(P, R, gamma, policies):
    ppi = np.einsum('kij,lik->lij', P, policies)  # shapes: P:(A,S,S'), policies:(N,S,A), P_pi:(N,S,S')
    rpi = np.einsum('ij,kij->ki', R, policies)  # shapes: R:(S,A), policies:(N,S,A), R_pi:(N,S)
    return np.linalg.solve(np.eye(R.shape[0]) - gamma * ppi, rpi)


def sample_random_policies(npolicies=1000):
    nstates, nactions = 2, 2
    random_policies = np.zeros((npolicies, nstates, nactions))
    random_policies[:, :, 0] = np.random.uniform(size=(npolicies, nstates))
    random_policies[:, :, 1] = 1 - random_policies[:, :, 0]
    return random_policies

if __name__ == "__main__":
    P, R, gamma = mdp_fig2d()
    random_policies = sample_random_policies()
    vfs = solve_mdp(P, R, gamma, random_policies)

    nstates, nactions = P.shape[0], P.shape[1]
    all_action_choices = np.array(list(product(range(nactions), repeat=nstates)))
    deterministic_policies = np.eye(nactions)[all_action_choices]
    dvfs = solve_mdp(P, R, gamma, deterministic_policies)

    plt.scatter(vfs[:, 0], vfs[:, 1], s=12)
    plt.scatter(dvfs[:, 0], dvfs[:, 1], c='r')

    plt.show()