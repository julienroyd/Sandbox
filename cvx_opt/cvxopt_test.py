from cvxopt import matrix, solvers
import numpy as np


# =======================================
# LINKS to documentation:
# - https://cvxopt.org/
# - https://cvxopt.org/examples/tutorial/lp.html
# - https://cvxopt.org/userguide/coneprog.html#linear-programming

# EXAMPLE
#
#Procedure:
# 1. Put objective in the form: minimize c.T @ x
# 2. Put constraint set is form Ax <= b  with A.shape=(n_constraints, n_variables)
#
#Maths:
#    minimize    2*x1 + x2
#
#       s.t.     -x1 + x2   <= 1
#                -x1 - x2   <= -2
#                    - x2   <= 0
#                 x1 - 2*x2 <= 4
#Code:
# A = matrix(np.array([[-1.,  1.],
#                      [-1., -1.],
#                      [ 0., -1.],
#                      [ 1., -2.]]))
#
# b = matrix([1.0, -2.0, 0.0, 4.0])
# c = matrix([2.0, 1.0])
#
# sol = solvers.lp(c, A, b)
# print([f"{x:.2f}" for x in sol['x']])
# =======================================

# GOAL: Solve the mdp below using cvxopt solver on the LP formulation presented in p.18 of Pierre-Luc's notes
# example mdp: https://github.com/pierrelux/vfpolytope/blob/master/polytope.ipynb

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

P, R, discount = mdp_fig2d()  # Shape of P:(A,S,S'), Shape of R:(S,A)
p0 = np.array([0.5, 0.5])

# We want to solve this MDP by finding the optimal value function
# This problem can be formulated as a Linear Program of the form:
#
#    minimize    c @ v
#
#       s.t.     -v(s) + gamma * \sum_s' p(s'|s,a)v(s')  <=  - r(s,a)
#

c = matrix(p0)  # we use p0 (initial dist) as cost
b = matrix(R.flatten()) * -1.

A1 = np.array([discount * P[:, 0, 0] - 1, discount * P[:, 0, 1]])
A2 = np.array([discount * P[:, 1, 0], discount * P[:, 1, 1] - 1])

A = np.concatenate([A1, A2], axis=1).T

sol = solvers.lp(c=c, G=matrix(A), h=b)
vstar = np.squeeze(np.array(sol['x']))
print([f"v(s={i})={v_s:.2f}" for i, v_s in enumerate(vstar)])
print(f"LP value={p0 @ vstar:.6f}")  # solution should be: 1.0272725


# Justification for why the LP problem that solves the MDP has this form:
# The LP problem basically implements the statement of equation 6.9.1 of Putterman (p.223)
# Putterman shows (in Theorem 6.2.2a) that if a given v satisfies this inequality,
# then this v is un upper-bound on the optimal value-function v*.
# Thus, by using 6.9.1 as the constraint of the LP and minimizing p0 @ v, we are essentially
# expressing the optimisation problem of minimizing an upper-bound on v*, the solution of which is v*.
#
# The following bit of code (from Pierre-Luc's notebook: https://colab.research.google.com/drive/1_bcFk_nN3Sul31Bualm4v64vgrUjqLvL)
# shows that constructing an upper-bound on v* satisfies that constraint:

from itertools import product

n_states, n_actions = P.shape[0], P.shape[1]
all_action_choices = np.array(list(product(range(n_actions), repeat=n_states)))
deterministic_policies = np.eye(n_actions)[all_action_choices]

ppi = np.einsum('kij,lik->lij', P, deterministic_policies)  # shapes: P:(A,S,S'), policies:(N,S,A), P_pi:(N,S,S')
rpi = np.einsum('ij,kij->ki', R, deterministic_policies)  # shapes: R:(S,A), policies:(N,S,A), R_pi:(N,S)

def test_putterman_bound(v):
    """Equation 6.9.1 (p.223) of Markov Decision Processes (Putterman, 2005)"""
    return rpi + np.einsum('lij,j->li', discount * ppi, v)

vtest = vstar + 2.
print("Putterman bound demo:\n",
      vtest >= test_putterman_bound(vtest))
