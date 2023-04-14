import numpy as np

from quadratic_prog_problem import solve_quadratic_prog_problem


def simple_test() -> None:
    c = np.array([[-8],
                  [-6],
                  [-4],
                  [-6]])

    D = np.array([[2, 1, 1, 0],
                  [1, 1, 0, 0],
                  [1, 0, 1, 0],
                  [0, 0, 0, 0]])

    A = np.array([[1, 0, 2, 1],
                  [0, 1, -1, 2]])

    x_T = np.array([[2, 3, 0, 0]])
    B = [0, 1]
    B_star = [0, 1]

    x = solve_quadratic_prog_problem(c, D, A, x_T, B, B_star)
    print('x:\n', x)
