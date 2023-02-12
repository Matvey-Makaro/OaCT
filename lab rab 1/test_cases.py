import numpy as np


def get_congenital_matrix_testcase() -> (np.ndarray, np.ndarray, np.ndarray, int):
    A = np.array([[1, 2, 1], [1, 10, 1], [3, 5, 6]])
    inv_A = np.linalg.inv(A)
    x = np.array([0, 0, 0]).reshape(3, 1)
    i = 1
    return A, inv_A, x, i


def get_simple_testcase() -> (np.ndarray, np.ndarray, np.ndarray, int):
    A = np.array([[1, -1, 1], [0, 1, 0], [0, 0, 1]])
    inv_A = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    x = np.array([1, 0, 1])
    i = 2
    return A, inv_A, x, i
