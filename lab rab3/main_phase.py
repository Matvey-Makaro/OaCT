import numpy as np
from matrix_reversal import reverse_matrix


class NotLimitedError(Exception):
    def __init__(self):
        self.message = "The target function is not unlimited from above" \
                       "on the set of permissible plans."

    def __str__(self):
        return self.message


def build_basis_matrix(A: np.ndarray, B: list) -> np.ndarray:
    basis_matrix = np.ndarray([A.shape[0], len(B)])
    for j in range(0, basis_matrix.shape[1]):
        for i in range(0, basis_matrix.shape[0]):
            basis_matrix[i, j] = A[i, B[j]]

    return basis_matrix


def build_basis_vector(c: np.ndarray, B: list) -> np.ndarray:
    basis_c = np.ndarray([len(B), 1])
    for i in range(0, len(B)):
        basis_c[i, 0] = c[B[i], 0]
    return basis_c


def build_potential_vector(basis_c: np.ndarray, inv_basis_matrix: np.ndarray) -> np.ndarray:
    return np.dot(basis_c.T, inv_basis_matrix).T


def build_estimate_vector(potential_vector: np.ndarray, A: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (np.dot(potential_vector.T, A) - c.T).T


def is_optimal_plan(estimate_vector: np.ndarray, B: list) -> bool:
    for i in range(0, estimate_vector.shape[0]):
        if i not in B:
            if estimate_vector[i, 0] < 0:
                return False
    return True


def get_j0(estimate_vector: np.ndarray) -> int:
    for i in range(0, estimate_vector.shape[0]):
        if estimate_vector[i, 0] < 0:
            return i


def build_z(inv_basis_matrix: np.ndarray, A: np.ndarray, j0: int) -> np.ndarray:
    A_j0 = np.ndarray([A.shape[0], 1])
    for i in range(0, A.shape[0]):
        A_j0[i, 0] = A[i, j0]
    return np.dot(inv_basis_matrix, A_j0)


def build_teta(x: np.ndarray, z: np.ndarray, B: list) -> np.ndarray:
    teta = np.ndarray([z.shape[0], 1])
    for i in range(0, z.shape[0]):
        if z[i, 0] > 0:
            teta[i, 0] = x[B[i], 0] / z[i, 0]
        else:
            teta[i, 0] = float('inf')
    return teta


def get_teta0(teta: np.ndarray) -> (float, int):
    teta0 = float('inf')
    index = 0
    for i in range(0, teta.shape[0]):
        if teta[i, 0] < teta0:
            teta0 = teta[i, 0]
            index = i
    return teta0, index


def is_limited(teta0: float) -> bool:
    return teta0 != float('inf')


def update_x(x: np.ndarray, j0: int, k: int, j_star: int, teta0: float, B: list, z: np.ndarray) -> None:
    x[j0] = teta0
    for i in range(0, len(B)):
        if i != k:
            x[B[i], 0] -= teta0 * z[i, 0]
    x[j_star] = 0


def main_phase(c: np.ndarray, A: np.ndarray, x: np.ndarray, B: list) -> (np.ndarray, list):
    # step 1
    basis_matrix = build_basis_matrix(A, B)
    inv_basis_matrix = np.linalg.inv(basis_matrix)

    while True:
        # step 2
        basis_c = build_basis_vector(c, B)

        # step 3
        potential_vector = build_potential_vector(basis_c, inv_basis_matrix)

        # step 4
        estimate_vector = build_estimate_vector(potential_vector, A, c)
        # step 5
        if is_optimal_plan(estimate_vector, B):
            return x, B

        # step 6
        j0 = get_j0(estimate_vector)

        # step 7
        z = build_z(inv_basis_matrix, A, j0)

        # step 8
        teta = build_teta(x, z, B)

        # step 9 and step 11
        teta0, k = get_teta0(teta)

        # step 10
        if not is_limited(teta0):
            raise NotLimitedError

        # step 11
        j_star = B[k]

        # step 12
        B[k] = j0

        #step 13
        update_x(x, j0, k, j_star, teta0, B, z)

        print("New x:\n", x)
        print("New B:\n", B)

        # step 1
        col = build_basis_matrix(A, [j0])
        is_inv, inv_basis_matrix = reverse_matrix(inv_basis_matrix, col, k)
        if not is_inv:
            raise RuntimeError("Basis matrix is not inversible")
