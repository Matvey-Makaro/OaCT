import numpy as np
from matrix_reversal import reverse_matrix


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


def build_pseudoplan(inv_A_basis: np.ndarray, b: np.ndarray, B: list, n: int) -> np.ndarray:
    k_b = inv_A_basis.dot(b)
    k = [[0] for i in range(n)]
    j = 0
    for i in B:
        k[i] = k_b[j]
        j += 1
    return np.array(k)


def is_optimal_plan(k: np.ndarray) -> bool:
    for row in k:
        if row[0] < 0:
            return False
    return True


def get_neg_component_index(k: np.ndarray) -> int:
    for i in range(len(k)):
        if k[i, 0] < 0:
            return i
    raise RuntimeError("Negative component must bi in k.")


def build_not_basis_index(n: int, B: list) -> list:
    return [i for i in range(n) if i not in B]


def calc_nu(delta_y_transpose: np.ndarray, A: np.ndarray, non_B: list) -> np.ndarray:
    nu = []
    for j in non_B:
        A_j = A[:, j]
        nu.append(delta_y_transpose.dot(A_j))
    return np.array(nu)


def is_task_compatible(nu: np.ndarray) -> bool:
    for nu_j in nu:
        if nu_j < 0:
            return True
    return False


def calc_sigma(nu: np.ndarray, c: np.ndarray, A: np.ndarray, non_B: list, y: np.ndarray) -> list:
    sigma = []
    for i in range(len(nu)):
        if nu[i] >= 0:
            continue
        j = non_B[i]
        A_transpose_j = A[:, j]
        sigma_j = (c[j] - A_transpose_j.dot(y)) / nu[i]
        sigma.append((j, sigma_j[0]))
    return sigma


def find_j0(sigma: list) -> int:
    min = sigma[0][1]
    j0 = sigma[0][0]
    for sigma_i in sigma:
        if sigma_i[1] < min:
            min = sigma_i[1]
            j0 = sigma_i[0]
    return j0


def dual_simplex_method(c: np.ndarray, A: np.ndarray, b: np.ndarray, B: list) -> np.ndarray:
    # step 1
    A_basis = build_basis_matrix(A, B)
    inv_A_basis = np.linalg.inv(A_basis)

    while True:
        # step 2
        c_basis = build_basis_vector(c, B)

        # step 3
        y_transposed = c_basis.T.dot(inv_A_basis)

        # step 4
        n = A.shape[1]
        k = build_pseudoplan(inv_A_basis, b, B, n)

        # step 5
        if is_optimal_plan(k):
            return k

        # step 6
        i = B.index(get_neg_component_index(k))

        # step 7
        delta_y_transpose = inv_A_basis[i]
        non_B = build_not_basis_index(n, B)
        nu = calc_nu(delta_y_transpose, A, non_B)

        # step 8
        if not is_task_compatible(nu):
            raise RuntimeError("The task is not compatible.")

        # step 9
        sigma = calc_sigma(nu, c, A, non_B, y_transposed.T)

        # step 10
        j0 = find_j0(sigma)

        # step 11
        B[i] = j0

        # step 1
        col = build_basis_matrix(A, [j0])
        is_inv, inv_A_basis = reverse_matrix(inv_A_basis, col, i)
        if not is_inv:
            raise RuntimeError("Basis matrix is not inversible")



