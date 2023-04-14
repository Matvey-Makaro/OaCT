import numpy as np


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


def calc_c_x_T(c: np.ndarray, x_T: np.ndarray, D: np.ndarray) -> np.ndarray:
    return c.T + x_T.dot(D)


def calc_u_x_T(c_x_T: np.ndarray, A: np.ndarray, B: list) -> np.ndarray:
    c_b_x_T = build_basis_vector(c_x_T.T, B).T
    inv_A_b = np.linalg.inv(build_basis_matrix(A, B))
    return (-c_b_x_T).dot(inv_A_b)


def calc_delta_T(u_x_T: np.ndarray, A: np.ndarray, c_x_T: np.ndarray) -> np.ndarray:
    return u_x_T.dot(A) + c_x_T


def is_optimal_plan(delta_T: np.ndarray) -> bool:
    return (delta_T >= 0).all()


def find_j_0(delta_T: np.ndarray) -> int:
    for i in range(delta_T.shape[1]):
        if delta_T[0, i] < 0:
            return i
    raise RuntimeError('Negative component was not found in delta.')


def build_D_star(D: np.ndarray, B_star: list) -> np.ndarray:
    n = len(B_star)
    D_star = np.zeros((n, n))
    row = 0
    col = 0
    for i in B_star:
        for j in B_star:
            D_star[row, col] = D[i, j]
            col += 1
        row += 1
        col = 0
    return D_star


def build_H(D_T: np.ndarray, A_b_star: np.ndarray) -> np.ndarray:
    row_num = D_T.shape[0] + A_b_star.shape[0]
    A_b_star_T = A_b_star.T
    col_num = D_T.shape[1] + A_b_star_T.shape[1]

    H = np.zeros((row_num, col_num))
    for i in range(D_T.shape[0]):
        for j in range(D_T.shape[1]):
            H[i, j] = D_T[i, j]

    for i in range(D_T.shape[0], row_num):
        for j in range(A_b_star.shape[1]):
            H[i, j] = A_b_star[i - D_T.shape[0], j]

    for i in range(A_b_star_T.shape[0]):
        for j in range(D_T.shape[1], col_num):
            H[i, j] = A_b_star_T[i, j - D_T.shape[1]]
    return H


def build_b_star(D: np.ndarray, j0: int, B_star: list, A: np.ndarray) -> np.ndarray:
    b_star = []
    for i in B_star:
        b_star.append(D[i, j0])
    for i in range(A.shape[0]):
        b_star.append(A[i, j0])
    return np.array([b_star]).T


def calc_l_b_star(D: np.ndarray, A: np.ndarray, B_star: list, j0: int) -> np.ndarray:
    H = build_H(build_D_star(D, B_star), build_basis_matrix(A, B_star))
    inv_H = np.linalg.inv(H)
    b_star = build_b_star(D, j0, B_star, A)
    x_stick = (-inv_H).dot(b_star)
    l_b_star = []
    for i in range(len(B_star)):
        l_b_star.append(x_stick[i, 0])
    return l_b_star


def calc_l(B_star: list, j0: int, D: np.ndarray, A: np.ndarray) -> (np.ndarray, list):
    n = D.shape[0]
    l = [0 for i in range(n)]
    indx_to_l_indx = B_star.copy()
    for i in range(n):
        if i not in B_star:
            indx_to_l_indx.append(i)
    l[indx_to_l_indx.index(j0)] = 1
    l_b_star = calc_l_b_star(D, A, B_star, j0)
    for i in range(len(l_b_star)):
        l[i] = l_b_star[i]
    return np.array([l]).T, indx_to_l_indx


def calc_teta_j0(l: np.ndarray, D: np.ndarray, delta_T: np.ndarray, j0: int) -> float:
    sigma = l.T.dot(D).dot(l)
    if sigma > 0:
        teta_j0 = abs(delta_T[0, j0]) / sigma[0, 0]
    elif sigma == 0:
        teta_j0 = float('infinity')
    else:
        raise RuntimeError("Unreachable!")
    return teta_j0


def calc_teta(x_T: np.ndarray, l: np.ndarray, B_star: list, indx_to_l_indx: list) -> list:
    teta = []
    for j in B_star:
        l_j = l[indx_to_l_indx.index(j), 0]
        if l_j < 0:
            teta.append(-x_T[0, j] / l_j)
        else:
            teta.append(float('infinity'))
    return teta


def find_teta0(teta: list, teta_jo: float, j0: float, B_star: list) -> (float, int):
    min = teta_jo
    index = j0
    for i in range(len(teta)):
        if teta[i] < min:
            min = teta[i]
            index = B_star[i]
    return min, index


def is_case3(j_star: int, B: list, B_star: list, A: np.ndarray) -> (bool, int):
    if j_star not in B:
        return False, -1
    inv_A_b = np.linalg.inv(build_basis_matrix(A, B))
    s = B.index(j_star)
    for j_plus in B_star:
        if j_plus in B:
            continue
        A_j_plus = build_basis_matrix(A, [j_plus])
        tmp = inv_A_b.dot(A_j_plus)
        if tmp[s, 0] != 0:
            return True, j_plus

    return False, -1


def solve_quadratic_prog_problem(c: np.ndarray, D: np.ndarray, A: np.ndarray, x_T: np.ndarray, B: list,
                                 B_star: list) -> np.ndarray:
    while True:
        # step 1
        c_x_T = calc_c_x_T(c, x_T, D)
        print('c_x_T:\n', c_x_T)
        u_x_T = calc_u_x_T(c_x_T, A, B)
        print('\nu_x_T: \n', u_x_T)
        delta_T = calc_delta_T(u_x_T, A, c_x_T)
        print('\ndelta_T:\n', delta_T)

        # step 2
        if is_optimal_plan(delta_T):
            return x_T.T

        # step 3
        j0 = find_j_0(delta_T)

        # step 4
        l, idx_to_l_idx = calc_l(B_star, j0, D, A)
        print("\nl:\n", l)

        # step 5
        teta_j0 = calc_teta_j0(l, D, delta_T, j0)
        print("\nteta_j0\n", teta_j0)
        teta = calc_teta(x_T, l, B_star, idx_to_l_idx)
        print('\nteta\n', teta)

        teta0, j_star = find_teta0(teta, teta_j0, j0, B_star)
        print("\nteta0\n", teta0)
        print("\nj_star\n", j_star)

        if teta0 == float('infinity'):
            raise RuntimeError("The target function is not unlimited from bottom" \
                           "on the set of permissible plans.")

        # step 6
        x_T = (x_T.T + teta0 * l).T
        print("\nx_T\n", x_T)

        if j_star == j0:
            B_star.append(j_star)
        elif j_star in B_star and j_star not in B:
            B_star.remove(j_star)
        else:
            is_case_3, j_plus = is_case3(j_star, B, B_star, A)
            if is_case_3:
                B[B.index(j_star)] = j_plus
                B_star.remove(j_star)
            else:
                B[B.index(j_star)] = j0
                B_star[B.index(j_star)] = j0


