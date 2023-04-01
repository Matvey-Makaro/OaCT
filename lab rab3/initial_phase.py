import numpy as np
from main_phase import main_phase, build_basis_matrix


def make_b_non_negative(A: np.ndarray, b: np.ndarray) -> None:
    for i in range(len(b)):
        if b[i, 0] < 0:
            b[i, 0] *= -1
            for j in range(A.shape[1]):
                A[i, j] *= -1


def make_auxiliary_task(A: np.ndarray) -> (np.ndarray, np.ndarray):
    m = A.shape[0]
    n = A.shape[1]

    c_wave = np.concatenate([[[0] for i in range(n)],[[-1] for i in range(m)]])
    A_wave = np.concatenate([A, np.eye(m)], axis=1)
    return c_wave, A_wave


def build_initial_base_plan(b: np.ndarray, n: int) -> (np.ndarray, list):
    m = len(b)
    x_wave = np.concatenate([[[0] for i in range(n)], b])
    B = [i for i in range(n, n + m)]
    return x_wave, B


def is_task_compatible(x_wave: np.ndarray, n: int) -> bool:
    for i in range(n, len(x_wave)):
        if x_wave[i, 0] != 0:
            return False
    return True


def form_allowable_plan(x_wave: np.ndarray, n: int) -> np.ndarray:
    x = np.array([x_wave[i, 0] for i in range(n)])
    return x


def is_resolved(B: list, n: int) -> bool:
    for i in B:
        if i >= n:
            return False
    return True


def max_index_of_artificial_variable(B: list, n: int) -> (int, int):
    k = i = 0
    max = -1
    for j in range(len(B)):
        if B[j] > max:
            max = B[j]
            k = j
            i = B[j] - n

    return k, i


def correct_B(A_wave: np.ndarray, B:list, n: int, k: int) -> bool:
    inv_A_wave_basis = np.linalg.inv(build_basis_matrix(A_wave, B))
    for j in range(n):
        if j in B:
            continue
        A_wave_j = A_wave[:, j]
        # step 9
        l = inv_A_wave_basis.dot(A_wave_j)
        if l[k] != 0:
            B[k] = j
            return True
    return False


def remove_main_restriction_of_task(A: np.ndarray, b: np.ndarray, B: list, A_wave: np.ndarray, i: int, k: int) -> \
        (np.ndarray, np.ndarray, list, np.ndarray):
    A = np.delete(A, (i), axis=0)
    b = np.delete(b, (i), axis=0)
    B.pop(k)
    A_wave = np.delete(A_wave, (i), axis=0)
    return A, b, B, A_wave


def initial_phase(c: np.ndarray, A: np.ndarray, b: np.ndarray) -> (np.ndarray, list):
    m = A.shape[0]
    n = A.shape[1]
    # step 1
    make_b_non_negative(A, b)

    # step 2
    c_wave, A_wave = make_auxiliary_task(A)

    # step 3
    x_wave, B = build_initial_base_plan(b, n)

    # step 4
    x_wave, B = main_phase(c_wave, A_wave, x_wave, B)

    # step 5
    if not is_task_compatible(x_wave, n):
        raise RuntimeError("The task is not compatible.")

    # step 6
    x = form_allowable_plan(x_wave, n)

    while True:
        # step 7
        if is_resolved(B, n):
            return x, B

        # step 8
        k, i = max_index_of_artificial_variable(B, n)

        # step 9, 10
        is_not_lin_depend = correct_B(A_wave, B, n, k)

        # step 11
        if not is_not_lin_depend:
            A, b, B, A_wave = remove_main_restriction_of_task(A, b, B, A_wave, i, k)
