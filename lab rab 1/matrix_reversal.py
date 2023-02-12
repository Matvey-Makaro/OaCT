import numpy as np


def step_1(inv_A: np.ndarray, x: np.ndarray, i: int) -> (bool, np.ndarray):
    l = inv_A.dot(x)
    if l[i] == 0:
        return False, l
    return True, l


def step_2(l: np.ndarray, i: int) -> np.ndarray:
    l_wave = l.copy()
    l_wave[i] = -1
    return l_wave


def step_3(l: np.ndarray, l_wave: np.ndarray, i: int) -> np.ndarray:
    return (-1 / l[i]) * l_wave


def step_4(l_lid: np.ndarray, i: int) -> np.ndarray:
    Q = np.eye(len(l_lid))
    for j in range(0, len(l_lid)):
        Q[j, i] = l_lid[j]
    return Q


def step_5(Q: np.ndarray, rev_A: np.ndarray) -> np.ndarray:
    return Q.dot(rev_A)


def reverse_matrix(inv_A: np.ndarray, x: np.ndarray, i: int) -> (bool, np.ndarray):
    is_rev, l = step_1(inv_A, x, i)
    if not is_rev:
        return False, np.array([])

    l_wave = step_2(l, i)
    l_lid = step_3(l, l_wave, i)
    Q = step_4(l_lid, i)
    res_A = step_5(Q, inv_A)
    return True, res_A
