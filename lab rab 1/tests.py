import random
from matrix_reversal import reverse_matrix
import numpy as np
import test_cases


def print_terms(A: np.ndarray, inv_A: np.ndarray, x: np.ndarray, i: int, stupid_A: np.ndarray) -> None:
    print("A", A, sep='\n')
    print("inv_A", inv_A, sep='\n')
    print("x", x, sep='\n')
    print("i: ", i)
    print("stupid_A", stupid_A, sep='\n')


def print_results(res_A: np.ndarray, is_inv: bool) -> None:
    print("is_inv: ", is_inv)
    if is_inv:
        print("res_A: ", res_A, sep='\n')


def stupid_matrix_reversal(A: np.ndarray) -> (bool, np.ndarray):
    try:
        inv_A = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return False, np.array([])
    return True, inv_A


def test_matrix_reversal(A: np.ndarray, inv_A: np.ndarray, x: np.ndarray, i: int) -> bool:
    stupid_A = A.copy()
    for j in range(0, len(A)):
        stupid_A[j, i] = x[j]

    is_inv_stupid, res_A_stupid = stupid_matrix_reversal(stupid_A)
    is_inv, res_A = reverse_matrix(inv_A, x, i)

    is_correct = True
    if is_inv_stupid != is_inv:
        is_correct = False
        print("is_inv_stupid != is_inv")
    if not np.isclose(res_A_stupid, res_A, rtol=1e-16).all():
        is_correct = False
        print("res_A_stupid != res_A")
    if not is_correct:
        print_terms(A, inv_A, x, i, stupid_A)
        print("Stupid:")
        print_results(res_A_stupid, is_inv_stupid)
        print("\nSmart:")
        print_results(res_A, is_inv)

    return is_correct


def stress_test(num_of_iter: int) -> None:
    is_correct = True
    for k in range(0, num_of_iter):
        size = random.randint(2, 100)
        A = np.random.randint(-100, 100, (size, size))
        try:
            rev_A = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            k -= 1
            continue

        x = np.random.randint(-100, 100, (size, 1))
        i = random.randint(0, size - 1)

        if not test_matrix_reversal(A, rev_A, x, i):
            is_correct = False
            break

    if is_correct:
        print("Stress test OK")


def test_congenital_matrix() -> None:
    A, inv_A, x, i = test_cases.get_congenital_matrix_testcase()
    if not test_matrix_reversal(A, inv_A, x, i):
        print("test_congenital_matrix failed")
    print("test_congenital_matrix OK")


def test_simple_matrix() -> None:
    A, inv_A, x, i = test_cases.get_simple_testcase()
    if not test_matrix_reversal(A, inv_A, x, i):
        print("test_simple_matrix failed")
    print("test_simple_matrix OK")
