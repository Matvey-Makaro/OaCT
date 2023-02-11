import random
from main import matrix_reversal
import numpy as np


def stupid_matrix_reversal(A: np.ndarray) -> (bool, np.ndarray):
    try:
        rev_A = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return False, np.array([])
    return True, rev_A


def print_terms(A: np.ndarray, rev_A: np.ndarray, x: np.ndarray, i: int, stupid_A: np.ndarray) -> None:
    print("A", A, sep='\n')
    print("rev_A", rev_A, sep='\n')
    print("x", x, sep='\n')
    print("i: ", i)
    print("stupid_A", stupid_A, sep='\n')


def print_results(res_A: np.ndarray, is_rev: bool) -> None:
    print("is_rev: ", is_rev)
    if is_rev:
        print("res_A: ", res_A, sep='\n')


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

        stupid_A = A.copy()
        for j in range(0, len(A)):
            stupid_A[j, i] = x[j]

        is_rev_stupid, res_A_stupid = stupid_matrix_reversal(stupid_A)
        is_rev, res_A = matrix_reversal(rev_A, x, i)

        if is_rev_stupid != is_rev:
            is_correct = False
            print("is_rev_stupid != is_rev")
        if not np.array_equal(res_A_stupid, res_A):
            is_correct = False
            print("res_A_stupid != res_A")
        if not is_correct:
            print_terms(A, rev_A, x, i, stupid_A)
            print("Stupid:")
            print_results(res_A_stupid, is_rev_stupid)
            print("Smart:")
            print_results(res_A, is_rev)
            break









