import numpy as np
from main_phase import *


def test_build_basis_matrix() -> None:
    A = np.array([[-1, 1, 1, 0, 0],
                  [1, 0, 0, 1, 0],
                  [0, 1, 0, 0, 1]])

    B = [2, 3, 4]

    expected_basis_matrix = np.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]])
    basis_matrix = build_basis_matrix(A, B)
    if (basis_matrix != expected_basis_matrix).any():
        print("test_build_basis_matrix failed!")
        exit()
    print("test_build_basis_matrix OK")


def test_build_basis_vector() -> None:
    c = np.array([[1],
                  [1],
                  [0],
                  [0],
                  [0]])
    B = [2, 3, 4]
    expected = np.array([[0],
                         [0],
                         [0]])

    basis_vector = build_basis_vector(c, B)
    if (basis_vector != expected).any():
        print("test_build_basis_matrix failed!")
        exit()
    print("test_build_basis_vector OK")


def test_build_potential_vector() -> None:
    basis_c = np.array([[0],
                        [0],
                        [0]])
    basis_matrix = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])
    inv_basis_matrix = np.linalg.inv(basis_matrix)
    expected = np.array([[0],
                        [0],
                        [0]])

    potential_vector = build_potential_vector(basis_c, inv_basis_matrix)
    if (potential_vector != expected).any():
        print("test_build_basis_matrix failed!")
        exit()
    print("test_build_potential_vector OK")


def test_main_phase() -> None:
    c = np.array([[1],
                  [1],
                  [0],
                  [0],
                  [0]])

    A = np.array([[-1, 1, 1, 0, 0],
                  [1, 0, 0, 1, 0],
                  [0, 1, 0, 0, 1]])

    x = np.array([[0],
                  [0],
                  [1],
                  [3],
                  [2]])

    B = [2, 3, 4]

    try:
        res = main_phase(c, A, x, B)
    except NotLimitedError as ex:
        print(ex)
        print("test_main_phase failed!")
        exit()

    expected = np.array([[3],
                      [2],
                      [2],
                      [0],
                      [0]])

    if (res != expected).any():
        print("test_main_phase failed!")
        exit()
    print("test_main_phase OK")



def run_tests() -> None:
    test_build_basis_matrix()
    test_build_basis_vector()
    test_build_potential_vector()
    test_main_phase()