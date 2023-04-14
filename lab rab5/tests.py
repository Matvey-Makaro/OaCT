import numpy as np
from potential_method import potential_method


def simple_test() -> None:
    a = np.array([100, 300, 300])
    b = np.array([300, 200, 200])
    c = np.array([[8, 4, 1],
                  [8, 4, 3],
                  [9, 7, 5]])

    x_expected = np.array([[0, 0, 100],
                           [0, 200, 100],
                           [300, 0, 0]])

    x = potential_method(a, b, c)

    if (x != x_expected).any():
        print("simple_test failed.")
    print("simple_test OK!")


def test_from_website() -> None:
    a = np.array([80, 60, 30, 60])
    b = np.array([10, 30, 40, 50, 70, 30])
    c = np.array([[3, 20, 8, 13, 4, 100],
                  [4, 4, 18, 14, 3, 0],
                  [10, 4, 18, 8, 6, 0],
                  [7, 19, 17, 10, 1, 100]])

    x_expected = np.array([[10, 0, 40, 20, 10, 0],
                           [0, 30, 0, 0, 0, 30],
                           [0, 0, 0, 30, 0, 0],
                           [0, 0, 0, 0, 60, 0]])

    x = potential_method(a, b, c)

    if (x != x_expected).any():
        print("test_from_website failed.")
    print("test_from_website OK!")


def run_tests() -> None:
    simple_test()
    test_from_website()
