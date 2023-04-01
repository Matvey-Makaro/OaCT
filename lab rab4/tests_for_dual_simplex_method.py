import numpy as np
from dual_simplex_method import dual_simplex_method

def test_example() -> None:
    c = np.array([[-4],
                  [-3],
                  [-7],
                  [0],
                  [0]])

    A = np.array([[-2, -1, -4, 1, 0],
                 [-2, -2, -2, 0, 1]])

    b = np.array([[-1],
                  [-3/2]])

    B = [3, 4]

    expected_x = np.array([[1/4],
                           [1/2],
                           [0],
                           [0],
                           [0]])

    try:
        x = dual_simplex_method(c, A, b, B)
    except RuntimeError as ex:
        print(f"test_example failed! Unexpected runtime error: {ex}")
        return

    if (x != expected_x).any():
        print("test_example failed!")
        return

    print("test_example OK!")

