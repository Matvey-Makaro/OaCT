import numpy as np
from initial_phase import initial_phase


def test_initial_phase() -> None:
    c = np.array([[1],
                  [0],
                  [0]])

    A = np.array([[1, 1, 1],
                  [2, 2, 2]])

    b = np.array([[0],
                  [0]])

    try:
        x, B = initial_phase(c, A, b)
    except RuntimeError as ex:
        print(f"test_initial_phase failed with runtime error: {ex}")
        return

    expected_x = np.array([[0],
                           [0],
                           [0]])
    expected_B = [0]

    if (x != expected_x).any():
        print("test_initial_phase failed")
        return
    if B != expected_B:
        print("test_initial_phase failed")
        return
    print("test_initial_phase OK!")
