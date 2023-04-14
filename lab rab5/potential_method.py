import numpy as np


class Node:
    def __init__(self, i: int, j: int, value: int, sign: str):
        self.i = i
        self.j = j
        self.value = value
        self.sign = sign


def northwest_node_method(a: np.ndarray, b: np.ndarray) -> (np.ndarray, list):
    # step 1
    m = a.shape[0]
    n = b.shape[0]
    x = np.zeros((m, n))
    B = []
    i = 0
    j = 0
    a_copy = a.copy()
    b_copy = b.copy()

    # step 2
    while i < m and j < n:
        B.append((i, j))
        if a_copy[i] < b_copy[j]:
            x[i, j] = a_copy[i]
            b_copy[j] -= a_copy[i]
            i += 1
        else:
            x[i, j] = b_copy[j]
            a_copy[i] -= b_copy[j]
            j += 1

    return x, B


def find_u_v(B: list, c: np.ndarray) -> np.ndarray:
    m = c.shape[0]
    n = c.shape[1]
    A = np.zeros((m + n, m + n))
    rhs = []
    # In A, the first m elements are u and then v.
    for k in range(len(B)):
        i, j = B[k]
        if i != 0:
            A[k, i] = 1
        A[k, m + j] = 1
        rhs.append(c[i, j])

    A[len(B), 0] = 1
    rhs.append(0)
    u_v = np.linalg.solve(A, rhs)
    return u_v


def find_new_base_pos(B: list, c: np.ndarray, u_v: np.ndarray) -> tuple or None:
    m = c.shape[0]
    n = c.shape[1]
    for i in range(m):
        for j in range(n):
            if (i, j) in B:
                continue
            if u_v[i] + u_v[m + j] > c[i, j]:
                return i, j
    return None


def is_problem_correct(a: np.ndarray, b: np.ndarray) -> bool:
    sum_a = 0
    for i in range(a.shape[0]):
        sum_a += a[i]
    sum_b = 0
    for j in range(b.shape[0]):
        sum_b += b[j]
    return sum_a == sum_b


def delete_row_in_B(B: list, row: int) -> list:
    return [(i, j) for (i, j) in B if i != row]


def delete_col_in_B(B: list, col: int) -> list:
    return [(i, j) for (i, j) in B if j != col]


def delete_non_corner_vertices(B: list, m: int, n: int) -> list:
    is_deleted = True
    num_in_row = 0
    while is_deleted:
        is_deleted = False
        for k in range(m):
            for i, j in B:
                if i == k:
                    num_in_row += 1
            if num_in_row ==1:
                B = delete_row_in_B(B, k)
                is_deleted = True
            num_in_row = 0

        num_in_col = 0
        for k in range(n):
            for i, j in B:
                if j == k:
                    num_in_col += 1
            if num_in_col == 1:
                B = delete_col_in_B(B, k)
                is_deleted = True
            num_in_col = 0
    return B


def get_next_sign(cur_sign: str) -> str:
    if cur_sign == '+':
        return '-'
    elif cur_sign == '-':
        return '+'
    else:
        raise RuntimeError('Unknown cur_sign')


def build_graph(B: list, x: np.ndarray, new_base_pos: tuple) -> list:
    new_B = [[i, j, False] for (i, j) in B]
    new_base_i, new_base_j = new_base_pos
    next_sign = '+'
    nodes = [Node(new_base_i, new_base_j, x[new_base_i, new_base_j], next_sign)]
    next_sign = get_next_sign(next_sign)
    prev_i = new_base_i
    prev_j = new_base_j

    is_end = False
    while not is_end:
        for k in range(len(new_B)):
            i, j, is_visited = new_B[k]
            if (i == prev_i and j != prev_j) or (j == prev_j and i != prev_i):
                if is_visited:
                    continue
                if i == new_base_i and j == new_base_j:
                    is_end = True
                    break
                nodes.append(Node(i, j, x[i, j], next_sign))
                next_sign = get_next_sign(next_sign)
                prev_i = i
                prev_j = j
                new_B[k][2] = True
                break
    return nodes


def get_theta(graph: list) -> (int, int):
    min = graph[1].value
    index = 1
    for i in range(3, len(graph), 2):
        if graph[i].value < min:
            min = graph[i].value
            index = i
    return min, index


def apply_theta(graph: list, theta: int) -> None:
    for n in graph:
        if n.sign == '+':
            n.value += theta
        else:
            n.value -= theta


def update_x(x: np.ndarray, graph: list) -> None:
    for n in graph:
        x[n.i, n.j] = n.value


def potential_method(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    if not is_problem_correct(a, b):
        raise RuntimeError("Problem is not correct. sum_a != sum_b.")

    # phase 1
    x, B = northwest_node_method(a, b)

    # phase 2
    while True:
        u_v = find_u_v(B, c)

        new_base_pos = find_new_base_pos(B, c, u_v)
        if new_base_pos is None:
            return x

        m = c.shape[0]
        n = c.shape[1]
        B.append(new_base_pos)
        B_without_non_corn_vert = delete_non_corner_vertices(B.copy(), m, n)
        graph = build_graph(B_without_non_corn_vert, x, new_base_pos)
        theta, index = get_theta(graph)
        apply_theta(graph, theta)
        update_x(x, graph)
        B.remove((graph[index].i, graph[index].j))
