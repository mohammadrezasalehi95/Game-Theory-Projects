import numpy as np
from scipy.optimize import linprog


def main(table):
    A, B, C = np.array(table[0]), np.array(table[1]), np.array(table[2])
    n, m, p = A.shape[0], A.shape[1], B.shape[1]
    combined1 = np.array(np.zeros(shape=(n + m + p, n + m + p, 2)))
    combined1[0:n, n:m + n, :] = A
    combined1[0:n, n + m:n + m + p, :] = B
    combined1[n:n + m, n + m:n + m + p, :] = C
    combined1 = combined1 + game_table_transpose(combined1)
    X = min_max(table=combined1, axis=0, n=n, m=m, p=p)
    return [X[0:n], X[n:n + m], X[n + m:n + m + p]]


def min_max(table, axis, n, m, p):
    over = 1 if axis == 0 else 0
    vars_len = table.shape[axis]
    enemy_len = table.shape[over]
    all_minus = np.array([-1] * enemy_len)
    c = ([0] * (vars_len)) + [-1]
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    for i in range(enemy_len):
        extract_axis = [slice(None), slice(None), 0]
        extract_axis[axis] = i
        extract_axis = tuple(extract_axis)
        A_ub += [(all_minus.T * table[extract_axis]).tolist() + [1]]
        b_ub += [0]
    A_eq += [[1] * vars_len + [0]]
    b_eq += [3]
    A_eq += [[1] * n + [0] * m + [0] * p + [0]]
    b_eq += [1]
    A_eq += [[0] * n + [1] * m + [0] * p + [0]]
    b_eq += [1]
    A_eq += [[0] * n + [0] * m + [1] * p + [0]]
    b_eq += [1]

    for i in range(vars_len):
        temp = [0] * (vars_len + 1)
        temp[i] = -1
        A_ub += [temp]

        b_ub += [0]
    result = linprog(c=c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(-50, 50), method='interior-point').x
    print(result)
    return result.tolist()


def game_table_transpose(combined1):
    return combined1.transpose([1, 0, 2])[:, :, ::-1]


print(main([[[[2, -2], [-3, 3]], [[-3, 3], [2, -2]]],
            [[[4, -4], [-1, 1], [-3, 3]], [[-4, 4], [1, -1], [3, -3]]],
            [[[-5, 5], [2, -2], [3, -3]], [[5, -5], [-2, 2], [-3, 3]]]]))
