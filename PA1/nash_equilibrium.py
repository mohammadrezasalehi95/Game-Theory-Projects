import itertools
from itertools import chain, combinations

from scipy.optimize import linprog
import numpy as np


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def mix_nash_powerset(x, y):
    x_powerset = reversed(list(powerset(range(x))))
    y_powerset = reversed(list(powerset(range(y))))
    for i, j in itertools.product(x_powerset, y_powerset):
        if len(i) + len(j) > 2 and i and j:
            yield i, j


def giveProperTable(table, row_profile, column_profile):
    temp = table.copy()
    temp = np.delete(temp, row_profile, 0)
    temp = np.delete(temp, column_profile, 1)
    return temp


def find_mix_equilibrium_specified_strategy(table, row_profile, column_profile, row_size, column_size):
    not_row_profile = set(range(row_size)) - set(row_profile)
    not_column_profile = set(range(column_size)) - set(column_profile)
    row_profile_size = len(row_profile)
    column_profile_size = len(column_profile)
    c = [0] * (row_profile_size + column_profile_size)
    A_eq = []
    b_eq = []
    A_ub = []
    b_ub = []
    for row in row_profile:
        temp = []
        for column in column_profile:
            temp.append(table[row, column, 0] - table[row_profile[0], column, 0])
        temp += ([0] * row_profile_size)
        A_eq += [temp]
        b_eq.append(0)

    for column in column_profile:
        temp = [0] * column_profile_size
        for row in row_profile:
            temp.append(table[row, column, 1] - table[row, column_profile[0], 1])
        A_eq += [temp]
        b_eq.append(0)

    for row in not_row_profile:
        temp = []
        for column in column_profile:
            temp.append(table[row, column, 0] - table[row_profile[0], column, 0])
        temp += ([0] * row_profile_size)
        A_ub += [temp]
        b_ub += [0]

    for column in not_column_profile:
        temp = [0] * column_profile_size
        for row in row_profile:
            temp.append(table[row, column, 1] - table[row, column_profile[0], 1])
        A_ub += [temp]
        b_ub += [0]

    A_eq += [[1] * column_profile_size + [0] * row_profile_size]
    A_eq += [[0] * column_profile_size + [1] * row_profile_size]
    b_eq += [1, 1]
    # tabled = giveProperTable(table, list(not_row_profile), list(not_column_profile))
    tabled = table[row_profile, column_profile, :]
    a = linprog(c, A_eq=A_eq, b_eq=b_eq,
                A_ub=A_ub if A_ub else None,
                b_ub=b_ub if b_ub else None,
                )
    b = find_nash_equilibrium(tabled)

    return a.x, (a.success and not b)


def find_mixed_nash_equilibrium(table):
    p1_strategy = []
    p2_strategy = []
    table = np.array(table)
    row_size, column_size, _ = table.shape
    for row_s_profile, column_s_profile in mix_nash_powerset(row_size, column_size):
        answer = find_mix_equilibrium_specified_strategy(table, row_s_profile, column_s_profile, row_size, column_size)
        if answer[1]:
            p1_strategy = [0] * row_size
            p2_strategy = [0] * column_size
            for i, j in zip(column_s_profile, answer[0][:len(column_s_profile)].tolist()):
                p2_strategy[i] = j
            for i, j in zip(row_s_profile, answer[0][len(column_s_profile):].tolist()):
                p1_strategy[i] = j

    return p1_strategy, p2_strategy


def find_nash_equilibrium(table):
    table = np.array(table)
    column_table = table[:, :, 1]
    row_table = table[:, :, 0]
    max_column_player = np.max(column_table, axis=1)
    max_row_player = np.max(row_table, axis=0)
    pure_nash = np.zeros(row_table.shape, dtype=int)
    for (x, y), value in np.ndenumerate(table[:, :, 0]):
        pure_nash[x, y] = 1 if value == max_row_player[y] else 0
    for (x, y), value in np.ndenumerate(table[:, :, 1]):
        pure_nash[x, y] &= 1 if value == max_column_player[x] else 0
    output = [[a[0] + 1, a[1] + 1] for a, v in np.ndenumerate(pure_nash) if v]
    return output


def main(table):
    all_nash_equilibriums = find_nash_equilibrium(table)
    p1_mixed_strategy, p2_mixed_strategy = find_mixed_nash_equilibrium(table)
    return [all_nash_equilibriums, p1_mixed_strategy, p2_mixed_strategy]


#
# main([[[3, 4], [7, 6], [1, 5]], [[2, 4], [1, 4], [2, 6]]])
# table = [
#     [[1, -1], [-1, 1]],
#     [[-1, 1], [1, -1]]
# ]
# table2 = [
#     [[9, 1], [2, 8]],
#     [[3, 7], [6, 4]]
# ]
# table3 = [
#     [[2, -3], [1, 2]],
#     [[1, 1], [4, -1]]
# ]
# table4 = [
#     [[3, 1], [1, 2]],
#     [[2, 3], [3, 4]],
#     [[2, 4], [3, 1]],
# ]
#
# print(main(table4))
# table = np.array(table)
# table2 = np.array(table2)
# table3 = np.array(table3)
# table4 = np.array(table4)
print("new test")
print(main([[[3, 2], [2, 2], [3, 1]],
            [[2, 4], [3, 1], [1, 3]],
            [[3, 1], [3, 3], [2, 4]],
            [[4, 4], [3, 3], [3, 1]]]))
print([[[4, 1]], [0, 0, 0.5, 0.5], [0, 1.0, 0]])
print("new test")
print(main([[[3, 1], [3, 3], [2, 3], [2, 1]],
            [[3, 4], [3, 1], [1, 4], [3, 2]],
            [[3, 1], [3, 2], [4, 2], [2, 4]]]))
print([[[1, 2], [2, 1]], [0, 0.5, 0.5], [0, 0, 0.25, 0.75]])
print("new test")
print(main([[[3, 2], [4, 1]],
            [[4, 2], [3, 4]],
            [[3, 1], [4, 3]]]))
print([[[3, 2]], [0.6666666666666666, 0.3333333333333333, 0], [0.5, 0.5]])
print("new test")
print(main([[[3, 2], [3, 4], [3, 2]],
            [[3, 3], [1, 2], [3, 4]],
            [[3, 3], [3, 4], [1, 1]]]))
print([[[1, 2], [2, 3], [3, 2]], [0, 0.5, 0.5], [1.0, 0, 0]])
print("new test")
print(main([[[1, 4], [4, 3], [3, 3], [4, 1]],
            [[4, 4], [1, 3], [3, 1], [2, 4]],
            [[4, 2], [4, 2], [4, 3], [3, 4]]]))
print([[[2, 1]], [0.33333333333333337, 0, 0.6666666666666666], [0, 0, 0.5, 0.5]])
print("new test")
print(main([[[2, 1], [4, 2], [1, 2]],
            [[1, 1], [1, 4], [3, 4]],
            [[2, 3], [4, 3], [2, 3]],
            [[3, 4], [4, 3], [1, 1]]]))
print([[[1, 2], [2, 3], [3, 2], [4, 1]], [0, 0, 1.0, 0], [0.5, 0, 0.5]])
print("new test")
print(main([[[4, 4], [2, 3], [3, 4], [4, 1]],
            [[1, 3], [3, 2], [4, 3], [3, 4]],
            [[4, 1], [2, 3], [1, 2], [3, 3]]]))
print([[[1, 1]], [0.25, 0.75, 0], [0, 0, 0.5, 0.5]])
print("new test")
print(main([[[3, 4], [1, 4], [1, 2], [3, 3]],
            [[3, 2], [4, 2], [2, 4], [3, 3]]]))
print([[[1, 1], [2, 3]], [0.5, 0.5], [0, 0, 0, 1.0]])
print("new test")

print(main([[[2, 1], [3, 2], [1, 3]],
            [[2, 4], [2, 1], [4, 1]],
            [[1, 4], [3, 3], [1, 1]]]))
print([[[2, 1]], [0.5, 0, 0.5], [0, 1.0, 0]])
print("new test")

print(main([[[1, 2], [4, 3], [2, 3]],
            [[1, 4], [2, 3], [3, 3]],
            [[2, 1], [4, 1], [2, 2]]]))
print([[[1, 2]], [0, 0.5, 0.5], [0.5, 0, 0.5]])
