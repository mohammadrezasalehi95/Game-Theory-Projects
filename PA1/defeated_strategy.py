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



def find_nash_equilibrium(table):
    table = np.array(table)
    column_table = table[:, :, 1]
    row_table = table[:, :, 0]
    max_column_player = np.max(column_table, axis=1)
    max_row_player = np.max(row_table, axis=0)
    pure_nash_x_axis = np.zeros(row_table.shape, dtype=int)
    pure_nash_y_axis = np.zeros(row_table.shape, dtype=int)
    for (x, y), value in np.ndenumerate(table[:, :, 0]):
        pure_nash_x_axis[x, y] = 1 if value == max_row_player[y] else 0

    for (x, y), value in np.ndenumerate(table[:, :, 1]):
        pure_nash_y_axis[x, y] = 1 if value == max_column_player[x] else 0

    return (pure_nash_y_axis, pure_nash_x_axis)


def main(table):
    all_nash_equilibriums = find_nash_equilibrium(table)
    p1_mixed_strategy, p2_mixed_strategy = find_mixed_nash_equilibrium(table)
    return [all_nash_equilibriums, p1_mixed_strategy, p2_mixed_strategy]

a=np.array([4,4,4,[]])
print(a.__str__())