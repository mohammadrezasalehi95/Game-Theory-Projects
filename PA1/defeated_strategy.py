import itertools
from scipy.optimize import linprog
import numpy as np


def without_defeated_strategy(table):
    table = np.array(table)
    while (True):
        temp = sum(table.shape)
        table = shrink(table)
        if temp == sum(table.shape):
            break
    return table.tolist()


def is_defeated_strategy(table, axis, i, dimension=2):
    c = [0] * table.shape[axis]
    A_ub = []
    b_ub = []
    A_eq = [[1] * table.shape[axis]]
    b_eq = [1]
    epsilon = 0.0001
    for block in itertools.product(*(range(table.shape[i]) for i in range(dimension) if i != axis)):
        temp = list(block)
        temp.insert(axis, slice(None))
        temp.append(axis)
        extracted_axis = table[tuple(temp)]
        A_ub.append(list(map(lambda x: -x, extracted_axis)))
        temp[axis] = i
        b_ub.append(-table[tuple(temp)] - epsilon)
    return linprog(c=c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub).success


def shrink(table):
    dimension = table.shape[-1]
    for axis in range(dimension):
        for i in reversed(range(table.shape[axis])):
            if (is_defeated_strategy(table=table, axis=axis, i=i, dimension=dimension)):
                table = np.delete(arr=table, axis=axis, obj=i)
    return table


def main(table):
    good_game = without_defeated_strategy(table)
    return good_game
