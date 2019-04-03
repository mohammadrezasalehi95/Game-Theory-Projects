import pytest
from PA1.nash_equilibrium import *


def test_find_pure_nash():
    nash = find_nash_equilibrium([[[3, 4], [7, 6], [1, 5]], [[2, 4], [1, 4], [2, 6]]])
    assert nash == [[0, 1], [1, 2]]


def test_powerset():
    assert set(powerset(range(2))) == {(), (1,), (0,), (0, 1)}


def test_mix_powerset():
    assert set(list(mix_nash_powerset(2, 3))) == {((0, 1), (0, 1, 2)), ((0, 1), (1, 2)), ((0, 1), (0, 2)),
                                                  ((0, 1), (0, 1)),
                                                  ((0, 1), (2,)), ((0, 1), (1,)), ((0, 1), (0,)), ((1,), (0, 1, 2)),
                                                  ((1,), (1, 2)), ((1,), (0, 2)), ((1,), (0, 1)), ((0,), (0, 1, 2)),
                                                  ((0,), (1, 2)), ((0,), (0, 2)), ((0,), (0, 1))}
