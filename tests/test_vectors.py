# tests/test_example.py
from __future__ import annotations

import pytest

from mlfp.linear_algebra import Vector


def test_vector_creation() -> None:
    new_vector = Vector([1, 2, 3, 4])
    assert isinstance(new_vector, Vector)


def test_element_presence() -> None:
    elements = [1, 2, 3, 4]
    new_vector = Vector([1, 2, 3, 4])
    assert elements == new_vector.__elements__()


def test_scalar_multiplication() -> None:
    scalars = [0, 1, -1, 10]
    vectors = [Vector([1, 1, 1, 1]), Vector([0, 0, 0]), Vector([-1, -2, -3, -4])]
    results = [
        [[0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0]],
        [[1, 1, 1, 1], [0, 0, 0], [-1, -2, -3, -4]],
        [[-1, -1, -1, -1], [0, 0, 0], [1, 2, 3, 4]],
        [[10, 10, 10, 10], [0, 0, 0], [-10, -20, -30, -40]],
    ]

    for i, scalar in enumerate(scalars):
        for j, vector in enumerate(vectors):
            assert results[i][j] == (vector * scalar).__elements__()
            assert results[i][j] == (scalar * vector).__elements__()
            assert results[i][j] == (vector.__mul__(scalar)).__elements__()
            assert results[i][j] == (vector.__rmul__(scalar)).__elements__()
            assert (vector * scalar).__elements__() == (scalar * vector).__elements__()


if __name__ == "__main__":
    pytest.main()
