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


def test_vector_addition() -> None:
    vectors_pairs = [
        (Vector([1, 2, 3]), Vector([4, 5, 6]), Vector([5, 7, 9])),
        (Vector([0, 0, 0]), Vector([1, 1, 1]), Vector([1, 1, 1])),
        (Vector([-1, -2, -3]), Vector([1, 2, 3]), Vector([0, 0, 0])),
        (Vector([10, 20]), Vector([-10, -20]), Vector([0, 0])),
    ]

    for v1, v2, expected in vectors_pairs:
        result = v1 + v2
        assert (
            expected.__elements__() == result.__elements__()
        ), f"Expected {expected.__elements__()}, got {result.__elements__()}"


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


def test_dot_product() -> None:
    vectors = [
        (Vector([1, 2, 3]), Vector([4, 5, 6]), 32),
        (Vector([0, 0, 0]), Vector([1, 1, 1]), 0),
        (Vector([-1, -2, -3]), Vector([1, 2, 3]), -14),
        (Vector([10, 20]), Vector([-10, -20]), -500),
    ]

    for v1, v2, expected in vectors:
        result = v1.dot(v2)
        assert expected == result, f"Expected {expected}, got {result}"


def test_vector_length_mismatch() -> None:
    with pytest.raises(ValueError) as e:
        Vector([1, 2, 3]) + Vector([1, 2])
    assert str(e.value) == "Vectors must be of the same length"

    with pytest.raises(ValueError) as e:
        Vector([1, 2, 3]) - Vector([1, 2])
    assert str(e.value) == "Vectors must be of the same length"

    with pytest.raises(ValueError) as e:
        Vector([1, 2, 3]).dot(Vector([1, 2]))
    assert str(e.value) == "Vectors must be of the same length"


if __name__ == "__main__":
    pytest.main()
