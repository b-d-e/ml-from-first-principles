from __future__ import annotations

import pytest

from mlfp.linear_algebra import Matrix, ones, zeros


def test_empty_matrix_creation():
    matrix = Matrix([])
    assert matrix.elements == [], "Empty matrix creation failed."


def test_matrix_creation_with_negative_numbers():
    matrix = Matrix([[-1, -2], [-3, -4]])
    assert matrix.elements == [
        [-1, -2],
        [-3, -4],
    ], "Matrix creation with negative numbers failed."


def test_matrix_addition_with_negative_numbers():
    matrix1 = Matrix([[1, -2], [-3, 4]])
    matrix2 = Matrix([[-5, 6], [7, -8]])
    result = matrix1 + matrix2
    assert result.elements == [
        [-4, 4],
        [4, -4],
    ], "Matrix addition with negative numbers failed."


def test_non_square_matrix_transpose():
    matrix = Matrix([[1, 2, 3], [4, 5, 6]])
    result = matrix.transpose()
    assert result.elements == [
        [1, 4],
        [2, 5],
        [3, 6],
    ], "Transpose of non-square matrix failed."


def test_matrix_multiplication_with_empty_matrix():
    matrix1 = Matrix([[1, 2], [3, 4]])
    matrix2 = Matrix([])
    with pytest.raises(ValueError, match="Matrices must have compatible shapes"):
        matrix1 @ matrix2


def test_matrix_addition_with_mismatched_dimensions():
    matrix1 = Matrix([[1, 2], [3, 4]])
    matrix2 = Matrix([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="Matrices must be of the same shape"):
        matrix1 + matrix2


def test_scalar_multiplication_with_zero():
    matrix = Matrix([[1, 2], [3, 4]])
    result = matrix * 0
    assert result.elements == [
        [0, 0],
        [0, 0],
    ], "Scalar multiplication with zero failed."


def test_ones_matrix_creation():
    matrix = ones((2, 3))
    assert matrix.elements == [[1, 1, 1], [1, 1, 1]], "Ones matrix creation failed."


def test_zeros_matrix_creation():
    matrix = zeros((2, 3))
    assert matrix.elements == [[0, 0, 0], [0, 0, 0]], "Zeros matrix creation failed."


if __name__ == "__main__":
    pytest.main()
