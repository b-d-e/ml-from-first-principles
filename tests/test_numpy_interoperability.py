from __future__ import annotations

import numpy as np
import pytest

import mlfp.linear_algebra as la


def test_zeros():
    shape = (2, 3)
    tensor_zeros = la.zeros(shape)
    numpy_zeros = np.zeros(shape)
    assert np.allclose(tensor_zeros.to_numpy(), numpy_zeros)


def test_ones():
    shape = (2, 3)
    tensor_ones = la.ones(shape)
    numpy_ones = np.ones(shape)
    assert np.allclose(tensor_ones.to_numpy(), numpy_ones)


def test_addition():
    tensor_a = la.Tensor([[1, 2], [3, 4]])
    tensor_b = la.Tensor([[5, 6], [7, 8]])
    tensor_result = tensor_a + tensor_b

    numpy_a = np.array([[1, 2], [3, 4]])
    numpy_b = np.array([[5, 6], [7, 8]])
    numpy_result = numpy_a + numpy_b

    assert np.allclose(tensor_result.to_numpy(), numpy_result)


def test_multiplication():
    tensor = la.Tensor([[1, 2], [3, 4]])
    scalar = 3
    tensor_result = tensor * scalar

    numpy_array = np.array([[1, 2], [3, 4]])
    numpy_result = numpy_array * scalar

    assert np.allclose(tensor_result.to_numpy(), numpy_result)


def test_matrix_multiplication():
    tensor_a = la.Tensor([[1, 2], [3, 4]])
    tensor_b = la.Tensor([[5, 6], [7, 8]])
    tensor_result = tensor_a @ tensor_b

    numpy_a = np.array([[1, 2], [3, 4]])
    numpy_b = np.array([[5, 6], [7, 8]])
    numpy_result = numpy_a @ numpy_b

    assert np.allclose(tensor_result.to_numpy(), numpy_result)


def test_reshape():
    tensor = la.Tensor([[1, 2, 3, 4]])
    new_shape = (2, 2)
    tensor_reshaped = tensor.reshape(new_shape)

    numpy_array = np.array([[1, 2, 3, 4]])
    numpy_reshaped = numpy_array.reshape(new_shape)

    assert np.allclose(tensor_reshaped.to_numpy(), numpy_reshaped)


if __name__ == "__main__":
    pytest.main()
