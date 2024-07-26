from __future__ import annotations

import pytest

from mlfp import error_messages
from mlfp.linear_algebra import Tensor, ones, zeros


def test_tensor_initialization():
    tensor = Tensor([[1, 2], [3, 4]])
    assert tensor.elements == [[1, 2], [3, 4]]
    assert tensor.shape == (2, 2)


def test_tensor_equality():
    tensor1 = Tensor([[1, 2], [3, 4]])
    tensor2 = Tensor([[1, 2], [3, 4]])
    assert tensor1 == tensor2


def test_tensor_addition():
    tensor1 = Tensor([[1, 2], [3, 4]])
    tensor2 = Tensor([[5, 6], [7, 8]])
    result = tensor1 + tensor2
    assert result.elements == [[6, 8], [10, 12]]


def test_tensor_subtraction():
    tensor1 = Tensor([[5, 6], [7, 8]])
    tensor2 = Tensor([[1, 2], [3, 4]])
    result = tensor1 - tensor2
    assert result.elements == [[4, 4], [4, 4]]


def test_tensor_scalar_multiplication():
    tensor = Tensor([[1, 2], [3, 4]])
    result = tensor * 2
    assert result.elements == [[2, 4], [6, 8]]


def test_tensor_flatten():
    tensor = Tensor([[1, 2], [3, 4]])
    assert tensor.flatten() == [1, 2, 3, 4]


def test_tensor_reshape():
    tensor = Tensor([1, 2, 3, 4])
    reshaped = tensor.reshape((2, 2))
    assert reshaped.elements == [[1, 2], [3, 4]]
    assert reshaped.shape == (2, 2)


def test_tensor_reshape_invalid_shape():
    tensor = Tensor([1, 2, 3, 4])
    with pytest.raises(error_messages.InvalidReshapeShapes):
        tensor.reshape((3, 3))


def test_tensor_matrix_multiplication():
    tensor1 = Tensor([[1, 2], [3, 4]])
    tensor2 = Tensor([[2, 0], [1, 2]])
    result = tensor1 @ tensor2
    assert result.elements == [[4, 4], [10, 8]]


def test_tensor_apply():
    tensor = Tensor([[1, 2], [3, 4]])
    result = tensor.apply(lambda x: x * 2)
    assert result.elements == [[2, 4], [6, 8]]


def test_tensor_zeros():
    tensor = zeros((2, 2))
    assert tensor.elements == [[0, 0], [0, 0]]


def test_tensor_ones():
    tensor = ones((2, 2))
    assert tensor.elements == [[1, 1], [1, 1]]


# def test_tensor_random():
#     tensor = Tensor().random((2, 2))
#     assert tensor.shape == (2, 2)
#     # Note: testing the random distribution is impractical - just check shape for now


def test_tensor_initialization_higher_dimensions():
    # 3D tensor
    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert tensor_3d.elements == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert tensor_3d.shape == (2, 2, 2)

    # 4D tensor
    tensor_4d = Tensor([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]])
    assert tensor_4d.elements == [[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]
    assert tensor_4d.shape == (2, 2, 2, 1)


def test_tensor_operations_higher_dimensions():
    # Addition, subtraction, and scalar multiplication for 3D tensors
    tensor1 = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    tensor2 = Tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])

    # Addition
    result_add = tensor1 + tensor2
    expected_add = Tensor([[[10, 12], [14, 16]], [[18, 20], [22, 24]]])
    assert result_add == expected_add

    # Subtraction
    result_sub = tensor2 - tensor1
    expected_sub = Tensor([[[8, 8], [8, 8]], [[8, 8], [8, 8]]])
    assert result_sub == expected_sub

    # Scalar Multiplication
    result_scalar = tensor1 * 2
    expected_scalar = Tensor([[[2, 4], [6, 8]], [[10, 12], [14, 16]]])
    assert result_scalar == expected_scalar


def test_tensor_flatten_higher_dimensions():
    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert tensor_3d.flatten() == [1, 2, 3, 4, 5, 6, 7, 8]


def test_tensor_reshape_higher_dimensions():
    tensor = Tensor([1, 2, 3, 4, 5, 6, 7, 8])
    reshaped_to_3d = tensor.reshape((2, 2, 2))
    assert reshaped_to_3d.elements == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert reshaped_to_3d.shape == (2, 2, 2)

    # Reshape to 4D tensor
    reshaped_to_4d = tensor.reshape((2, 2, 2, 1))
    assert reshaped_to_4d.elements == [
        [[[1], [2]], [[3], [4]]],
        [[[5], [6]], [[7], [8]]],
    ]
    assert reshaped_to_4d.shape == (2, 2, 2, 1)


def test_tensor_apply_higher_dimensions():
    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    result = tensor_3d.apply(lambda x: x * 2)
    expected = Tensor([[[2, 4], [6, 8]], [[10, 12], [14, 16]]])
    assert result == expected


def test_matmul_with_tensors_bigger_than_2d():
    tensor1 = Tensor([[[1, 2], [3, 4]]])  # 3D tensor
    tensor2 = Tensor([[5, 6], [7, 8]])  # 2D tensor
    with pytest.raises(error_messages.MatMulMaxSizeError):
        tensor1 @ tensor2


def test_matmul_with_incompatible_shapes():
    tensor1 = Tensor([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
    tensor2 = Tensor([[1, 2], [3, 4]])  # Shape: (2, 2)
    with pytest.raises(error_messages.MatMulIncompatibleShapesError):
        tensor1 @ tensor2


def test_addition_with_mismatched_shapes():
    tensor1 = Tensor([[1, 2], [3, 4]])  # Shape: (2, 2)
    tensor2 = Tensor([[1, 2, 3]])  # Shape: (1, 3)
    with pytest.raises(error_messages.ShapeMismatchError):
        tensor1 + tensor2


if __name__ == "__main__":
    pytest.main()
