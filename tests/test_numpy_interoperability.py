# from __future__ import annotations

# import numpy as np
# import pytest

# from mlfp.linear_algebra import Matrix, Vector


# def test_vector_creation():
#     data = [1, 2, 3]
#     vector = Vector(data)
#     np_array = np.array(data)
#     assert np.allclose(
#         vector.to_numpy(), np_array
#     ), "Vector creation does not match NumPy array."


# def test_matrix_creation():
#     data = [[1, 2], [3, 4]]
#     matrix = Matrix(data)
#     np_array = np.array(data)
#     assert np.allclose(
#         matrix.to_numpy(), np_array
#     ), "Matrix creation does not match NumPy array."


# def test_vector_addition():
#     data1 = [1, 2, 3]
#     data2 = [4, 5, 6]
#     vector1 = Vector(data1)
#     vector2 = Vector(data2)
#     np_array1 = np.array(data1)
#     np_array2 = np.array(data2)
#     assert np.allclose(
#         (vector1 + vector2).to_numpy(), np_array1 + np_array2
#     ), "Vector addition does not match NumPy."


# def test_matrix_addition():
#     data1 = [[1, 2], [3, 4]]
#     data2 = [[5, 6], [7, 8]]
#     matrix1 = Matrix(data1)
#     matrix2 = Matrix(data2)
#     np_array1 = np.array(data1)
#     np_array2 = np.array(data2)
#     assert np.allclose(
#         (matrix1 + matrix2).to_numpy(), np_array1 + np_array2
#     ), "Matrix addition does not match NumPy."


# def test_vector_scalar_multiplication():
#     data = [1, 2, 3]
#     scalar = 2
#     vector = Vector(data)
#     np_array = np.array(data)
#     assert np.allclose(
#         (vector * scalar).to_numpy(), np_array * scalar
#     ), "Vector scalar multiplication does not match NumPy."


# def test_matrix_scalar_multiplication():
#     data = [[1, 2], [3, 4]]
#     scalar = 2
#     matrix = Matrix(data)
#     np_array = np.array(data)
#     assert np.allclose(
#         (matrix * scalar).to_numpy(), np_array * scalar
#     ), "Matrix scalar multiplication does not match NumPy."


# def test_vector_dot_product():
#     data1 = [1, 2, 3]
#     data2 = [4, 5, 6]
#     vector1 = Vector(data1)
#     vector2 = Vector(data2)
#     np_array1 = np.array(data1)
#     np_array2 = np.array(data2)
#     assert np.allclose(
#         vector1 @ vector2, np_array1 @ np_array2
#     ), "Vector dot product does not match NumPy."


# def test_matrix_dot_product():
#     data1 = [[1, 2], [3, 4]]
#     data2 = [[5, 6], [7, 8]]
#     matrix1 = Matrix(data1)
#     matrix2 = Matrix(data2)
#     np_array1 = np.array(data1)
#     np_array2 = np.array(data2)
#     assert np.allclose(
#         (matrix1 @ matrix2).to_numpy(), np_array1 @ np_array2
#     ), "Matrix multiplication does not match NumPy."


# if __name__ == "__main__":
#     pytest.main()
