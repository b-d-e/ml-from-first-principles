# This file implements a versatile n-d Tensor data class,
# without relying on Numpy or other Python libraries

from __future__ import annotations  # allow methods of Tensor class to return Tensor

import random as randlib
from collections.abc import Callable, Sequence
from typing import Any  # , Union

import numpy as np  # _only_ used for to_numpy method

from mlfp import error_messages as em

# TensorElementsType = Union[
#     Sequence["TensorElementsType"], float
# ]  # Allow arbitrary depth
TensorElementsType = (
    Sequence["TensorElementsType"] | float | None
    # n.b. not sure if the None will break things here
)  # Allow arbitrary depth
TensorShapeType = tuple[int, ...]


class Tensor:
    def __init__(self, elements: TensorElementsType) -> None:
        self.elements = elements
        self.shape = self.__get_shape(self.elements)

    def __repr__(self) -> str:
        return f"Tensor(elements={self.elements})"

    def __get_shape(self, ndarray: TensorElementsType | float) -> TensorShapeType:
        if isinstance(ndarray, list):
            outermost_size = len(ndarray)
            if outermost_size == 0:
                return (0,)
            row_shape = self.__get_shape(ndarray[0])
            for element in ndarray:
                if self.__get_shape(element) != row_shape:
                    raise em.InconsistentShapeError()
            return (outermost_size, *row_shape)
        return ()

    def __eq__(self, other: object) -> bool:
        if type(other) is not Tensor:
            return False
        # do we need to check shape?
        # check element match
        return not self.elements != other.elements

    def flatten(self) -> list[float]:
        # Implement this method to flatten the tensor
        flat_list = []

        def __flatten(lst: TensorElementsType) -> None:
            if isinstance(lst, Sequence) and not isinstance(lst, str | bytes):
                for item in lst:
                    __flatten(item)
            # Ensure only floats and integers are added
            elif isinstance(lst, float | int):
                flat_list.append(float(lst))  # Convert integers to floats
            else:
                raise em.TensorElementError()

        __flatten(self.elements)
        return flat_list

    def _reshape_recursive(
        self, flat: list[float], shape: tuple[int, ...]
    ) -> TensorElementsType:
        if len(shape) == 1:
            return flat[: shape[0]]
        size = shape[0]
        sub_size = int(len(flat) / size)
        return [
            self._reshape_recursive(flat[i * sub_size : (i + 1) * sub_size], shape[1:])
            for i in range(size)
        ]

    def reshape(self, shape: tuple[int, ...]) -> Tensor:
        # return a new tensor with the same elements, but a different shape
        # flatten and restructure - is there a better way?
        # flatten
        flat = self.flatten()
        # check new shape is valid
        shape_len = 1
        for dim in shape:
            shape_len *= dim
        if shape_len != len(flat):
            raise em.InvalidReshapeShapes(self.shape, len(shape), shape_len)
        # restructure
        return Tensor(self._reshape_recursive(flat, shape))

    def __add__(self, other: Tensor) -> Tensor:
        # slightly janky - flatten, add, reshape
        # check same shape
        if self.shape != other.shape:
            raise em.ShapeMismatchError(self.shape, other.shape)
        # element-wise addition
        # flatten
        flat_1 = self.flatten()
        flat_2 = other.flatten()
        flat_sum = [a + b for a, b in zip(flat_1, flat_2, strict=False)]
        # reshape
        return Tensor(self._reshape_recursive(flat_sum, self.shape))

    def __sub__(self, other: Tensor) -> Tensor:
        return self + other.__mul__(-1)

    def __mul__(self, scalar: float | Tensor) -> Tensor:
        if isinstance(scalar, Tensor):
            if self.shape != scalar.shape:
                raise em.ShapeMismatchError(self.shape, scalar.shape)
            flat_1 = self.flatten()
            flat_2 = scalar.flatten()
            flat_prod = [a * b for a, b in zip(flat_1, flat_2, strict=False)]
            return Tensor(self._reshape_recursive(flat_prod, self.shape))
        # otherwise, it's a float (removed 'else' to keep ruff happy)
        flat = self.flatten()
        flat_prod = [a * scalar for a in flat]
        return Tensor(self._reshape_recursive(flat_prod, self.shape))

    def __rmul__(self, scalar: float) -> Tensor:
        return self.__mul__(scalar)  # scalar multiplication is commutative

    def __matmul__(self, other: Tensor) -> Tensor:
        if (
            len(self.shape) > 2
            or len(other.shape) > 2
            or len(self.shape) == 0
            or len(other.shape) == 0
        ):
            raise em.MatMulMaxSizeError(self.shape, other.shape)
        if self.shape[1] != other.shape[0]:
            raise em.MatMulIncompatibleShapesError(self.shape, other.shape)

        # Ensure elements are lists of lists
        if (
            not isinstance(self.elements, list)
            or not all(isinstance(row, list) for row in self.elements)
            or not isinstance(other.elements, list)
            or not all(isinstance(row, list) for row in other.elements)
        ):
            raise em.MatMulTypeError(type(self.elements), type(other.elements))

        # Perform matrix multiplication
        result = [
            [
                sum(
                    self.elements[i][k] * other.elements[k][j]
                    for k in range(self.shape[1])
                )
                for j in range(other.shape[1])
            ]
            for i in range(self.shape[0])
        ]

        return Tensor(result)

    def __pow__(self, power: float) -> Tensor:
        def _power(x: float) -> float:
            return float(x**power)

        return self.apply(_power)

    def sum(self) -> float:
        return sum(self.flatten())

    def apply(self, func: Callable[[float], float]) -> Tensor:
        def _apply(lst: TensorElementsType) -> TensorElementsType:
            if isinstance(lst, list):
                return [_apply(item) for item in lst]
            if isinstance(lst, float | int):
                return func(float(lst))
            raise em.TensorElementError()

        return Tensor(_apply(self.elements))

    def transpose(self) -> Tensor:
        # must handle non-square matrices
        raise NotImplementedError
        # return Tensor([[row[i] for row in self.elements] \
        # for i in range(self.shape[1])])

    def to_numpy(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return np.array(self.elements)


# Function which mirror numpy usage...


def zeros(shape: TensorShapeType) -> Tensor:
    """Return a new array of given shape, filled with zeros."""

    def _zeros(shape: TensorShapeType) -> TensorElementsType:
        if len(shape) == 1:
            return [0.0] * shape[0]
        return [_zeros(shape[1:]) for _ in range(shape[0])]

    return Tensor(_zeros(shape))


def ones(shape: TensorShapeType) -> Tensor:
    """Return a new array of given shape, filled with ones."""

    def _ones(shape: TensorShapeType) -> TensorElementsType:
        if len(shape) == 1:
            return [1.0] * shape[0]
        return [_ones(shape[1:]) for _ in range(shape[0])]

    return Tensor(_ones(shape))


def empty(shape: TensorShapeType) -> Tensor:
    """Return a new array of given shape, without initializing entries."""

    def _empty(shape: TensorShapeType) -> TensorElementsType:
        if len(shape) == 1:
            return [None] * shape[0]
        return [_empty(shape[1:]) for _ in range(shape[0])]

    return Tensor(_empty(shape))


def full(shape: TensorShapeType, fill_value: float) -> Tensor:
    """Return a new array of given shape, filled with fill_value."""

    def _full(shape: TensorShapeType) -> TensorElementsType:
        if len(shape) == 1:
            return [fill_value] * shape[0]
        return [_full(shape[1:]) for _ in range(shape[0])]

    return Tensor(_full(shape))


def eye(N: int) -> Tensor:
    """Return a 2-D array with ones on the diagonal and zeros elsewhere."""

    def _eye(N: int) -> TensorElementsType:
        return [[1.0 if i == j else 0.0 for j in range(N)] for i in range(N)]

    return Tensor(_eye(N))


def random(shape: TensorShapeType) -> Tensor:
    """Return a new array of given shape, filled with random values."""

    def _random(shape: TensorShapeType) -> TensorElementsType:
        if len(shape) == 1:
            return [randlib.random() for _ in range(shape[0])]
        return [_random(shape[1:]) for _ in range(shape[0])]

    return Tensor(_random(shape))
