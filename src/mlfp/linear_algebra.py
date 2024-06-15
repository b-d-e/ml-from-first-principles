# a file to implement linear algebra operations without numpy

# vector implementation

from __future__ import annotations

import numpy as np  # for conversion purposes only

from mlfp import error_messages


class Vector:
    # TODO: vectors can be reimplemented as a subclass of a Matrix, OR removed entirely?
    def __init__(self, elements: list[float]) -> None:
        self.elements = elements

    def __repr__(self) -> str:
        return f"Vector({self.elements})"

    def __len__(self) -> int:
        return len(self.elements)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return self.elements == other.elements

    def __elements__(self) -> list[float]:
        return self.elements

    def __add__(self, other: Vector) -> Vector:
        if len(self.elements) != len(other.elements):
            raise ValueError(error_messages.vector_length_mismatch())
        return Vector(
            [a + b for a, b in zip(self.elements, other.elements, strict=False)]
        )

    def __sub__(self, other: Vector) -> Vector:
        if len(self.elements) != len(other.elements):
            raise ValueError(error_messages.vector_length_mismatch())
        return Vector(
            [a - b for a, b in zip(self.elements, other.elements, strict=False)]
        )

    def __mul__(self, scalar: float) -> Vector:
        if isinstance(scalar, int | float):
            return Vector([a * scalar for a in self.elements])
        raise TypeError(error_messages.vector_unsupported_operand(scalar))

    def __rmul__(self, scalar: float) -> Vector:
        return self.__mul__(scalar)

    def __matmul__(self, other: Vector) -> float:
        if len(self.elements) != len(other.elements):
            raise ValueError(error_messages.vector_length_mismatch())
        return sum(a * b for a, b in zip(self.elements, other.elements, strict=False))

    def dot(self, other: Vector) -> float:
        return self.__matmul__(other)

    # def magnitude(self) -> float:
    #     mag: float = sum(x**2 for x in self.elements) ** 0.5
    #     return mag

    def to_numpy(self) -> np.ndarray:
        return np.array(self.elements)


class Matrix:
    def __init__(self, elements: list[list[float]]) -> None:
        if not elements:  # handle empty
            self.elements = []
            self.shape = (0, 0)
            return
        self.elements = elements
        self.shape = (len(elements), len(elements[0]))

    def __repr__(self) -> str:
        # pretty print
        # get num chars in longest element, and pad items
        max_len = max(len(str(item)) for row in self.elements for item in row)
        m = ""
        for row in self.elements:
            m += "  ".join(str(item).ljust(max_len) for item in row) + "\n"

        return f"Matrix:\n{m}"

    def __len__(self) -> int:
        return self.shape[0] * self.shape[1]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return False
        return self.elements == other.elements

    def __elements__(self) -> list[list[float]]:
        return self.elements

    def __add__(self, other: Matrix) -> Matrix:
        if self.shape != other.shape:
            raise ValueError(
                error_messages.matrix_shape_mismatch(self.shape, other.shape)
            )
        return Matrix(
            [
                [a + b for a, b in zip(row1, row2, strict=False)]
                for row1, row2 in zip(self.elements, other.elements, strict=False)
            ]
        )

    def __sub__(self, other: Matrix) -> Matrix:
        if self.shape != other.shape:
            raise ValueError(
                error_messages.matrix_shape_mismatch(self.shape, other.shape)
            )
        return Matrix(
            [
                [a - b for a, b in zip(row1, row2, strict=False)]
                for row1, row2 in zip(self.elements, other.elements, strict=False)
            ]
        )

    def __mul__(self, scalar: float) -> Matrix:
        if isinstance(scalar, int | float):
            return Matrix([[a * scalar for a in row] for row in self.elements])
        raise TypeError(
            error_messages.matrix_unsupported_operand(type(scalar).__name__)
        )  # we actually should never hit this, given strict type hints

    def __rmul__(self, scalar: float) -> Matrix:
        return self.__mul__(scalar)

    def __matmul__(self, other: Matrix) -> Matrix:
        if self.shape[1] != other.shape[0]:
            raise ValueError(
                error_messages.matrix_incompatible_shapes(self.shape, other.shape)
            )
        return Matrix(
            [
                [
                    sum(a * b for a, b in zip(row1, col2, strict=False))
                    for col2 in zip(*other.elements, strict=False)
                ]
                for row1 in self.elements
            ]
        )

    def dot(self, other: Matrix) -> Matrix:
        return self.__matmul__(other)

    def transpose(self) -> Matrix:
        # must handle non-square matrices
        return Matrix([[row[i] for row in self.elements] for i in range(self.shape[1])])

    def to_numpy(self) -> np.ndarray:
        return np.array(self.elements)


def ones(shape: tuple[int, int]) -> Matrix:
    return Matrix([[1 for _ in range(shape[1])] for _ in range(shape[0])])


def zeros(shape: tuple[int, int]) -> Matrix:  # feels bad setting shape here...
    return Matrix([[0 for _ in range(shape[1])] for _ in range(shape[0])])
