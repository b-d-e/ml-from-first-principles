# a file to implement linear algebra operations without numpy

# vector implementation

from __future__ import annotations

from mlfp import error_messages


class Vector:
    def __init__(self, elements: list[float]) -> None:
        self.elements = elements

    def __repr__(self) -> str:
        return f"Vector({self.elements})"

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

    def dot(self, other: Vector) -> float:
        if len(self.elements) != len(other.elements):
            raise ValueError(error_messages.vector_length_mismatch())
        return sum(a * b for a, b in zip(self.elements, other.elements, strict=False))

    # def magnitude(self) -> float:
    #     mag: float = sum(x**2 for x in self.elements) ** 0.5
    #     return mag
