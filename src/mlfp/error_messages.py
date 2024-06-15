# feels wrong to implement these as funcs, but also better to have standardisation?
from __future__ import annotations


def vector_length_mismatch() -> str:
    return "Vectors must be of the same length"


def vector_unsupported_operand(other: str) -> str:
    return f"Unsupported operand type(s) for *: 'Vector' and '{type(other).__name__}"


def matrix_shape_mismatch(shape_a: tuple, shape_b: tuple) -> str:
    return f"Matrices must be of the same shape. Got {shape_a} and {shape_b}"


def matrix_unsupported_operand(other_type: str) -> str:
    return f"Unsupported operand type(s) for *: 'Matrix' and '{other_type}'"


def matrix_incompatible_shapes(self_shape: tuple, other_shape: tuple) -> str:
    err = f"Matrices must have compatible shapes - \n Matrix A is {self_shape}, Matrix B is {other_shape}\n{self_shape[1]} != {other_shape[0]}"
    if self_shape[0] == other_shape[1]:
        err += "\nHint: It looks like you might want to product these the other way around?"
    return
