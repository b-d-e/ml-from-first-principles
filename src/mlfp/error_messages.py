# feels wrong to implement these as funcs, but also better to have standardisation?
from __future__ import annotations


def vector_length_mismatch() -> str:
    return "Vectors must be of the same length"


def vector_unsupported_operand(other: str) -> str:
    return f"Unsupported operand type(s) for *: 'Vector' and '{type(other).__name__}"
