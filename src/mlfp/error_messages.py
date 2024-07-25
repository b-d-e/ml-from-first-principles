# feels wrong to implement these as funcs, but also better to have standardisation?
from __future__ import annotations


class InconsistentShapeError(Exception):
    def __init__(self) -> None:
        self.message = "Inconsistent shaped layer in Tensor."
        super().__init__(self.message)


class InvalidReshapeShapes(Exception):
    def __init__(self, shape: tuple[int, ...], shape_len: int, flat_len: int) -> None:
        self.message = f"Invalid shape {shape}. \
            Expected a shape of length {shape_len} with a total of {flat_len} elements."
        super().__init__(self.message)


class ShapeMismatchError(Exception):
    def __init__(self, shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> None:
        self.message = f"Shapes {shape_a} and {shape_b} do not match."
        super().__init__(self.message)


class MatMulMaxSizeError(Exception):
    def __init__(self, shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> None:
        self.message = f"Tensor multiplication only supported for 2D tensors. \
            Got shapes {shape_a} and {shape_b}."
        super().__init__(self.message)


class MatMulTypeError(Exception):
    def __init__(self, type_a: type, type_b: type) -> None:
        self.message = f"Tensor multiplication only supported for tensors. \
            Got types {type_a} and {type_b}."
        super().__init__(self.message)


class MatMulIncompatibleShapesError(Exception):
    def __init__(self, shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> None:
        self.message = f"Incompatible tensor shapes for multiplication. \
            Got shapes {shape_a} and {shape_b} where {shape_a[1]} != {shape_b[0]}."
        super().__init__(self.message)


class TensorElementError(Exception):
    def __init__(
        self,
    ) -> None:
        self.message = "All elements must be either float, int, \
                        or sequences of these types"
        super().__init__(self.message)
