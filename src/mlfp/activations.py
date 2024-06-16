from __future__ import annotations

import math

from mlfp.linear_algebra import Matrix


class ActivationFunction:
    def __init__(self, name: str) -> None:
        self.name: str = name

    def __call__(self, x: Matrix) -> Matrix:
        return self.forward(x)

    def forward(self, x: Matrix) -> Matrix:
        raise NotImplementedError

    def backward(self, x: Matrix) -> Matrix:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name


class Sigmoid(ActivationFunction):
    def __init__(self) -> None:
        super().__init__("Sigmoid")

    def forward(self, x: Matrix) -> Matrix:
        # Apply the sigmoid function element-wise to the matrix x
        return x.apply(lambda x: 1 / (1 + math.exp(-x)))

    # def backward(self, x: Matrix) -> Matrix:
    #     return self.forward(x) * (1 - self.forward(x))
