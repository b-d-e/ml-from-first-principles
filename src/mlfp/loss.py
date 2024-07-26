# # losses for training the model, no libraries
# from __future__ import annotations

# import math

# import mlfp.linear_algebra as la


# class LossFunction:
#     def __init__(self, name: str) -> None:
#         self.name: str = name

#     def __call__(self, y_true: la.Tensor, y_pred: la.Tensor) -> float:
#         return self.calculate(y_true, y_pred)

#     def calculate(self, y_true: la.Tensor, y_pred: la.Tensor) -> float:
#         raise NotImplementedError

#     def gradient(self, y_true: la.Tensor, y_pred: la.Tensor) -> la.Tensor:
#         raise NotImplementedError

#     def __repr__(self) -> str:
#         return self.name


# class MeanSquaredError(LossFunction):
#     def __init__(self) -> None:
#         super().__init__("MeanSquaredError")

#     def calculate(self, y_true: la.Tensor, y_pred: la.Tensor) -> float:
#         return 0.5 * ((y_true - y_pred) ** 2).sum()

#     def gradient(self, y_true: la.Tensor, y_pred: la.Tensor) -> la.Tensor:
#         return y_pred - y_true


# class CrossEntropy(LossFunction):
#     def __init__(self) -> None:
#         super().__init__("CrossEntropy")

#     def calculate(self, y_true: la.Tensor, y_pred: la.Tensor) -> float:
#         return -1 * (y_true * y_pred.apply(math.log)).sum()

#     def gradient(self, y_true: la.Tensor, y_pred: la.Tensor) -> la.Tensor:
#         return y_pred - y_true
