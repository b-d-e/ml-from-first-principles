# from __future__ import annotations

# import math

# from mlfp.linear_algebra import Matrix


# class ActivationFunction:
#     def __init__(self, name: str) -> None:
#         self.name: str = name

#     def __call__(self, x: Matrix) -> Matrix:
#         return self.forward(x)

#     def forward(self, x: Matrix) -> Matrix:
#         raise NotImplementedError

#     def backward(self, x: Matrix) -> Matrix:
#         raise NotImplementedError

#     def __repr__(self) -> str:
#         return self.name


# class Linear(ActivationFunction):
#     def __init__(self) -> None:
#         super().__init__("Linear")

#     def forward(self, x: Matrix) -> Matrix:
#         return x

#     def backward(self, x: Matrix) -> Matrix:
#         return Matrix([[1 for _ in range(x.shape[1])]])


# class Sigmoid(ActivationFunction):
#     def __init__(self) -> None:
#         super().__init__("Sigmoid")

#     def forward(self, x: Matrix) -> Matrix:
#         # Apply the sigmoid function element-wise to the matrix x
#         return x.apply(lambda x: 1 / (1 + math.exp(-x)))

#     def backward(self, x: Matrix) -> Matrix:
#         # Apply the derivative of the sigmoid function element-wise to the matrix x
#         return x.apply(lambda x: x * (1 - x))


# class ReLU(ActivationFunction):
#     def __init__(self) -> None:
#         super().__init__("ReLU")

#     def forward(self, x: Matrix) -> Matrix:
#         return x.apply(lambda x: max(0, x))

#     def backward(self, x: Matrix) -> Matrix:
#         return x.apply(lambda x: 1 if x > 0 else 0)


# class Tanh(ActivationFunction):
#     def __init__(self) -> None:
#         super().__init__("Tanh")

#     def forward(self, x: Matrix) -> Matrix:
#         return x.apply(lambda x: math.tanh(x))

#     def backward(self, x: Matrix) -> Matrix:
#         return x.apply(lambda x: 1 - x**2)


# class Softmax(ActivationFunction):
#     def __init__(self) -> None:
#         super().__init__("Softmax")

#     def forward(self, x: Matrix) -> Matrix:
#         # Apply the softmax function element-wise to the matrix x
#         exp_x = x.apply(math.exp)
#         sum_exp_x = sum(exp_x.elements[0])
#         return exp_x * (1 / sum_exp_x)

#     def backward(self, x: Matrix) -> Matrix:
#         # Apply the derivative of the softmax function element-wise to the matrix x
#         return x.apply(lambda x: x * (1 - x))
