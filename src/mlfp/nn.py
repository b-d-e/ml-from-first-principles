from __future__ import annotations

import mlfp.activations as activations

# import mlfp.error_messages as error_messages
import mlfp.linear_algebra as la

# implement basic densely connected neural network,
# with a weights and biases representation


class Layer:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation_function: activations.ActivationFunction,
    ) -> None:
        self.weights = la.random((input_size, output_size))
        self.biases = la.zeros((1, output_size))
        self.activation_function = activation_function

    def __repr__(self) -> str:
        return f"Layer(weights={self.weights}, biases={self.biases}\
            , activation_function={self.activation_function})"

    def forward(self, inputs: la.Tensor) -> la.Tensor:
        z = inputs @ self.weights + self.biases
        return self.activation_function(z)


class Network:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def __repr__(self) -> str:
        return f"NeuralNetwork(layers={self.layers})"

    def forward(self, inputs: la.Tensor) -> la.Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
