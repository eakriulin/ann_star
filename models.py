import time
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from shared import relu, softmax, activation_fn

Layer = tuple[np.ndarray, np.ndarray]

class Model(ABC):
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, Any]:
        pass

    @abstractmethod
    def backward(self, inputs: np.ndarray, cache: Any, loss_derivative: np.ndarray) -> Any:
        pass

    @abstractmethod
    def update(self, gradients: Any, learning_rate: float) -> None:
        pass

    @abstractmethod
    def parameters(self) -> list:
        pass

    @abstractmethod
    def copy(self) -> 'Model':
        pass

    def plot(self, dataset: str) -> None:
        pass

    model_type: str


class MLP(Model):
    model_type = 'mlp'

    def __init__(self, layers: list[Layer], last_activation: str):
        self.layers = layers
        self.last_activation = last_activation
        self.last_activation_fn = activation_fn(last_activation)

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, tuple[list[np.ndarray], list[np.ndarray]]]:
        n_layers = len(self.layers)
        last_layer_idx = n_layers - 1

        Zs = [None] * n_layers # note: layer input x layer weights + bias
        As = [None] * n_layers # note: layer activations

        for l in range(0, n_layers):
            W, b = self.layers[l]
            layer_input = inputs if l == 0 else As[l - 1]

            Zs[l] = np.matmul(layer_input, W) + np.squeeze(b)
            As[l] = self.last_activation_fn(Zs[l]) if l == last_layer_idx else relu(Zs[l])

        return As[-1], (As, Zs)

    def backward(self, inputs: np.ndarray, cache: tuple[list[np.ndarray], list[np.ndarray]], loss_derivative: np.ndarray) -> list[Layer]:
        gradients, _ = self._backward(inputs, cache, loss_derivative)
        return gradients

    def _backward(self, inputs: np.ndarray, cache: tuple[list[np.ndarray], list[np.ndarray]], loss_derivative: np.ndarray) -> tuple[list[Layer], np.ndarray]:
        As, Zs = cache
        n_layers = len(self.layers)
        last_layer_idx = n_layers - 1

        gradients = [None] * n_layers

        for l in range(last_layer_idx, -1, -1):
            if l == last_layer_idx:
                delta = loss_derivative * self.last_activation_fn(Zs[l], as_derivative=True)
            else:
                W_of_next_layer, _ = self.layers[l + 1]
                delta = np.matmul(delta, W_of_next_layer.T) * relu(Zs[l], as_derivative=True)

            layer_input = inputs if l == 0 else As[l - 1]
            W_gradient = np.matmul(layer_input.T, delta)
            b_gradient = np.sum(delta, keepdims=True, axis=0)

            gradients[l] = (W_gradient, b_gradient)

        W_of_first_layer, _ = self.layers[0]
        inputs_gradient = np.matmul(delta, W_of_first_layer.T)

        return gradients, inputs_gradient

    def update(self, gradients: list[Layer], learning_rate: float) -> None:
        for (W, b), (W_grad, b_grad) in zip(self.layers, gradients):
            W -= learning_rate * W_grad
            b -= learning_rate * b_grad.T

    def parameters(self) -> list[list[np.ndarray]]:
        return [[W, b] for W, b in self.layers]

    def copy(self) -> 'MLP':
        return MLP([(W.copy(), b.copy()) for W, b in self.layers], self.last_activation)


class LearnedActivationMLP(Model):
    model_type = 'lamlp'

    def __init__(self, layers: list[Layer], activations: list[MLP], activation_learning_rate_scale: float = 0.1):
        self.layers = layers
        self.activations = activations # note: one 1 -> ... -> 1 MLP per layer, shared by every neuron of that layer
        self.activation_learning_rate_scale = activation_learning_rate_scale

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, tuple[list[np.ndarray], list[np.ndarray], list[Any]]]:
        n_layers = len(self.layers)

        Zs = [None] * n_layers
        As = [None] * n_layers
        activation_caches = [None] * n_layers

        for l in range(0, n_layers):
            W, b = self.layers[l]
            layer_input = inputs if l == 0 else As[l - 1]

            Zs[l] = np.matmul(layer_input, W) + np.squeeze(b)

            activation_output, activation_caches[l] = self.activations[l].forward(Zs[l].reshape(-1, 1))
            As[l] = activation_output.reshape(Zs[l].shape)

        return As[-1], (As, Zs, activation_caches)

    def backward(self, inputs: np.ndarray, cache: tuple[list[np.ndarray], list[np.ndarray], list[Any]], loss_derivative: np.ndarray) -> tuple[list[Layer], list[list[Layer]]]:
        As, Zs, activation_caches = cache
        n_layers = len(self.layers)
        last_layer_idx = n_layers - 1

        gradients = [None] * n_layers
        activation_gradients = [None] * n_layers

        for l in range(last_layer_idx, -1, -1):
            if l == last_layer_idx:
                activation_output_gradient = loss_derivative
            else:
                W_of_next_layer, _ = self.layers[l + 1]
                activation_output_gradient = np.matmul(delta, W_of_next_layer.T)

            activation_gradients[l], activation_input_gradient = self.activations[l]._backward(Zs[l].reshape(-1, 1), activation_caches[l], activation_output_gradient.reshape(-1, 1))
            delta = activation_input_gradient.reshape(Zs[l].shape)

            layer_input = inputs if l == 0 else As[l - 1]
            W_gradient = np.matmul(layer_input.T, delta)
            b_gradient = np.sum(delta, keepdims=True, axis=0)

            gradients[l] = (W_gradient, b_gradient)

        return gradients, activation_gradients

    def update(self, gradients: tuple[list[Layer], list[list[Layer]]], learning_rate: float) -> None:
        layer_gradients, activation_gradients = gradients

        for (W, b), (W_grad, b_grad) in zip(self.layers, layer_gradients):
            W -= learning_rate * W_grad
            b -= learning_rate * b_grad.T

        for activation, activation_gradient in zip(self.activations, activation_gradients):
            activation.update(activation_gradient, learning_rate * self.activation_learning_rate_scale)

    def parameters(self) -> list:
        parameters = []

        for (W, b), activation in zip(self.layers, self.activations):
            parameters.append([W, b])
            parameters.append(activation.parameters())

        return parameters

    def copy(self) -> 'LearnedActivationMLP':
        return LearnedActivationMLP(
            [(W.copy(), b.copy()) for W, b in self.layers],
            [activation.copy() for activation in self.activations],
            self.activation_learning_rate_scale,
        )

    def plot(self, dataset: str) -> None:
        xs = np.linspace(-3, 3, 300)

        plt.figure(figsize=(10, 6))
        for l, activation in enumerate(self.activations):
            ys, _ = activation.forward(xs.reshape(-1, 1))
            plt.plot(xs, ys.reshape(xs.shape), label=f'layer {l + 1} ({activation.last_activation} output)')
        plt.plot(xs, relu(xs), 'k--', label='ReLU')
        plt.title('Learned activations')
        plt.xlabel('x')
        plt.ylabel('g(x)')
        plt.grid(True)
        plt.legend()

        plt.savefig(f'./learned_activations_{dataset}_{time.time()}.png')


class MoE(Model):
    model_type = 'moe'

    def __init__(self, gater: Layer, experts: list[MLP]):
        self.gater = gater
        self.experts = experts

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, list[Any]]]:
        W_gater, b_gater = self.gater
        gates = softmax(np.matmul(inputs, W_gater) + np.squeeze(b_gater))

        experts_predictions = []
        experts_caches = []
        for expert in self.experts:
            expert_predictions, expert_cache = expert.forward(inputs)
            experts_predictions.append(expert_predictions)
            experts_caches.append(expert_cache)

        predictions = np.zeros_like(experts_predictions[0])
        for expert_idx, expert_predictions in enumerate(experts_predictions):
            predictions += gates[:, expert_idx:expert_idx + 1] * expert_predictions

        return predictions, (gates, experts_caches)

    def backward(self, inputs: np.ndarray, cache: tuple[np.ndarray, list[Any]], loss_derivative: np.ndarray) -> tuple[Layer, list[list[Layer]]]:
        gates, experts_caches = cache

        experts_gradients = []
        gates_gradient = np.zeros_like(gates)

        for expert_idx, expert in enumerate(self.experts):
            expert_gate = gates[:, expert_idx:expert_idx + 1]
            expert_loss_derivative = expert_gate * loss_derivative
            experts_gradients.append(expert.backward(inputs, experts_caches[expert_idx], expert_loss_derivative))

            expert_predictions = experts_caches[expert_idx][0][-1]
            gates_gradient[:, expert_idx] = np.sum(loss_derivative * expert_predictions, axis=1)

        weighted_average_gates_gradient = np.sum(gates * gates_gradient, axis=1, keepdims=True)
        gater_delta = gates * (gates_gradient - weighted_average_gates_gradient)

        gater_W_gradient = np.matmul(inputs.T, gater_delta)
        gater_b_gradient = np.sum(gater_delta, keepdims=True, axis=0)

        return (gater_W_gradient, gater_b_gradient), experts_gradients

    def update(self, gradients: tuple[Layer, list[list[Layer]]], learning_rate: float) -> None:
        (gater_W_gradient, gater_b_gradient), experts_gradients = gradients

        W_gater, b_gater = self.gater
        W_gater -= learning_rate * gater_W_gradient
        b_gater -= learning_rate * gater_b_gradient.T

        for expert, expert_gradients in zip(self.experts, experts_gradients):
            expert.update(expert_gradients, learning_rate)

    def parameters(self) -> list[list[list[np.ndarray]]]:
        return [[list(self.gater)], *[expert.parameters() for expert in self.experts]]

    def copy(self) -> 'MoE':
        W_gater, b_gater = self.gater
        return MoE((W_gater.copy(), b_gater.copy()), [expert.copy() for expert in self.experts])
