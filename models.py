import time
from abc import ABC, abstractmethod
from typing import Any, NamedTuple
import numpy as np
import matplotlib.pyplot as plt
from shared import relu, softmax, activation_fn_by_name

Layer = tuple[np.ndarray, np.ndarray]

MAX_PLOTTED_ACTIVATIONS = 7

class MLPCache(NamedTuple):
    Z_per_layer: list[np.ndarray]
    A_per_layer: list[np.ndarray]

class MoECache(NamedTuple):
    A_gater: np.ndarray
    experts_caches: list[MLPCache]

class TargetPropagationMLPCache(NamedTuple):
    Z_hidden: np.ndarray
    A_hidden: np.ndarray
    neuron_activation_caches: list[MLPCache]
    Z_output: np.ndarray

class NeuronActivationMLPCache(NamedTuple):
    Z_per_layer: list[np.ndarray]
    A_per_layer: list[np.ndarray]
    Z_inner_per_layer: list[np.ndarray]

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

    def plot(self, dataset: str, algo: str) -> None:
        pass

    model_type: str


class MLP(Model):
    model_type = 'mlp'

    def __init__(self, layers: list[Layer], last_activation: str):
        self.layers = layers
        self.last_activation = last_activation
        self.last_activation_fn = activation_fn_by_name(last_activation)

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, MLPCache]:
        n_layers = len(self.layers)
        last_layer_idx = n_layers - 1

        Z_per_layer = [None] * n_layers # note: layer input x layer weights + bias
        A_per_layer = [None] * n_layers # note: layer activations

        for l in range(0, n_layers):
            W, b = self.layers[l]
            layer_input = inputs if l == 0 else A_per_layer[l - 1]

            Z_per_layer[l] = np.matmul(layer_input, W) + np.squeeze(b)
            A_per_layer[l] = self.last_activation_fn(Z_per_layer[l]) if l == last_layer_idx else relu(Z_per_layer[l])

        return A_per_layer[-1], MLPCache(Z_per_layer, A_per_layer)

    def backward(self, inputs: np.ndarray, cache: MLPCache, loss_derivative: np.ndarray) -> list[Layer]:
        layer_gradients, _ = self.backward_with_inputs_gradient(inputs, cache, loss_derivative)
        return layer_gradients

    def backward_with_inputs_gradient(self, inputs: np.ndarray, cache: MLPCache, loss_derivative: np.ndarray) -> tuple[list[Layer], np.ndarray]:
        Z_per_layer, A_per_layer = cache
        n_layers = len(self.layers)
        last_layer_idx = n_layers - 1

        layer_gradients = [None] * n_layers

        for l in range(last_layer_idx, -1, -1):
            if l == last_layer_idx:
                Z_gradient = loss_derivative * self.last_activation_fn(Z_per_layer[l], as_derivative=True)
            else:
                W_next_layer, _ = self.layers[l + 1]
                Z_gradient = np.matmul(Z_gradient, W_next_layer.T) * relu(Z_per_layer[l], as_derivative=True)

            layer_input = inputs if l == 0 else A_per_layer[l - 1]
            W_gradient = np.matmul(layer_input.T, Z_gradient)
            b_gradient = np.sum(Z_gradient, keepdims=True, axis=0)

            layer_gradients[l] = (W_gradient, b_gradient)

        W_first_layer, _ = self.layers[0]
        inputs_gradient = np.matmul(Z_gradient, W_first_layer.T)

        return layer_gradients, inputs_gradient

    def update(self, gradients: list[Layer], learning_rate: float) -> None:
        update_layers(self.layers, gradients, learning_rate)

    def parameters(self) -> list[list[np.ndarray]]:
        return [[W, b] for W, b in self.layers]

    def copy(self) -> 'MLP':
        return MLP([(W.copy(), b.copy()) for W, b in self.layers], self.last_activation)


class NeuronActivations:
    def __init__(self, W_in: np.ndarray, b_in: np.ndarray, W_out: np.ndarray, b_out: np.ndarray):
        self.W_in = W_in
        self.b_in = b_in
        self.W_out = W_out
        self.b_out = b_out

    def forward(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Z_inner = Z[:, :, np.newaxis] * self.W_in + self.b_in
        A = np.sum(relu(Z_inner) * self.W_out, axis=2) + self.b_out[:, 0]

        return A, Z_inner

    def inputs_gradient(self, Z_inner: np.ndarray, A_gradient: np.ndarray) -> np.ndarray:
        Z_inner_gradient = A_gradient[:, :, np.newaxis] * self.W_out * relu(Z_inner, as_derivative=True)
        return np.sum(Z_inner_gradient * self.W_in, axis=2)

    def neuron(self, u: int) -> 'NeuronActivations':
        return NeuronActivations(*[parameter[u:u + 1].copy() for parameter in self.parameters()])

    @staticmethod
    def stack(neurons: list['NeuronActivations']) -> 'NeuronActivations':
        return NeuronActivations(*[np.concatenate(parameter_per_neuron) for parameter_per_neuron in zip(*[neuron.parameters() for neuron in neurons])])

    def n_neurons(self) -> int:
        return self.W_in.shape[0]

    def parameters(self) -> list[np.ndarray]:
        return [self.W_in, self.b_in, self.W_out, self.b_out]

    def copy(self) -> 'NeuronActivations':
        return NeuronActivations(*[parameter.copy() for parameter in self.parameters()])


class NeuronActivationMLP(Model):
    model_type = 'namlp'

    def __init__(self, layers: list[Layer], neuron_activations_per_layer: list[NeuronActivations], last_activation: str):
        self.layers = layers
        self.neuron_activations_per_layer = neuron_activations_per_layer
        self.last_activation = last_activation
        self.last_activation_fn = activation_fn_by_name(last_activation)

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, NeuronActivationMLPCache]:
        cache = self._forward_layers(first_layer_idx=0, layer_input=inputs)
        return cache.A_per_layer[-1], cache

    def forward_from_layer(self, first_layer_idx: int, layer_input: np.ndarray) -> np.ndarray:
        return self._forward_layers(first_layer_idx, layer_input).A_per_layer[-1]

    def _forward_layers(self, first_layer_idx: int, layer_input: np.ndarray) -> NeuronActivationMLPCache:
        last_layer_idx = len(self.layers) - 1

        Z_per_layer = []
        A_per_layer = []
        Z_inner_per_layer = []

        for l in range(first_layer_idx, len(self.layers)):
            W, b = self.layers[l]
            Z = np.matmul(layer_input, W) + np.squeeze(b, axis=1)

            if l == last_layer_idx:
                A = self.last_activation_fn(Z)
            else:
                A, Z_inner = self.neuron_activations_per_layer[l].forward(Z)
                Z_inner_per_layer.append(Z_inner)

            Z_per_layer.append(Z)
            A_per_layer.append(A)
            layer_input = A

        return NeuronActivationMLPCache(Z_per_layer, A_per_layer, Z_inner_per_layer)

    def backward(self, inputs: np.ndarray, cache: NeuronActivationMLPCache, loss_derivative: np.ndarray) -> list[Layer]:
        Z_per_layer, A_per_layer, Z_inner_per_layer = cache
        last_layer_idx = len(self.layers) - 1

        layer_gradients = [None] * len(self.layers)
        Z_gradient = loss_derivative * self.last_activation_fn(Z_per_layer[last_layer_idx], as_derivative=True)

        for l in range(last_layer_idx, -1, -1):
            if l < last_layer_idx:
                W_next_layer, _ = self.layers[l + 1]
                A_gradient = np.matmul(Z_gradient, W_next_layer.T)
                Z_gradient = self.neuron_activations_per_layer[l].inputs_gradient(Z_inner_per_layer[l], A_gradient)

            layer_input = inputs if l == 0 else A_per_layer[l - 1]
            layer_gradients[l] = (np.matmul(layer_input.T, Z_gradient), np.sum(Z_gradient, keepdims=True, axis=0))

        return layer_gradients

    def update(self, gradients: list[Layer], learning_rate: float) -> None:
        update_layers(self.layers, gradients, learning_rate)

    def parameters(self) -> list:
        layer_parameters = [[W, b] for W, b in self.layers]
        neuron_activation_parameters = [neuron_activations.parameters() for neuron_activations in self.neuron_activations_per_layer]

        return [layer_parameters, neuron_activation_parameters]

    def copy(self) -> 'NeuronActivationMLP':
        return NeuronActivationMLP(
            [(W.copy(), b.copy()) for W, b in self.layers],
            [neuron_activations.copy() for neuron_activations in self.neuron_activations_per_layer],
            self.last_activation,
        )

    def plot(self, dataset: str, algo: str) -> None:
        labeled_activations = [
            (f'hidden layer {l + 1}, neuron {u + 1}', neuron_activations.neuron(u))
            for l, neuron_activations in enumerate(self.neuron_activations_per_layer)
            for u in range(neuron_activations.n_neurons())
        ]
        plot_learned_activations(labeled_activations, algo, self.model_type, dataset)


class MoE(Model):
    model_type = 'moe'

    def __init__(self, gater_layer: Layer, experts: list[MLP]):
        self.gater_layer = gater_layer
        self.experts = experts

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, MoECache]:
        W_gater, b_gater = self.gater_layer
        A_gater = softmax(np.matmul(inputs, W_gater) + np.squeeze(b_gater))

        A_experts = []
        experts_caches = []
        for expert in self.experts:
            A_expert, expert_cache = expert.forward(inputs)
            A_experts.append(A_expert)
            experts_caches.append(expert_cache)

        A_output = np.zeros_like(A_experts[0])
        for expert_idx, A_expert in enumerate(A_experts):
            A_output += A_gater[:, expert_idx:expert_idx + 1] * A_expert

        return A_output, MoECache(A_gater, experts_caches)

    def backward(self, inputs: np.ndarray, cache: MoECache, loss_derivative: np.ndarray) -> tuple[Layer, list[list[Layer]]]:
        A_gater, experts_caches = cache

        experts_gradients = []
        A_gater_gradient = np.zeros_like(A_gater)

        for expert_idx, expert in enumerate(self.experts):
            A_gater_of_expert = A_gater[:, expert_idx:expert_idx + 1]
            expert_loss_derivative = A_gater_of_expert * loss_derivative
            experts_gradients.append(expert.backward(inputs, experts_caches[expert_idx], expert_loss_derivative))

            A_expert = experts_caches[expert_idx].A_per_layer[-1]
            A_gater_gradient[:, expert_idx] = np.sum(loss_derivative * A_expert, axis=1)

        weighted_average_A_gater_gradient = np.sum(A_gater * A_gater_gradient, axis=1, keepdims=True)
        Z_gater_gradient = A_gater * (A_gater_gradient - weighted_average_A_gater_gradient)

        W_gater_gradient = np.matmul(inputs.T, Z_gater_gradient)
        b_gater_gradient = np.sum(Z_gater_gradient, keepdims=True, axis=0)

        return (W_gater_gradient, b_gater_gradient), experts_gradients

    def update(self, gradients: tuple[Layer, list[list[Layer]]], learning_rate: float) -> None:
        (W_gater_gradient, b_gater_gradient), experts_gradients = gradients

        W_gater, b_gater = self.gater_layer
        W_gater -= learning_rate * W_gater_gradient
        b_gater -= learning_rate * b_gater_gradient.T

        for expert, expert_gradients in zip(self.experts, experts_gradients):
            expert.update(expert_gradients, learning_rate)

    def parameters(self) -> list[list[list[np.ndarray]]]:
        return [[list(self.gater_layer)], *[expert.parameters() for expert in self.experts]]

    def copy(self) -> 'MoE':
        W_gater, b_gater = self.gater_layer
        return MoE((W_gater.copy(), b_gater.copy()), [expert.copy() for expert in self.experts])


class TargetPropagationMLP(Model):
    model_type = 'tpmlp'

    def __init__(self, hidden_layer: Layer, neuron_activation_mlps: list[MLP], output_layer: Layer, last_activation: str, neuron_activation_learning_rate_scale: float = 0.1):
        self.hidden_layer = hidden_layer
        self.neuron_activation_mlps = neuron_activation_mlps
        self.output_layer = output_layer
        self.last_activation = last_activation
        self.last_activation_fn = activation_fn_by_name(last_activation)
        self.neuron_activation_learning_rate_scale = neuron_activation_learning_rate_scale

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, TargetPropagationMLPCache]:
        W_hidden, b_hidden = self.hidden_layer
        W_output, b_output = self.output_layer

        Z_hidden = np.matmul(inputs, W_hidden) + np.squeeze(b_hidden, axis=1)
        A_hidden, neuron_activation_caches = apply_neuron_activations(self.neuron_activation_mlps, Z_hidden)

        Z_output = np.matmul(A_hidden, W_output) + np.squeeze(b_output, axis=1)
        A_output = self.last_activation_fn(Z_output)

        return A_output, TargetPropagationMLPCache(Z_hidden, A_hidden, neuron_activation_caches, Z_output)

    def backward(self, inputs: np.ndarray, cache: TargetPropagationMLPCache, loss_derivative: np.ndarray) -> tuple[Layer, list[list[Layer]], Layer]:
        Z_hidden, A_hidden, neuron_activation_caches, Z_output = cache
        batch_size = len(inputs)

        # note: the output layer gets the usual gradient, treating hidden values as fixed inputs
        Z_output_gradient = loss_derivative * self.last_activation_fn(Z_output, as_derivative=True)
        output_layer_gradient = (np.matmul(A_hidden.T, Z_output_gradient), np.sum(Z_output_gradient, keepdims=True, axis=0))

        # note: each hidden neuron trains alone toward its own target
        output_error = -loss_derivative * batch_size
        A_hidden_targets = self._hidden_targets_from_output_error(A_hidden, output_error)
        A_hidden_gradient = 2 * (A_hidden - A_hidden_targets) / batch_size

        neuron_activation_gradients, Z_hidden_gradient = backward_through_neuron_activations(self.neuron_activation_mlps, Z_hidden, neuron_activation_caches, A_hidden_gradient)

        hidden_layer_gradient = (np.matmul(inputs.T, Z_hidden_gradient), np.sum(Z_hidden_gradient, keepdims=True, axis=0))

        return hidden_layer_gradient, neuron_activation_gradients, output_layer_gradient

    def hidden_targets(self, cache: TargetPropagationMLPCache, labels: np.ndarray) -> np.ndarray:
        output_error = labels - self.last_activation_fn(cache.Z_output)

        return self._hidden_targets_from_output_error(cache.A_hidden, output_error)

    def _hidden_targets_from_output_error(self, A_hidden: np.ndarray, output_error: np.ndarray) -> np.ndarray:
        W_output, _ = self.output_layer # note: (n_hidden_neurons, 1)
        w_output = W_output[:, 0] # note: (n_hidden_neurons,)
        min_weight = 0.1

        # note: unlike the paper's sigmoid units, learned activations can go negative, so blame follows |h|; 1e-8 splits evenly when all are 0
        activity = np.abs(A_hidden) + 1e-8 # note: (batch_size, n_hidden_neurons)
        responsibilities = activity / np.sum(activity, axis=1, keepdims=True) # note: (batch_size, n_hidden_neurons)

        # note: for training stability, tiny output weights count as +-min_weight so residuals stay bounded, at the cost of an undelivered share
        safe_w_output = np.where(w_output < 0, -1, 1) * np.maximum(np.abs(w_output), min_weight) # note: (n_hidden_neurons,)

        # note: output_error is used as a small gradient step instead of the paper's full correction through the inverse sigmoid
        # note: (batch_size, n_hidden_neurons) * (batch_size, 1) / (n_hidden_neurons,) -> (batch_size, n_hidden_neurons)
        residuals = responsibilities * output_error / safe_w_output

        return A_hidden + residuals

    def divide(self) -> tuple[list['HiddenNeuron'], MLP]:
        W_hidden, b_hidden = self.hidden_layer
        W_output, b_output = self.output_layer

        hidden_neurons = [
            HiddenNeuron(W_hidden[:, u:u + 1].copy(), b_hidden[u:u + 1].copy(), activation_mlp.copy())
            for u, activation_mlp in enumerate(self.neuron_activation_mlps)
        ]
        output_mlp = MLP([(W_output.copy(), b_output.copy())], self.last_activation)

        return hidden_neurons, output_mlp

    @staticmethod
    def merge(hidden_neurons: list['HiddenNeuron'], output_mlp: MLP) -> 'TargetPropagationMLP':
        W_hidden = np.concatenate([neuron.w for neuron in hidden_neurons], axis=1)
        b_hidden = np.concatenate([neuron.b for neuron in hidden_neurons], axis=0)
        W_output, b_output = output_mlp.layers[0]

        return TargetPropagationMLP(
            (W_hidden, b_hidden),
            [neuron.activation_mlp.copy() for neuron in hidden_neurons],
            (W_output.copy(), b_output.copy()),
            output_mlp.last_activation,
        )

    def update(self, gradients: tuple[Layer, list[list[Layer]], Layer], learning_rate: float) -> None:
        hidden_layer_gradient, neuron_activation_gradients, output_layer_gradient = gradients

        update_layers([self.hidden_layer, self.output_layer], [hidden_layer_gradient, output_layer_gradient], learning_rate)

        for activation_mlp, neuron_activation_gradient in zip(self.neuron_activation_mlps, neuron_activation_gradients):
            activation_mlp.update(neuron_activation_gradient, learning_rate * self.neuron_activation_learning_rate_scale)

    def parameters(self) -> list:
        return [list(self.hidden_layer), [activation_mlp.parameters() for activation_mlp in self.neuron_activation_mlps], list(self.output_layer)]

    def copy(self) -> 'TargetPropagationMLP':
        W_hidden, b_hidden = self.hidden_layer
        W_output, b_output = self.output_layer

        return TargetPropagationMLP(
            (W_hidden.copy(), b_hidden.copy()),
            [activation_mlp.copy() for activation_mlp in self.neuron_activation_mlps],
            (W_output.copy(), b_output.copy()),
            self.last_activation,
            self.neuron_activation_learning_rate_scale,
        )

    def plot(self, dataset: str, algo: str) -> None:
        labeled_activations = [(f'hidden neuron {u + 1}', activation_mlp) for u, activation_mlp in enumerate(self.neuron_activation_mlps)]
        plot_learned_activations(labeled_activations, algo, self.model_type, dataset)

class HiddenNeuron:
    def __init__(self, w: np.ndarray, b: np.ndarray, activation_mlp: MLP):
        self.w = w
        self.b = b
        self.activation_mlp = activation_mlp

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        a, _ = self.activation_mlp.forward(np.matmul(inputs, self.w) + self.b[0])
        return a

    def parameters(self) -> list:
        return [[self.w, self.b], self.activation_mlp.parameters()]

    def copy(self) -> 'HiddenNeuron':
        return HiddenNeuron(self.w.copy(), self.b.copy(), self.activation_mlp.copy())

def update_layers(layers: list[Layer], gradients: list[Layer], learning_rate: float) -> None:
    for (W, b), (W_gradient, b_gradient) in zip(layers, gradients):
        W -= learning_rate * W_gradient
        b -= learning_rate * b_gradient.T

def apply_neuron_activations(activation_mlps: list[MLP], Z: np.ndarray) -> tuple[np.ndarray, list[MLPCache]]:
    A = np.zeros_like(Z)
    neuron_activation_caches = []

    for u, activation_mlp in enumerate(activation_mlps):
        A[:, u:u + 1], neuron_activation_cache = activation_mlp.forward(Z[:, u:u + 1])
        neuron_activation_caches.append(neuron_activation_cache)

    return A, neuron_activation_caches

def backward_through_neuron_activations(activation_mlps: list[MLP], Z: np.ndarray, neuron_activation_caches: list[MLPCache], A_gradient: np.ndarray) -> tuple[list[list[Layer]], np.ndarray]:
    Z_gradient = np.zeros_like(Z)
    neuron_activation_gradients = []

    for u, activation_mlp in enumerate(activation_mlps):
        neuron_activation_gradient, Z_gradient[:, u:u + 1] = activation_mlp.backward_with_inputs_gradient(Z[:, u:u + 1], neuron_activation_caches[u], A_gradient[:, u:u + 1])
        neuron_activation_gradients.append(neuron_activation_gradient)

    return neuron_activation_gradients, Z_gradient

def plot_learned_activations(labeled_activations: list[tuple[str, MLP | NeuronActivations]], algo: str, model_type: str, dataset: str) -> None:
    xs = np.linspace(-3, 3, 300)
    labeled_curves = [(label, activation.forward(xs.reshape(-1, 1))[0].reshape(xs.shape)) for label, activation in labeled_activations]
    most_different_overall = most_different_from_relu(xs, labeled_curves, overall_distance_from_relu)
    most_different_shape = most_different_from_relu(xs, labeled_curves, shape_distance_from_relu)
    color_by_label = colors_by_label(most_different_overall + most_different_shape)

    figure, (overall_axes, shape_axes) = plt.subplots(1, 2, figsize=(16, 6))
    plot_curves(overall_axes, xs, most_different_overall, color_by_label, f'Most different from ReLU: overall\n{len(most_different_overall)} of {len(labeled_curves)} hidden neurons')
    plot_curves(shape_axes, xs, most_different_shape, color_by_label, f'Most different from ReLU: shape (slope and offset ignored)\n{len(most_different_shape)} of {len(labeled_curves)} hidden neurons')
    figure.suptitle(f'algo: {algo}, model: {model_type}, dataset: {dataset}')
    figure.tight_layout()

    figure.savefig(f'./learned_activations_{algo}_{model_type}_{dataset}_{time.time()}.png')
    plt.close(figure)

def most_different_from_relu(xs: np.ndarray, labeled_curves: list[tuple[str, np.ndarray]], distance_fn) -> list[tuple[str, np.ndarray]]:
    return sorted(labeled_curves, key=lambda labeled_curve: distance_fn(xs, labeled_curve[1]), reverse=True)[:MAX_PLOTTED_ACTIVATIONS]

def colors_by_label(labeled_curves: list[tuple[str, np.ndarray]]) -> dict[str, tuple]:
    unique_labels = list(dict.fromkeys(label for label, _ in labeled_curves))
    paired_palette = plt.get_cmap('tab20').colors
    palette = paired_palette[0::2] + paired_palette[1::2]
    return {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}

def plot_curves(axes, xs: np.ndarray, labeled_curves: list[tuple[str, np.ndarray]], color_by_label: dict[str, tuple], title: str) -> None:
    for label, ys in labeled_curves:
        axes.plot(xs, ys, label=label, color=color_by_label[label])
    axes.plot(xs, relu(xs), 'k--', label='ReLU')
    axes.set_title(title)
    axes.set_xlabel('x')
    axes.set_ylabel('f(x)')
    axes.grid(True)
    axes.legend()

def overall_distance_from_relu(xs: np.ndarray, ys: np.ndarray) -> float:
    return float(np.mean(np.abs(ys - relu(xs))))

def shape_distance_from_relu(xs: np.ndarray, ys: np.ndarray) -> float:
    scaled_relu_basis = np.column_stack([relu(xs), np.ones_like(xs)])
    best_fit_coefficients, *_ = np.linalg.lstsq(scaled_relu_basis, ys, rcond=None)
    residual = ys - scaled_relu_basis @ best_fit_coefficients

    return float(np.sqrt(np.mean(residual ** 2)) / (np.std(ys) + 1e-12))
