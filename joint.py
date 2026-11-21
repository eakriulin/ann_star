from concurrent.futures import Executor, ProcessPoolExecutor
from typing import Callable
import numpy as np
import backprop
from a_star import Node, all_parameters, expand
from models import NeuronActivationMLP, NeuronActivations
from shared import accuracy, mse_loss, count_parameters

def train(
    model: NeuronActivationMLP,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    test_inputs: np.ndarray,
    test_labels: np.ndarray,
    learning_rate: float,
    batch_size: int,
    n_epochs: int,
    goal: float | None,
    dataset: str,
    should_plot: bool,
    should_stop_after_epoch: Callable[[NeuronActivationMLP, int], bool] | None = None,
) -> tuple[float, float, int, float]:
    with ProcessPoolExecutor() as pool:
        def search_neuron_activations_in_pool(model: NeuronActivationMLP, batch_inputs: np.ndarray, batch_labels: np.ndarray) -> None:
            search_neuron_activations(model, batch_inputs, batch_labels, pool)

        return backprop.train(
            model,
            train_inputs,
            train_labels,
            test_inputs,
            test_labels,
            learning_rate,
            batch_size,
            n_epochs,
            dataset,
            should_plot,
            after_each_step=search_neuron_activations_in_pool,
            goal=goal,
            algo='joint',
            should_stop_after_epoch=should_stop_after_epoch,
        )

def search_neuron_activations(model: NeuronActivationMLP, batch_inputs: np.ndarray, batch_labels: np.ndarray, pool: Executor) -> None:
    _, cache = model.forward(batch_inputs)

    futures_per_layer = []
    for l, neuron_activations in enumerate(model.neuron_activations_per_layer):
        futures = []
        for u in range(neuron_activations.n_neurons()):
            Z_neuron = cache.Z_per_layer[l][:, u:u + 1]
            futures.append(pool.submit(best_neuron_activation, model, l, u, Z_neuron, cache.A_per_layer[l], batch_labels))
        futures_per_layer.append(futures)

    winners_per_layer = []
    for futures in futures_per_layer:
        winners = []
        for future in futures:
            winners.append(future.result())
        winners_per_layer.append(NeuronActivations.stack(winners))

    model.neuron_activations_per_layer = winners_per_layer

def best_neuron_activation(model: NeuronActivationMLP, l: int, u: int, Z_neuron: np.ndarray, A_layer: np.ndarray, batch_labels: np.ndarray) -> NeuronActivations:
    def evaluate_neuron_activation(neuron_activation: NeuronActivations) -> tuple[float, float]:
        A_neuron, _ = neuron_activation.forward(Z_neuron)
        A_layer_with_neuron = A_layer.copy()
        A_layer_with_neuron[:, u:u + 1] = A_neuron

        predictions = model.forward_from_layer(l + 1, A_layer_with_neuron)
        return mse_loss(batch_labels, predictions), accuracy(batch_labels, predictions)

    root_neuron_activation = model.neuron_activations_per_layer[l].neuron(u)
    root_loss, root_accuracy = evaluate_neuron_activation(root_neuron_activation)
    root = Node(model=root_neuron_activation, depth=0, loss=root_loss, accuracy=root_accuracy)

    best = root
    for candidate in expand(root, all_parameters, evaluate_neuron_activation, count_parameters(root_neuron_activation)):
        if is_better(candidate, best):
            best = candidate

    return best.model

def is_better(candidate: Node, best: Node) -> bool:
    if candidate.accuracy != best.accuracy:
        return candidate.accuracy > best.accuracy

    return candidate.loss < best.loss
