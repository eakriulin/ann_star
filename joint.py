from concurrent.futures import Executor, ProcessPoolExecutor
import numpy as np
import backprop
from a_star import Node, expand
from models import MLP, NeuronActivationMLP
from shared import evaluate, count_parameters

def train(
    model: NeuronActivationMLP,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    test_inputs: np.ndarray,
    test_labels: np.ndarray,
    learning_rate: float,
    batch_size: int,
    n_epochs: int,
    dataset: str,
    should_plot: bool,
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
        )

def search_neuron_activations(model: NeuronActivationMLP, batch_inputs: np.ndarray, batch_labels: np.ndarray, pool: Executor) -> None:
    root_loss, root_accuracy = evaluate(model, batch_inputs, batch_labels)
    root = Node(model=model, depth=0, loss=root_loss, accuracy=root_accuracy)

    futures_per_layer = []
    for l, activation_mlps in enumerate(model.neuron_activation_mlps_per_layer):
        futures = []
        for u in range(len(activation_mlps)):
            futures.append(pool.submit(best_neuron_activation_mlp, root, l, u, batch_inputs, batch_labels))
        futures_per_layer.append(futures)

    winners_per_layer = []
    for futures in futures_per_layer:
        winners = []
        for future in futures:
            winners.append(future.result())
        winners_per_layer.append(winners)

    model.neuron_activation_mlps_per_layer = winners_per_layer

def best_neuron_activation_mlp(root: Node, l: int, u: int, batch_inputs: np.ndarray, batch_labels: np.ndarray) -> MLP:
    def neuron_activation_parameters(model: NeuronActivationMLP) -> list:
        return model.neuron_activation_mlps_per_layer[l][u].parameters()

    n_params_to_update = count_parameters(root.model.neuron_activation_mlps_per_layer[l][u])

    best = root
    for candidate in expand(root, neuron_activation_parameters, batch_inputs, batch_labels, n_params_to_update):
        if is_better(candidate, best):
            best = candidate

    return best.model.neuron_activation_mlps_per_layer[l][u]

def is_better(candidate: Node, best: Node) -> bool:
    if candidate.accuracy != best.accuracy:
        return candidate.accuracy > best.accuracy

    return candidate.loss < best.loss
