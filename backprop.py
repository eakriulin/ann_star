import time
import numpy as np
from shared import forward, relu, accuracy, cross_entropy_loss, sigmoid, copy_neural_network, save_neural_network, count_parameters, count_updated_parameters, forward_mixture_of_experts, copy_mixture_of_experts, count_mixture_of_experts_parameters, count_mixture_of_experts_updated_parameters

def train(
    neural_network: list[tuple[np.ndarray, np.ndarray]],
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    test_inputs: np.ndarray,
    test_labels: np.ndarray,
    learning_rate: float,
    batch_size: int,
    n_epochs: int,
    last_activation: str,
    dataset: str
) -> tuple[float, float]:
    start_time = time.time()

    initial_neural_network = copy_neural_network(neural_network)

    print('Training...')
    
    for e in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_accuracy = 0

        for batch_idx in range(0, len(train_inputs), batch_size):
            batch_inputs = train_inputs[batch_idx:batch_idx + batch_size]
            batch_labels = train_labels[batch_idx:batch_idx + batch_size]

            batch_As, batch_Zs = forward(neural_network, batch_inputs, last_activation)
            batch_loss = cross_entropy_loss(batch_labels, batch_As[-1])

            batch_loss_derivative = cross_entropy_loss(batch_labels, batch_As[-1], as_derivative=True)
            gradients = backward(neural_network, batch_inputs, batch_As, batch_Zs, batch_loss_derivative, last_activation)
            update_parameters(neural_network, gradients, learning_rate)

            epoch_loss += batch_loss * len(batch_inputs)
            epoch_accuracy += accuracy(batch_labels, batch_As[-1]) * len(batch_inputs)

        if e % 1 == 0:
            epoch_loss /= len(train_inputs)
            epoch_accuracy /= len(train_inputs)
            print(f'\tepoch {e}: loss {epoch_loss:.3f}, accuracy {epoch_accuracy:.3f}')

    train_time = time.time() - start_time
    
    test_As, _ = forward(neural_network, test_inputs, last_activation)
    test_loss = cross_entropy_loss(test_labels, test_As[-1])
    test_accuracy = accuracy(test_labels, test_As[-1])
    print(f'--> Test: loss = {test_loss:.4f}, accuracy = {test_accuracy:.4f}')

    updated_params_count = count_updated_parameters(initial_neural_network, neural_network)
    print(f'Updated {updated_params_count} parameters out of {count_parameters(neural_network)}')

    print(f'Train time, sec = {train_time:.2f}')

    return test_accuracy, test_loss, updated_params_count, train_time

def train_mixture_of_experts(
    mixture_of_experts: tuple[list[tuple[np.ndarray, np.ndarray]], list[list[tuple[np.ndarray, np.ndarray]]]],
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    test_inputs: np.ndarray,
    test_labels: np.ndarray,
    learning_rate: float,
    batch_size: int,
    n_epochs: int,
    last_activation: str,
    dataset: str
) -> tuple[float, float, int, float]:
    start_time = time.time()

    initial_mixture_of_experts = copy_mixture_of_experts(mixture_of_experts)

    print('Training...')

    for e in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_accuracy = 0

        for batch_idx in range(0, len(train_inputs), batch_size):
            batch_inputs = train_inputs[batch_idx:batch_idx + batch_size]
            batch_labels = train_labels[batch_idx:batch_idx + batch_size]

            batch_predictions, batch_gates, batch_experts_As, batch_experts_Zs = forward_mixture_of_experts(mixture_of_experts, batch_inputs, last_activation)
            batch_loss = cross_entropy_loss(batch_labels, batch_predictions)

            batch_loss_derivative = cross_entropy_loss(batch_labels, batch_predictions, as_derivative=True)
            gater_gradients, experts_gradients = backward_mixture_of_experts(mixture_of_experts, batch_inputs, batch_gates, batch_experts_As, batch_experts_Zs, batch_loss_derivative, last_activation)
            update_mixture_of_experts_parameters(mixture_of_experts, gater_gradients, experts_gradients, learning_rate)

            epoch_loss += batch_loss * len(batch_inputs)
            epoch_accuracy += accuracy(batch_labels, batch_predictions) * len(batch_inputs)

        if e % 1 == 0:
            epoch_loss /= len(train_inputs)
            epoch_accuracy /= len(train_inputs)
            print(f'\tepoch {e}: loss {epoch_loss:.3f}, accuracy {epoch_accuracy:.3f}')

    train_time = time.time() - start_time

    test_predictions, _, _, _ = forward_mixture_of_experts(mixture_of_experts, test_inputs, last_activation)
    test_loss = cross_entropy_loss(test_labels, test_predictions)
    test_accuracy = accuracy(test_labels, test_predictions)
    print(f'--> Test: loss = {test_loss:.4f}, accuracy = {test_accuracy:.4f}')

    updated_params_count = count_mixture_of_experts_updated_parameters(initial_mixture_of_experts, mixture_of_experts)
    print(f'Updated {updated_params_count} parameters out of {count_mixture_of_experts_parameters(mixture_of_experts)}')

    print(f'Train time, sec = {train_time:.2f}')

    return test_accuracy, test_loss, updated_params_count, train_time

def backward(
    neural_network: list[tuple[np.ndarray, np.ndarray]],
    inputs: np.ndarray,
    As: np.ndarray,
    Zs: np.ndarray,
    loss_derivative: np.ndarray,
    last_activation: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_layers = len(neural_network)
    last_layer_idx = n_layers - 1

    gradients = [None] * n_layers
    last_activation_fn = sigmoid if last_activation == 'sigmoid' else relu

    for l in range(last_layer_idx, -1, -1):
        if l == last_layer_idx:
            delta = loss_derivative * last_activation_fn(Zs[l], as_derivative=True)
        else:
            W_of_next_layer, _ = neural_network[l + 1]
            delta = np.matmul(delta, W_of_next_layer.T) * relu(Zs[l], as_derivative=True)

        W_gradient = None
        if l != 0:
            W_gradient = np.matmul(As[l - 1].T, delta)
        else:
            W_gradient = np.matmul(inputs.T, delta)

        b_gradient = np.sum(delta, keepdims=True, axis=0)

        gradients[l] = (W_gradient, b_gradient)

    return gradients

def backward_mixture_of_experts(
    mixture_of_experts: tuple[list[tuple[np.ndarray, np.ndarray]], list[list[tuple[np.ndarray, np.ndarray]]]],
    inputs: np.ndarray,
    gates: np.ndarray,
    experts_As: list[list[np.ndarray]],
    experts_Zs: list[list[np.ndarray]],
    loss_derivative: np.ndarray,
    last_activation: str,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[list[tuple[np.ndarray, np.ndarray]]]]:
    _, experts = mixture_of_experts

    experts_gradients = []
    gates_gradient = np.zeros_like(gates)

    for expert_idx, expert in enumerate(experts):
        expert_gate = gates[:, expert_idx:expert_idx + 1]
        expert_loss_derivative = expert_gate * loss_derivative
        experts_gradients.append(backward(expert, inputs, experts_As[expert_idx], experts_Zs[expert_idx], expert_loss_derivative, last_activation))

        expert_predictions = experts_As[expert_idx][-1]
        gates_gradient[:, expert_idx] = np.sum(loss_derivative * expert_predictions, axis=1)

    weighted_average_gates_gradient = np.sum(gates * gates_gradient, axis=1, keepdims=True)
    gater_delta = gates * (gates_gradient - weighted_average_gates_gradient)

    gater_W_gradient = np.matmul(inputs.T, gater_delta)
    gater_b_gradient = np.sum(gater_delta, keepdims=True, axis=0)

    return [(gater_W_gradient, gater_b_gradient)], experts_gradients

def update_parameters(
    neural_network: list[tuple[np.ndarray, np.ndarray]],
    gradients: list[tuple[np.ndarray, np.ndarray]],
    learning_rate: float,
) -> None:
    n_layers = len(neural_network)

    for l in range(0, n_layers):
        W, b = neural_network[l]
        W_grad, b_grad = gradients[l]

        W -= learning_rate * W_grad
        b -= learning_rate * b_grad.T

def update_mixture_of_experts_parameters(
    mixture_of_experts: tuple[list[tuple[np.ndarray, np.ndarray]], list[list[tuple[np.ndarray, np.ndarray]]]],
    gater_gradients: list[tuple[np.ndarray, np.ndarray]],
    experts_gradients: list[list[tuple[np.ndarray, np.ndarray]]],
    learning_rate: float,
) -> None:
    gater, experts = mixture_of_experts

    update_parameters(gater, gater_gradients, learning_rate)

    for i in range(len(experts)):
        update_parameters(experts[i], experts_gradients[i], learning_rate)