import numpy as np

np.seterr(divide='ignore')

def forward(
    neural_network: list[tuple[np.ndarray, np.ndarray]],
    inputs: np.ndarray,
    last_activation: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n_layers = len(neural_network)
    last_layer_idx = n_layers - 1

    Zs = [None] * n_layers # note: layer input x layer weights + bias
    As = [None] * n_layers # note: layer activations

    last_activation = sigmoid if last_activation == 'sigmoid' else relu

    for l in range(0, n_layers):
        W, b = neural_network[l]
        layer_input = inputs if l == 0 else As[l - 1]

        Zs[l] = np.matmul(layer_input, W) + np.squeeze(b)
        As[l] = last_activation(Zs[l]) if l == last_layer_idx else relu(Zs[l])

    return As, Zs

def forward_mixture_of_experts(
    mixture_of_experts: tuple[list[tuple[np.ndarray, np.ndarray]], list[list[tuple[np.ndarray, np.ndarray]]]],
    inputs: np.ndarray,
    last_activation: str,
) -> tuple[np.ndarray, np.ndarray, list[list[np.ndarray]], list[list[np.ndarray]]]:
    gater, experts = mixture_of_experts
    W_gater, b_gater = gater[0]

    gates = softmax(np.matmul(inputs, W_gater) + np.squeeze(b_gater))

    experts_As = []
    experts_Zs = []
    for expert in experts:
        expert_As, expert_Zs = forward(expert, inputs, last_activation)
        experts_As.append(expert_As)
        experts_Zs.append(expert_Zs)

    predictions = np.zeros_like(experts_As[0][-1])
    for expert_idx, expert_As in enumerate(experts_As):
        predictions += gates[:, expert_idx:expert_idx + 1] * expert_As[-1]

    return predictions, gates, experts_As, experts_Zs

def relu(data: np.ndarray, as_derivative=False) -> np.ndarray:
    if as_derivative:
        return (data > 0).astype(np.float64)

    return np.maximum(0, data)

def sigmoid(data: np.ndarray, as_derivative=False):
    if (as_derivative):
        return np.exp(-data) / ((1 + np.exp(-data)) ** 2)
    
    return 1 / (1 + np.exp(-data))

def softmax(data: np.ndarray) -> np.ndarray:
    shifted_data = data - np.max(data, axis=1, keepdims=True)
    exponents = np.exp(shifted_data)

    return exponents / np.sum(exponents, axis=1, keepdims=True)

def cross_entropy_loss(labels: np.ndarray, predictions: np.ndarray, as_derivative=False) -> np.float64 | np.ndarray:
    is_binary = predictions.shape[1] == 1

    if is_binary:
        target = labels.reshape(-1, 1) if labels.ndim == 1 else labels
    else:
        target = np.zeros((labels.shape[0], predictions.shape[1]))
        target[np.arange(labels.shape[0]), labels[:, -1]] = 1

    clipped_predictions = np.clip(predictions, 1e-10, 1.0)

    if as_derivative:
        if is_binary:
            # Return true dL/dA so backward's sigmoid'(Z) multiplication gives correct dL/dZ
            denom = clipped_predictions * (1 - clipped_predictions)
            denom = np.clip(denom, 1e-10, 1.0)
            return (clipped_predictions - target) / (denom * labels.shape[0])
        return (clipped_predictions - target) / labels.shape[0]

    if is_binary:
        return -np.mean(target * np.log(clipped_predictions) + (1 - target) * np.log(1 - clipped_predictions))

    return -np.mean(np.sum(target * np.log(clipped_predictions), axis=1))

def accuracy(labels: np.ndarray, predictions: np.ndarray) -> np.float64:
    if predictions.shape[1] == 1:
        target = labels.reshape(-1, 1) if labels.ndim == 1 else labels
        return np.mean((predictions >= 0.5).astype(int) == target)
    return np.mean(np.argmax(predictions, axis=1, keepdims=True) == labels)

def copy_neural_network(neural_network: list[tuple[np.ndarray, np.ndarray]]):
    return [(W.copy(), b.copy()) for W, b in neural_network]

def copy_mixture_of_experts(mixture_of_experts: tuple[list[tuple[np.ndarray, np.ndarray]], list[list[tuple[np.ndarray, np.ndarray]]]]) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[list[tuple[np.ndarray, np.ndarray]]]]:
    gater, experts = mixture_of_experts

    return copy_neural_network(gater), [copy_neural_network(expert) for expert in experts]

def save_neural_network(neural_network: list[tuple[np.ndarray, np.ndarray]], filename: str):
    with open(filename, 'w') as f:
        for i, (W, b) in enumerate(neural_network):
            f.write(f'Layer {i + 1}\n')
            f.write('-----------\n')
            
            f.write(f'Weights shape: {W.shape}\n')
            f.write('Weights:\n')
            for row in W:
                row_string = ' '.join([f'{value:.5f}' for value in row])
                f.write(row_string + '\n')
            
            f.write('\n')
            
            f.write(f'Biases shape: {b.shape}\n')
            f.write('Biases:\n')
            for row in b:
                row_string = ' '.join([f'{value:.5f}' for value in row])
                f.write(row_string + '\n')

def count_parameters(neural_network: list[tuple[np.ndarray, np.ndarray]]) -> int:
    n_params = 0
    for W, b in neural_network:
        n_params += W.size
        n_params += b.size
    
    return n_params

def count_updated_parameters(initial_neural_network: list[tuple[np.ndarray, np.ndarray]], neural_network: list[tuple[np.ndarray, np.ndarray]]) -> int:
    n_params_updated = 0

    for layer_idx in range(len(neural_network)):
        for param_idx in range(len(neural_network[layer_idx])):
            for row_idx in range(len(neural_network[layer_idx][param_idx])):
                for col_idx in range(len(neural_network[layer_idx][param_idx][row_idx])):
                    if initial_neural_network[layer_idx][param_idx][row_idx][col_idx] != neural_network[layer_idx][param_idx][row_idx][col_idx]:
                        n_params_updated += 1

    return n_params_updated

def count_mixture_of_experts_parameters(mixture_of_experts: tuple[list[tuple[np.ndarray, np.ndarray]], list[list[tuple[np.ndarray, np.ndarray]]]]) -> int:
    gater, experts = mixture_of_experts
    n_params = count_parameters(gater)

    for expert in experts:
        n_params += count_parameters(expert)

    return n_params

def count_mixture_of_experts_updated_parameters(
    initial_mixture_of_experts: tuple[list[tuple[np.ndarray, np.ndarray]], list[list[tuple[np.ndarray, np.ndarray]]]],
    mixture_of_experts: tuple[list[tuple[np.ndarray, np.ndarray]], list[list[tuple[np.ndarray, np.ndarray]]]],
) -> int:
    initial_gater, initial_experts = initial_mixture_of_experts
    gater, experts = mixture_of_experts
    n_params_updated = count_updated_parameters(initial_gater, gater)

    for initial_expert, expert in zip(initial_experts, experts):
        n_params_updated += count_updated_parameters(initial_expert, expert)

    return n_params_updated