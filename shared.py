import numpy as np

np.seterr(divide='ignore')

def forward(
    neural_network: list[tuple[np.ndarray, np.ndarray]],
    inputs: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n_layers = len(neural_network)
    last_layer_idx = n_layers - 1

    Zs = [None] * n_layers # note: layer input x layer weights + bias
    As = [None] * n_layers # note: layer activations

    for l in range(0, n_layers):
        W, b = neural_network[l]
        layer_input = inputs if l == 0 else As[l - 1]

        Zs[l] = np.matmul(layer_input, W) + np.squeeze(b)
        # As[l] = sigmoid(Zs[l]) if l == last_layer_idx else relu(Zs[l])
        As[l] = relu(Zs[l])

    return As, Zs

def relu(data: np.ndarray, as_derivative=False) -> np.ndarray:
    if as_derivative:
        return (data > 0).astype(np.float64)

    return np.maximum(0, data)

def sigmoid(data: np.ndarray, as_derivative=False):
    if (as_derivative):
        return np.exp(-data) / ((1 + np.exp(-data)) ** 2)
    
    return 1 / (1 + np.exp(-data))

def cross_entropy_loss(labels: np.ndarray, predictions: np.ndarray, as_derivative=False, as_derivative_for_softmax=False) -> np.float64 | np.ndarray:
    one_hot_labels = np.zeros((labels.shape[0], predictions.shape[1]))
    one_hot_labels[np.arange(labels.shape[0]), labels[:, -1]] = 1

    if as_derivative_for_softmax:
        return predictions - one_hot_labels
    
    clipped_predictions = np.clip(predictions, 1e-10, 1.0)

    if as_derivative:
        return (clipped_predictions - one_hot_labels) / labels.shape[0]

    return -np.mean(np.sum(one_hot_labels * np.log(clipped_predictions), axis=1))

def accuracy(labels: np.ndarray, predictions: np.ndarray) -> np.float64:
    return np.mean(np.argmax(predictions, axis=1, keepdims=True) == labels)

def copy_neural_network(neural_network: list[tuple[np.ndarray, np.ndarray]]):
    return [(W.copy(), b.copy()) for W, b in neural_network]

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