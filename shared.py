import numpy as np

np.seterr(divide='ignore')

def relu(data: np.ndarray, as_derivative=False) -> np.ndarray:
    if as_derivative:
        return (data > 0).astype(np.float64)

    return np.maximum(0, data)

def sigmoid(data: np.ndarray, as_derivative=False):
    if (as_derivative):
        return np.exp(-data) / ((1 + np.exp(-data)) ** 2)
    
    return 1 / (1 + np.exp(-data))

def linear(data: np.ndarray, as_derivative=False) -> np.ndarray:
    if as_derivative:
        return np.ones_like(data)

    return data

def activation_fn_by_name(name: str):
    return {'sigmoid': sigmoid, 'linear': linear}.get(name, relu)

def softmax(data: np.ndarray) -> np.ndarray:
    shifted_data = data - np.max(data, axis=1, keepdims=True)
    exponents = np.exp(shifted_data)

    return exponents / np.sum(exponents, axis=1, keepdims=True)

def mse_loss(labels: np.ndarray, predictions: np.ndarray, as_derivative=False) -> np.float64 | np.ndarray:
    target = target_matrix(labels, n_outputs=predictions.shape[1])
    clipped_predictions = np.clip(predictions, 1e-10, 1.0)

    if as_derivative:
        return (clipped_predictions - target) / labels.shape[0]

    return np.sum((clipped_predictions - target) ** 2) / (2 * labels.shape[0])

def target_matrix(labels: np.ndarray, n_outputs: int) -> np.ndarray:
    if n_outputs == 1:
        return labels.reshape(-1, 1) if labels.ndim == 1 else labels

    target = np.zeros((labels.shape[0], n_outputs))
    target[np.arange(labels.shape[0]), labels[:, -1]] = 1

    return target

def accuracy(labels: np.ndarray, predictions: np.ndarray) -> np.float64:
    if predictions.shape[1] == 1:
        return np.mean((predictions >= 0.5).astype(int) == target_matrix(labels, n_outputs=1))
    return np.mean(np.argmax(predictions, axis=1, keepdims=True) == labels)

def evaluate(model, inputs: np.ndarray, labels: np.ndarray) -> tuple[np.float64, np.float64]:
    predictions, _ = model.forward(inputs)
    return mse_loss(labels, predictions), accuracy(labels, predictions)

def flatten_parameters(parameters) -> list[np.ndarray]:
    if isinstance(parameters, np.ndarray):
        return [parameters]

    return [p for nested in parameters for p in flatten_parameters(nested)]

def count_parameters(model) -> int:
    return sum(p.size for p in flatten_parameters(model.parameters()))

def count_updated_parameters(initial_model, model) -> int:
    return int(sum(np.sum(initial_p != p) for initial_p, p in zip(flatten_parameters(initial_model.parameters()), flatten_parameters(model.parameters()))))

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
