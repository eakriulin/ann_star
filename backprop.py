import time
import numpy as np
from models import Model
from shared import accuracy, cross_entropy_loss, evaluate, count_parameters, count_updated_parameters

def train(
    model: Model,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    test_inputs: np.ndarray,
    test_labels: np.ndarray,
    learning_rate: float,
    batch_size: int,
    n_epochs: int,
    dataset: str
) -> tuple[float, float, int, float]:
    start_time = time.time()

    initial_model = model.copy()

    print('Training...')
    
    for e in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_accuracy = 0

        for batch_idx in range(0, len(train_inputs), batch_size):
            batch_inputs = train_inputs[batch_idx:batch_idx + batch_size]
            batch_labels = train_labels[batch_idx:batch_idx + batch_size]

            batch_predictions, batch_cache = model.forward(batch_inputs)
            batch_loss = cross_entropy_loss(batch_labels, batch_predictions)

            batch_loss_derivative = cross_entropy_loss(batch_labels, batch_predictions, as_derivative=True)
            gradients = model.backward(batch_inputs, batch_cache, batch_loss_derivative)
            model.update(gradients, learning_rate)

            epoch_loss += batch_loss * len(batch_inputs)
            epoch_accuracy += accuracy(batch_labels, batch_predictions) * len(batch_inputs)

        if e % 1 == 0:
            epoch_loss /= len(train_inputs)
            epoch_accuracy /= len(train_inputs)
            print(f'\tepoch {e}: loss {epoch_loss:.3f}, accuracy {epoch_accuracy:.3f}')

    train_time = time.time() - start_time
    
    test_loss, test_accuracy = evaluate(model, test_inputs, test_labels)
    print(f'--> Test: loss = {test_loss:.4f}, accuracy = {test_accuracy:.4f}')

    updated_params_count = count_updated_parameters(initial_model, model)
    print(f'Updated {updated_params_count} parameters out of {count_parameters(model)}')

    print(f'Train time, sec = {train_time:.2f}')

    return test_accuracy, test_loss, updated_params_count, train_time
