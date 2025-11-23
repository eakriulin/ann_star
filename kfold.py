import math
import typing
import numpy as np
import data
import nn

def eval(k: int, dataset: str, initialization_method: str, train_fn: typing.Callable, **kwargs):
    inputs, labels = data.get(dataset, should_split=False)
    dataset_size = len(inputs)
    
    indices = np.random.permutation(dataset_size)
    inputs = inputs[indices]
    labels = labels[indices]
    
    fold_size = math.ceil(dataset_size / k)
    accuracies = []
    losses = []

    for i in range(k):
        from_idx = i * fold_size
        to_idx = min((i * fold_size) + fold_size, dataset_size)

        test_indices = np.arange(from_idx, to_idx)
        train_indices = np.concat([np.arange(0, from_idx), np.arange(to_idx, dataset_size)])

        test_inputs = inputs[test_indices]
        test_labels = labels[test_indices]

        train_inputs = inputs[train_indices]
        train_labels = labels[train_indices]

        neural_network = nn.initialize(dataset, initialization_method)
        accuracy, loss = train_fn(neural_network, train_inputs, train_labels, test_inputs, test_labels, dataset=dataset, **kwargs)

        accuracies.append(accuracy)
        losses.append(loss)

    print('\nK-Fold Results:')
    print(f'[Accuracy]: mean = {np.mean(accuracies):.4f}, std = {np.std(accuracies):.4f}')
    print(f'[Loss]: mean = {np.mean(losses):.4f}, std = {np.std(losses):.4f}')