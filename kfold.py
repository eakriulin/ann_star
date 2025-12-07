import math
import typing
import numpy as np
import data
import nn

def eval(k: int, should_stratify: bool, dataset: str, architecture_type: str, initialization_method: str, train_fn: typing.Callable, **kwargs):
    inputs, labels = data.get(dataset, should_split=False)
    dataset_size = len(inputs)
    
    indices = np.random.permutation(dataset_size)
    inputs = inputs[indices]
    labels = labels[indices]
    
    accuracies = []
    losses = []
    updated_params_counts = []
    train_times = []

    for train_indices, test_indices in get_folds(k, should_stratify, labels):
        test_inputs = inputs[test_indices]
        test_labels = labels[test_indices]

        train_inputs = inputs[train_indices]
        train_labels = labels[train_indices]
        unique, counts = np.unique(train_labels, return_counts=True)

        neural_network = nn.initialize(dataset, architecture_type, initialization_method)
        accuracy, loss, updated_params_count, train_time = train_fn(neural_network, train_inputs, train_labels, test_inputs, test_labels, dataset=dataset, **kwargs)

        accuracies.append(accuracy)
        losses.append(loss)
        updated_params_counts.append(updated_params_count)
        train_times.append(train_time)

    print('\nK-Fold Results:')
    print(f'[Accuracy]: mean = {np.mean(accuracies):.4f}, std = {np.std(accuracies):.4f}')
    print(f'[Loss]: mean = {np.mean(losses):.4f}, std = {np.std(losses):.4f}')
    print(f'[# upd. parameters]: mean = {np.mean(updated_params_counts):.4f}, std = {np.std(updated_params_counts):.4f}')
    print(f'[Train time, sec]: mean = {np.mean(train_times):.4f}, std = {np.std(train_times):.4f}')

def get_folds(k: int, should_stratify: bool, labels: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if should_stratify:
        return get_stratified_folds(k, labels)

    return get_simple_folds(k, labels)

def get_simple_folds(k: int, labels: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    dataset_size = len(labels)
    fold_size = math.ceil(dataset_size / k)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        from_idx = i * fold_size
        to_idx = min((i * fold_size) + fold_size, dataset_size)

        test_indices = np.arange(from_idx, to_idx)
        train_indices = np.concat([np.arange(0, from_idx), np.arange(to_idx, dataset_size)])

        folds.append((train_indices, test_indices))

    return folds

def get_stratified_folds(k: int, labels: np.ndarray)  -> list[tuple[np.ndarray, np.ndarray]]:
    folds_by_label = []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        label_folds = np.array_split(label_indices, k)
        folds_by_label.append(label_folds)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        test_indices = np.concat([folds_by_label[l][i] for l in range(len(unique_labels))])
        train_indices = np.concat([np.concat([folds_by_label[l][j] for j in range(k) if j != i]) for l in range(len(unique_labels))])

        folds.append((train_indices, test_indices))

    return folds
