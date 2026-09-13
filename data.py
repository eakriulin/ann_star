import os
import numpy as np
from gen.stripes_dataset import generate_stripes_dataset

def get(dataset: str, should_split: bool = True)  -> tuple:
    if dataset == 'iris':
        return get_iris(should_split)
    
    if dataset == 'wine':
        return get_wine(should_split)

    if dataset == 'digits':
        return get_digits(should_split)

    if dataset == 'stripes2':
        return get_stripes(should_split, 2)

    if dataset == 'stripes3':
        return get_stripes(should_split, 3)

    raise Exception(f'Unknown dataset — {dataset}')

def get_iris(should_split: bool = True) -> tuple:
    class_name_to_id = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

    data = np.loadtxt(os.path.join('datasets', 'iris', 'iris.data'), dtype=np.float64, delimiter=',', converters={4: lambda class_name: class_name_to_id[class_name]})

    inputs = data[:, :-1]
    labels = data[:, -1].reshape(len(data), 1).astype(int)

    inputs = standardize(inputs)

    if should_split:
        return split(inputs, labels)
    
    return inputs, labels

def get_wine(should_split: bool = True) -> tuple:
    data = np.loadtxt(os.path.join('datasets', 'wine', 'wine.data'), dtype=np.float64, delimiter=',', converters={0: lambda class_id: int(class_id) - 1})

    inputs = data[:, 1:]
    labels = data[:, 0].reshape(len(data), 1).astype(int)

    inputs = standardize(inputs)

    if should_split:
        return split(inputs, labels)

    return inputs, labels

def get_digits(should_split: bool = True) -> tuple:
    train_data = np.loadtxt(os.path.join('datasets', 'digits', 'optdigits.tra'), dtype=np.float64, delimiter=',')
    test_data = np.loadtxt(os.path.join('datasets', 'digits', 'optdigits.tes'), dtype=np.float64, delimiter=',')

    train_inputs = train_data[:, :-1] / 16.0
    train_labels = train_data[:, -1].reshape(len(train_data), 1).astype(int)

    test_inputs = test_data[:, :-1] / 16.0
    test_labels = test_data[:, -1].reshape(len(test_data), 1).astype(int)

    if should_split:
        train_indices = np.random.permutation(len(train_inputs))
        train_inputs = train_inputs[train_indices]
        train_labels = train_labels[train_indices]

        test_indices = np.random.permutation(len(test_inputs))
        test_inputs = test_inputs[test_indices]
        test_labels = test_labels[test_indices]

        return train_inputs, train_labels, test_inputs, test_labels

    inputs = np.concat([train_inputs, test_inputs])
    labels = np.concat([train_labels, test_labels])

    return inputs, labels

def get_stripes(should_split: bool = True, n_classes: int = 2) -> tuple:
    stripes_per_class = 2
    n_samples = n_classes * 200
    weights = [1 / n_classes for _ in range(n_classes)]

    inputs, labels = generate_stripes_dataset(
        n_classes=n_classes,
        n_samples=n_samples,
        weights=weights,
        stripe_spacing=1.0,
        stripe_spread=0.25,
        stripes_per_class=stripes_per_class,
        seed=42,
    )

    inputs = standardize(inputs)
    labels = labels.reshape(len(labels), 1).astype(int)

    if should_split:
        return split(inputs, labels)

    return inputs, labels

def standardize(data: np.ndarray) -> np.ndarray:
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def split(inputs: np.ndarray, labels: np.ndarray, test_set_percentage=0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_classes = np.unique(labels)
    class_indices = {c: np.where(labels == c)[0] for c in unique_classes}
    
    train_set_indices, test_set_indices = [], []
    
    for c in unique_classes:
        indices = class_indices[c]
        np.random.shuffle(indices)
        
        n_test = int(len(indices) * test_set_percentage)
        
        test_set_indices.extend(indices[:n_test])
        train_set_indices.extend(indices[n_test:])

    train_inputs = inputs[train_set_indices]
    train_labels = labels[train_set_indices]

    test_inputs = inputs[test_set_indices]
    test_labels = labels[test_set_indices]

    shuffled_train_set_indices = np.random.permutation(len(train_inputs))
    shuffled_test_set_indices = np.random.permutation(len(test_inputs))

    return train_inputs[shuffled_train_set_indices], train_labels[shuffled_train_set_indices], test_inputs[shuffled_test_set_indices], test_labels[shuffled_test_set_indices]