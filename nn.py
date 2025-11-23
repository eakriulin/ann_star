import numpy as np

def initialize(dataset: str, architecture_type: str, method: str) -> list[tuple[np.ndarray, np.ndarray]]:
    return {
        'iris': {
            'no': [
                layer(n_features_in=4, n_features_out=3, method=method)
            ], 
            'single': [
                layer(n_features_in=4, n_features_out=8, method=method),
                layer(n_features_in=8, n_features_out=3, method=method)
            ]
        },
        'wine': {
            'no': [
                layer(n_features_in=13, n_features_out=3, method=method),
            ],
            'single': [
                layer(n_features_in=13, n_features_out=16, method=method),
                layer(n_features_in=16, n_features_out=3, method=method),
            ]
        },
        'digits': {
            'no': [
                layer(n_features_in=64, n_features_out=10, method=method),
            ],
            'single': [
                layer(n_features_in=64, n_features_out=32, method=method),
                layer(n_features_in=32, n_features_out=10, method=method),
            ]
        },
        'mnist': {
            'no': [
                layer(n_features_in=784, n_features_out=10, method=method),
            ],
            'single': [
                layer(n_features_in=784, n_features_out=128, method=method),
                layer(n_features_in=128, n_features_out=10, method=method)
            ]
        },
    }[dataset][architecture_type]

def layer(n_features_in: int, n_features_out: int, method: str) -> tuple[np.ndarray, np.ndarray]:
    if method == 'zeros':
        W = np.zeros((n_features_in, n_features_out))
        b = np.zeros((n_features_out, 1))
    else:
        std = np.sqrt(2.0 / n_features_in)
        W = np.random.normal(0, std, (n_features_in, n_features_out))
        b = np.zeros((n_features_out, 1))

    return W, b