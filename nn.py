import numpy as np

def initialize(dataset: str, architecture_type: str, method: str) -> list[tuple[np.ndarray, np.ndarray]]:
    return {
        'iris': {
            '2': [
                layer(n_features_in=4, n_features_out=3, method=method)
            ], 
            '3': [
                layer(n_features_in=4, n_features_out=8, method=method),
                layer(n_features_in=8, n_features_out=3, method=method)
            ],
            '4': [
                layer(n_features_in=4, n_features_out=16, method=method),
                layer(n_features_in=16, n_features_out=8, method=method),
                layer(n_features_in=8, n_features_out=3, method=method)
            ],
        },
        'wine': {
            '2': [
                layer(n_features_in=13, n_features_out=3, method=method),
            ],
            '3': [
                layer(n_features_in=13, n_features_out=16, method=method),
                layer(n_features_in=16, n_features_out=3, method=method),
            ],
            '4': [
                layer(n_features_in=13, n_features_out=32, method=method),
                layer(n_features_in=32, n_features_out=16, method=method),
                layer(n_features_in=16, n_features_out=3, method=method),
            ],
        },
        'digits': {
            '2': [
                layer(n_features_in=64, n_features_out=10, method=method),
            ],
            '3': [
                layer(n_features_in=64, n_features_out=32, method=method),
                layer(n_features_in=32, n_features_out=10, method=method),
            ],
            '4': [
                layer(n_features_in=64, n_features_out=64, method=method),
                layer(n_features_in=64, n_features_out=32, method=method),
                layer(n_features_in=32, n_features_out=10, method=method),
            ],
        },
        'stripes2': {
            '2': [
                layer(n_features_in=2, n_features_out=1, method=method),
            ],
            '3': [
                layer(n_features_in=2, n_features_out=3, method=method),
                layer(n_features_in=3, n_features_out=1, method=method),
            ],
        },
        'stripes3': {
            '2': [
                layer(n_features_in=2, n_features_out=3, method=method),
            ],
            '3': [
                layer(n_features_in=2, n_features_out=5, method=method),
                layer(n_features_in=5, n_features_out=3, method=method),
            ],
        },
    }[dataset][architecture_type]

def initialize_mixture_of_experts(dataset: str, architecture_type: str, method: str) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[list[tuple[np.ndarray, np.ndarray]]]]:
    return {
        'iris': {
            '2': mixture_of_experts(
                n_features_in=4,
                experts=[
                    [
                        layer(n_features_in=4, n_features_out=3, method=method),
                    ],
                    [
                        layer(n_features_in=4, n_features_out=3, method=method),
                    ],
                    [
                        layer(n_features_in=4, n_features_out=3, method=method),
                    ],
                ],
                method=method,
            ),
        },
        'wine': {
            '2': mixture_of_experts(
                n_features_in=13,
                experts=[
                    [
                        layer(n_features_in=13, n_features_out=3, method=method),
                    ],
                    [
                        layer(n_features_in=13, n_features_out=3, method=method),
                    ],
                    [
                        layer(n_features_in=13, n_features_out=3, method=method),
                    ],
                ],
                method=method,
            ),
        },
        'digits': {
            '2': mixture_of_experts(
                n_features_in=64,
                experts=[
                    [
                        layer(n_features_in=64, n_features_out=10, method=method),
                    ],
                    [
                        layer(n_features_in=64, n_features_out=10, method=method),
                    ],
                    [
                        layer(n_features_in=64, n_features_out=10, method=method),
                    ],
                ],
                method=method,
            ),
        },
        'stripes2': {
            '2': mixture_of_experts(
                n_features_in=2,
                experts=[
                    [
                        layer(n_features_in=2, n_features_out=1, method=method),
                    ],
                    [
                        layer(n_features_in=2, n_features_out=1, method=method),
                    ],
                    [
                        layer(n_features_in=2, n_features_out=1, method=method),
                    ],
                ],
                method=method
            ),
            '3': mixture_of_experts(
                n_features_in=2,
                experts=[
                    [
                        layer(n_features_in=2, n_features_out=3, method=method),
                        layer(n_features_in=3, n_features_out=1, method=method),
                    ],
                    [
                        layer(n_features_in=2, n_features_out=3, method=method),
                        layer(n_features_in=3, n_features_out=1, method=method),
                    ],
                    [
                        layer(n_features_in=2, n_features_out=3, method=method),
                        layer(n_features_in=3, n_features_out=1, method=method),
                    ],
                ],
                method=method
            ),
        },
        'stripes3': {
            '2': mixture_of_experts(
                n_features_in=2,
                experts=[
                    [
                        layer(n_features_in=2, n_features_out=3, method=method),
                    ],
                    [
                        layer(n_features_in=2, n_features_out=3, method=method),
                    ]
                ],
                method=method
            ),
            '3': mixture_of_experts(
                n_features_in=2,
                experts=[
                    [
                        layer(n_features_in=2, n_features_out=5, method=method),
                        layer(n_features_in=5, n_features_out=3, method=method),
                    ],
                    [
                        layer(n_features_in=2, n_features_out=5, method=method),
                        layer(n_features_in=5, n_features_out=3, method=method),
                    ],
                    [
                        layer(n_features_in=2, n_features_out=5, method=method),
                        layer(n_features_in=5, n_features_out=3, method=method),
                    ],
                ],
                method=method
            ),
        }
    }[dataset][architecture_type]

def mixture_of_experts(
    n_features_in: int,
    experts: list[list[tuple[np.ndarray, np.ndarray]]],
    method: str
) -> tuple[
    list[tuple[np.ndarray, np.ndarray]],
    list[list[tuple[np.ndarray, np.ndarray]]]
]:
    gater = [layer(n_features_in=n_features_in, n_features_out=len(experts), method=method)]

    return gater, experts

def layer(n_features_in: int, n_features_out: int, method: str) -> tuple[np.ndarray, np.ndarray]:
    if method == 'zeros':
        W = np.zeros((n_features_in, n_features_out))
        b = np.zeros((n_features_out, 1))
    else:
        std = np.sqrt(2.0 / n_features_in)
        W = np.random.normal(0, std, (n_features_in, n_features_out))
        b = np.zeros((n_features_out, 1))

    return W, b