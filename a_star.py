import time
import heapq
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable
from shared import forward, accuracy, save_neural_network, cross_entropy_loss, copy_neural_network, count_parameters, count_updated_parameters

def train(
    neural_network: list[tuple[np.ndarray, np.ndarray]],
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    test_inputs: np.ndarray,
    test_labels: np.ndarray,
    goal: float,
    batch_size: int,
    dataset: str,
    last_activation: str,
    should_plot_accuracy_vs_depth: bool,
) -> tuple[float, float]:
    start_time = time.time()

    initial_neural_network = copy_neural_network(neural_network)
    
    n_params = count_parameters(neural_network)
    n_params_to_update = 200
    n_solutions = 1

    actions = [0.001, -0.001]
    solutions: list[Node] = []

    def cost_fn(node: Node):
        """
        Computes the cost associated with a given node based on its depth, loss, and accuracy.
        - Depth-based cost: Penalizes deeper nodes relative to the total number of parameters.
        - Loss-based cost: Penalizes nodes with higher loss values.
        - Accuracy-based cost: Penalizes nodes with lower accuracy.
        """

        d = 0.5
        l = 0.2
        a = 0.3

        return d * (node.depth / n_params) + l * node.loss + a * (1 - node.accuracy)

    frontier = PriorityQueue(f=cost_fn, max_size=1000)
    frontier.append(Node(depth=0, neural_network=neural_network, loss=0, accuracy=0, parent=None))

    print('Training...\n')

    i = 0
    while len(frontier) > 0:
        i += 1

        node: Node = frontier.pop()
        print(f'\t{i}: loss = {node.loss:.4f}, accuracy = {node.accuracy:.4f}, depth = {node.depth}')

        if is_goal(node, goal, train_inputs, train_labels, batch_size, last_activation):
            solutions.append(node)
            if len(solutions) == n_solutions:
                break

        def optimize():
            # note: optimizing for a random batch
            batch_idx = np.random.randint(0, len(train_inputs) // batch_size)
            batch_inputs = train_inputs[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_labels = train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            
            # note: optimizing random portion of the neural network
            n_params_updated = 0
            for layer_idx in np.random.permutation(len(neural_network)):
                for param_idx in np.random.permutation(len(neural_network[layer_idx])):
                    for row_idx in np.random.permutation(len(neural_network[layer_idx][param_idx])):
                        for col_idx in np.random.permutation(len(neural_network[layer_idx][param_idx][row_idx])):
                            for action in actions:
                                next_neural_network = copy_neural_network(node.neural_network)
                                next_neural_network[layer_idx][param_idx][row_idx][col_idx] += action

                                next_loss, next_accuracy = evaluate(next_neural_network, batch_inputs, batch_labels, last_activation)
                                frontier.append(Node(depth=node.depth + 1, neural_network=next_neural_network, loss=next_loss, accuracy=next_accuracy, parent=node))

                            n_params_updated += 1
                            if n_params_updated == n_params_to_update:
                                return
        
        optimize()

    train_time = time.time() - start_time

    for i, node in enumerate(solutions):
        print(f'\nSolution {i + 1}:')
        print(f'Train: loss = {node.loss:.4f}, accuracy = {node.accuracy:.4f}, depth = {node.depth}')

        test_loss, test_accuracy = evaluate(node.neural_network, test_inputs, test_labels, last_activation)
        print(f'--> Test: loss = {test_loss:.4f}, accuracy = {test_accuracy:.4f}')

        # save_neural_network(node.neural_network, f'./finals/a*_{dataset}_{time.time()}.txt')

        updated_params_count = count_updated_parameters(initial_neural_network, node.neural_network)
        print(f'Updated {updated_params_count} parameters out of {count_parameters(neural_network)}')

    print(f'Train time, sec = {train_time:.2f}')

    if should_plot_accuracy_vs_depth:
        plot_accuracy_vs_depth(solutions[0], goal, dataset)

    return test_accuracy, test_loss, updated_params_count, train_time

class Node:
    def __init__(self, depth: int, neural_network: list[tuple[np.ndarray, np.ndarray]], loss: float, accuracy: float, parent = None):
        self.depth = depth
        self.neural_network = neural_network # note: represents current state of the search
        self.loss = loss
        self.accuracy = accuracy
        self.parent = parent

    def __lt__(self, other) -> bool:
        return self.accuracy < other.accuracy

    def __lr__(self, other) -> bool:
        return self.accuracy <= other.accuracy

    def __gt__(self, other) -> bool:
        return self.accuracy > other.accuracy

    def __ge__(self, other) -> bool:
        return self.accuracy >= other.accuracy
    
class PriorityQueue:
    def __init__(self, f: Callable, max_size: int):
        self.heap = []
        self.f = f
        self.max_size = max_size

    def append(self, item: Any):
        priority = self.f(item)
        if len(self.heap) <= self.max_size:
            heapq.heappush(self.heap, (priority, item))
        else:
            lowest_priority, _ = self.heap[0]
            if priority < lowest_priority:
                heapq.heapreplace(self.heap, (priority, item))

    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception('Cannot pop from empty queue.')

    def __len__(self):
        return len(self.heap)

def is_goal(node: Node, goal: float, inputs: np.ndarray, labels: np.ndarray, batch_size: int, last_activation: str):
    if node.accuracy < goal:
        return False

    n_batches = len(inputs) // batch_size
    total_accuracy = 0.0

    for batch_idx in np.random.permutation(n_batches):
        batch_inputs = inputs[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_labels = labels[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        _, batch_accuracy = evaluate(node.neural_network, batch_inputs, batch_labels, last_activation)
        total_accuracy += batch_accuracy

    total_accuracy /= n_batches
    print(f'\t🎯 evaluated accuracy = {total_accuracy:.4f}')

    return total_accuracy >= goal

def evaluate(
    neural_network: list[tuple[np.ndarray, np.ndarray]],
    inputs: np.ndarray,
    labels: np.ndarray,
    last_activation: str,
) -> np.float64:
    As, _ = forward(neural_network, inputs, last_activation)
    return cross_entropy_loss(labels, As[-1]), accuracy(labels, As[-1])

def plot_accuracy_vs_depth(solution: Node, goal: float, dataset: str) -> None:
    current = solution
    accuracy_per_depth: dict = {}

    while current.parent is not None:
        accuracy_per_depth[current.depth] = current.accuracy
        current = current.parent
    
    plt.figure(figsize=(10, 6))
    depths = sorted(accuracy_per_depth.keys())
    accuracies = [accuracy_per_depth[d] for d in depths]
    
    plt.plot(depths, accuracies, 'b-', marker='o')
    plt.title('Accuracy vs Search Depth')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Add horizontal line at goal accuracy
    plt.axhline(y=goal, color='r', linestyle='--', label=f'Goal Accuracy: {goal}')
    plt.legend()
    
    plt.savefig(f'./accuracy_vs_depth_{dataset}_{time.time()}.png')