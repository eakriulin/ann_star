import time
import heapq
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable, Iterator
from models import Model, TargetPropagationMLP
from shared import evaluate, mse_loss, count_parameters, count_updated_parameters

def train(
    model: Model,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    test_inputs: np.ndarray,
    test_labels: np.ndarray,
    goal: float,
    batch_size: int,
    dataset: str,
    should_plot: bool,
    max_time: float | None = None,
) -> tuple[float, float, int, float]:
    if isinstance(model, TargetPropagationMLP):
        return train_divide_and_conquer(model, train_inputs, train_labels, test_inputs, test_labels, goal, batch_size, dataset, should_plot, max_time)

    return train_flat(model, train_inputs, train_labels, test_inputs, test_labels, goal, batch_size, dataset, should_plot, max_time)

def train_flat(
    model: Model,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    test_inputs: np.ndarray,
    test_labels: np.ndarray,
    goal: float,
    batch_size: int,
    dataset: str,
    should_plot: bool,
    max_time: float | None = None,
) -> tuple[float, float, int, float]:
    start_time = time.time()

    initial_model = model.copy()
    
    n_params = count_parameters(model)
    n_params_to_update = 200
    n_solutions = 1

    solutions: list[Node] = []

    def cost_fn(node: Node):
        """
        Computes the cost associated with a given node based on its depth, loss, and accuracy.
        - Depth-based cost: Penalizes deeper nodes relative to the total number of parameters.
        - Loss-based cost: Penalizes nodes with higher loss values.
        - Accuracy-based cost: Penalizes nodes with lower accuracy.
        """

        d = 0.5
        l = 0.05
        a = 0.45

        return d * (node.depth / n_params) + l * node.loss + a * (1 - node.accuracy)

    frontier = PriorityQueue(f=cost_fn, max_size=1000)
    frontier.append(Node(model=model, depth=0, loss=0, accuracy=0, parent=None))

    print('Training...\n')

    best: Node | None = None

    i = 0
    while len(frontier) > 0:
        i += 1

        node: Node = frontier.pop()
        print(f'\t{i}: loss = {node.loss:.4f}, accuracy = {node.accuracy:.4f}, depth = {node.depth}')

        if node.depth > 0 and (best is None or node.accuracy > best.accuracy):
            best = node

        if max_time is not None and time.time() - start_time > max_time:
            print(f'\t⏱ time limit of {max_time:.0f}s reached, returning the most accurate node')
            solutions.append(best or node)
            break

        if is_goal(node, goal, train_inputs, train_labels, batch_size):
            solutions.append(node)
            if len(solutions) == n_solutions:
                break

        batch_idx = np.random.randint(0, len(train_inputs) // batch_size)
        batch_inputs = train_inputs[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_labels = train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        for next_node in expand(node, all_parameters, batch_inputs, batch_labels, n_params_to_update):
            frontier.append(next_node)

    train_time = time.time() - start_time

    for i, node in enumerate(solutions):
        print(f'\nSolution {i + 1}:')
        print(f'Train: loss = {node.loss:.4f}, accuracy = {node.accuracy:.4f}, depth = {node.depth}')

        test_loss, test_accuracy = evaluate(node.model, test_inputs, test_labels)
        print(f'--> Test: loss = {test_loss:.4f}, accuracy = {test_accuracy:.4f}')

        updated_params_count = count_updated_parameters(initial_model, node.model)
        print(f'Updated {updated_params_count} parameters out of {n_params}')

    print(f'Train time, sec = {train_time:.2f}')

    if should_plot:
        plot_accuracy_vs_depth(solutions[0], goal, dataset, model.model_type)
        solutions[0].model.plot(dataset)

    return test_accuracy, test_loss, updated_params_count, train_time

def train_divide_and_conquer(
    model: TargetPropagationMLP,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    test_inputs: np.ndarray,
    test_labels: np.ndarray,
    goal: float,
    batch_size: int,
    dataset: str,
    should_plot: bool,
    max_time: float | None = None,
) -> tuple[float, float, int, float]:
    start_time = time.time()

    initial_model = model.copy()
    n_params = count_parameters(model)
    n_expansions = 10

    current = Node(model=model, depth=0, loss=0, accuracy=0, parent=None)
    best: Node | None = None
    solution = None

    print('Training (divide and conquer)...\n')

    i = 0
    while solution is None:
        i += 1

        batch_idx = np.random.randint(0, len(train_inputs) // batch_size)
        batch_inputs = train_inputs[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_labels = train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        _, cache = current.model.forward(batch_inputs)
        A_hidden_targets = current.model.hidden_targets(cache, batch_labels)
        A_hidden_at_round_start = cache.A_hidden

        hidden_neurons, output_mlp = current.model.divide()

        for u, hidden_neuron in enumerate(hidden_neurons):
            def neuron_loss_fn(candidate_hidden_neuron, u=u):
                return np.mean((candidate_hidden_neuron.forward(batch_inputs) - A_hidden_targets[:, u:u + 1]) ** 2)

            hidden_neurons[u] = local_search(hidden_neuron, neuron_loss_fn, n_expansions)

        def output_loss_fn(candidate_output_mlp):
            return mse_loss(batch_labels, candidate_output_mlp.forward(A_hidden_at_round_start)[0])

        output_mlp = local_search(output_mlp, output_loss_fn, n_expansions)

        next_model = TargetPropagationMLP.merge(hidden_neurons, output_mlp)
        next_loss, next_accuracy = evaluate(next_model, batch_inputs, batch_labels)
        current = Node(model=next_model, depth=current.depth + 1, loss=next_loss, accuracy=next_accuracy, parent=current)
        print(f'\t{i}: loss = {current.loss:.4f}, accuracy = {current.accuracy:.4f}, depth = {current.depth}')

        if best is None or current.accuracy > best.accuracy:
            best = current

        if is_goal(current, goal, train_inputs, train_labels, batch_size):
            solution = current
        
        if max_time is not None and time.time() - start_time > max_time:
            print(f'\t⏱ time limit of {max_time:.0f}s reached, returning the most accurate network')
            solution = best

    train_time = time.time() - start_time

    print(f'\nSolution:')
    print(f'Train: loss = {solution.loss:.4f}, accuracy = {solution.accuracy:.4f}, depth = {solution.depth}')

    test_loss, test_accuracy = evaluate(solution.model, test_inputs, test_labels)
    print(f'--> Test: loss = {test_loss:.4f}, accuracy = {test_accuracy:.4f}')

    updated_params_count = count_updated_parameters(initial_model, solution.model)
    print(f'Updated {updated_params_count} parameters out of {n_params}')

    print(f'Train time, sec = {train_time:.2f}')

    if should_plot:
        plot_accuracy_vs_depth(solution, goal, dataset, model.model_type)
        solution.model.plot(dataset)

    return test_accuracy, test_loss, updated_params_count, train_time

def all_parameters(model: Model) -> list:
    return model.parameters()

def expand(node: 'Node', select_parameters: Callable[[Model], list], batch_inputs: np.ndarray, batch_labels: np.ndarray, n_params_to_update: int) -> Iterator['Node']:
    actions = [0.001, -0.001]

    for n_params_updated, (*tensor_indices, row_idx, col_idx) in enumerate(random_parameter_indices(select_parameters(node.model)), start=1):
        for action in actions:
            next_model = node.model.copy()

            tensor = select_parameters(next_model)
            for idx in tensor_indices:
                tensor = tensor[idx]

            tensor[row_idx, col_idx] += action

            next_loss, next_accuracy = evaluate(next_model, batch_inputs, batch_labels)
            yield Node(model=next_model, depth=node.depth + 1, loss=next_loss, accuracy=next_accuracy, parent=node)

        if n_params_updated == n_params_to_update:
            return

def local_search(submodel: Any, loss_fn: Callable, n_expansions: int) -> Any:
    actions = [0.001, -0.001]

    frontier = PriorityQueue(f=lambda node: node.loss, max_size=10000)
    best = Node(model=submodel, depth=0, loss=loss_fn(submodel), accuracy=0)
    frontier.append(best)

    for _ in range(n_expansions):
        node: Node = frontier.pop()

        for *tensor_indices, row_idx, col_idx in random_parameter_indices(node.model.parameters()):
            for action in actions:
                next_submodel = node.model.copy()

                tensor = next_submodel.parameters()
                for idx in tensor_indices:
                    tensor = tensor[idx]

                tensor[row_idx, col_idx] += action

                next_node = Node(model=next_submodel, depth=node.depth + 1, loss=loss_fn(next_submodel), accuracy=0, parent=node)
                frontier.append(next_node)

                if next_node.loss < best.loss:
                    best = next_node

    return best.model

class Node:
    def __init__(self, model: Model, depth: int, loss: float, accuracy: float, parent = None):
        self.model = model # note: represents current state of the search
        self.depth = depth
        self.loss = loss
        self.accuracy = accuracy
        self.parent = parent
    
    def __lt__(self, other) -> bool:
        return self.accuracy < other.accuracy

    def __le__(self, other) -> bool:
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

def random_parameter_indices(parameters, indices: tuple = ()):
    if isinstance(parameters, np.ndarray):
        for row_idx in np.random.permutation(parameters.shape[0]):
            for col_idx in np.random.permutation(parameters.shape[1]):
                yield (*indices, row_idx, col_idx)
    else:
        for idx in np.random.permutation(len(parameters)):
            yield from random_parameter_indices(parameters[idx], (*indices, idx))

def is_goal(node: Node, goal: float, inputs: np.ndarray, labels: np.ndarray, batch_size: int):
    if node.accuracy < goal:
        return False

    n_batches = len(inputs) // batch_size
    total_accuracy = 0.0

    for batch_idx in np.random.permutation(n_batches):
        batch_inputs = inputs[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_labels = labels[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        _, batch_accuracy = evaluate(node.model, batch_inputs, batch_labels)
        total_accuracy += batch_accuracy

    total_accuracy /= n_batches
    print(f'\t🎯 evaluated accuracy = {total_accuracy:.4f}')

    return total_accuracy >= goal

def plot_accuracy_vs_depth(solution: Node, goal: float, dataset: str, model_type: str) -> None:
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
    
    plt.savefig(f'./accuracy_vs_depth_{model_type}_{dataset}_{time.time()}.png')
