import time
import heapq
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable
from models import Model
from shared import evaluate, count_parameters, count_updated_parameters

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
) -> tuple[float, float, int, float]:
    start_time = time.time()

    initial_model = model.copy()
    
    n_params = count_parameters(model)
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
        l = 0.05
        a = 0.45

        return d * (node.depth / n_params) + l * node.loss + a * (1 - node.accuracy)

    frontier = GreedyQueue(f=cost_fn, max_size=100000)
    frontier.append(Node(model=model, depth=0, loss=0, accuracy=0, parent=None))

    print('Training...\n')

    i = 0
    while len(frontier) > 0:
        i += 1

        node: Node = frontier.pop()
        print(f'\t{i}: loss = {node.loss:.4f}, accuracy = {node.accuracy:.4f}, depth = {node.depth}')

        if is_goal(node, goal, train_inputs, train_labels, batch_size):
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
            for indices in random_parameter_indices(node.model.parameters()):
                *tensor_indices, row_idx, col_idx = indices

                for action in actions:
                    next_model = node.model.copy()

                    tensor = next_model.parameters()
                    for idx in tensor_indices:
                        tensor = tensor[idx]

                    tensor[row_idx, col_idx] += action

                    next_loss, next_accuracy = evaluate(next_model, batch_inputs, batch_labels)
                    frontier.append(Node(model=next_model, depth=node.depth + 1, loss=next_loss, accuracy=next_accuracy, parent=node))

                n_params_updated += 1
                if n_params_updated == n_params_to_update:
                    return
        
        optimize()

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
        self.f = f
        self.max_size = max_size
        self.items: dict[int, Any] = {}
        self.min_heap: list[tuple[float, int]] = []
        self.max_heap: list[tuple[float, int]] = [] # note: stores -priority
        self.next_id = 0

    def append(self, item: Any):
        priority = self.f(item)

        if len(self.items) >= self.max_size:
            self._drop_dead(self.max_heap)
            worst_priority = -self.max_heap[0][0]
            if priority >= worst_priority:
                return

            _, worst_id = heapq.heappop(self.max_heap)
            del self.items[worst_id]

        item_id = self.next_id
        self.next_id += 1

        self.items[item_id] = item
        heapq.heappush(self.min_heap, (priority, item_id))
        heapq.heappush(self.max_heap, (-priority, item_id))

        if len(self.min_heap) > 2 * self.max_size or len(self.max_heap) > 2 * self.max_size:
            self._compact()

    def pop(self):
        self._drop_dead(self.min_heap)
        if not self.min_heap:
            raise Exception('Cannot pop from empty queue.')

        _, item_id = heapq.heappop(self.min_heap)
        return self.items.pop(item_id)

    def _drop_dead(self, heap: list):
        while heap and heap[0][1] not in self.items:
            heapq.heappop(heap)

    def _compact(self):
        self.min_heap = [entry for entry in self.min_heap if entry[1] in self.items]
        self.max_heap = [entry for entry in self.max_heap if entry[1] in self.items]
        heapq.heapify(self.min_heap)
        heapq.heapify(self.max_heap)

    def __len__(self):
        return len(self.items)

class GreedyQueue:
    """
    Greedy best-first search with backtracking.
    - Of the children appended since the last pop, only the best n_best_children are kept.
    - On pop(), they join the frontier (a bounded PriorityQueue) and the lowest-cost node overall is returned.
    - So the search usually follows the best child, but switches back to an older node
      (e.g. shallower, lower cost) when no new child beats it.
    """

    def __init__(self, f: Callable, max_size: int, n_best_children: int = 1):
        self.frontier = PriorityQueue(f=f, max_size=max_size)
        self.children = PriorityQueue(f=f, max_size=n_best_children)

    def append(self, item: Any):
        self.children.append(item)

    def pop(self):
        while len(self.children) > 0:
            self.frontier.append(self.children.pop())

        return self.frontier.pop()

    def __len__(self):
        return len(self.frontier) + len(self.children)

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
