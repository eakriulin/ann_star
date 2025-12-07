# ANNs Optimization with A*

This repository introduces an A*-based optimization algorithm for feedforward neural networks. The study examines whether A* can be repurposed for ANN training and whether it is capable of escaping local minima that trap gradient-based methods. First, neural network optimization was redefined as a search problem. Then, an A*-based optimization algorithm was developed. For comparison, a gradient-based backpropagation algorithm was implemented. Both algorithms were tested on three classification tasks using the Iris, Wine, and Digits datasets. Results demonstrate that A* can successfully train feedforward neural networks with performance comparable to backpropagation.

## How to Run

To run the script, use the `main.py` file with various command-line arguments.

### Examples

Run A* algorithm on Iris dataset with k-fold cross-validation:

```zsh
python main.py --algo=astar --k=5 --stratified=0 --dataset=iris --arch=no --batch=60 --goal=97 --lact=relu
```

Run backpropagation algorithm on Digits dataset without k-fold:

```zsh
python main.py --algo=backprop --stratified=1 --dataset=digits --arch=single --epochs=50000 --batch=1000 --lact=sigmoid
```