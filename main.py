import argparse
import data
import nn
import backprop
import a_star
import kfold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', help='training algorithm; astar or backprop', default='astar')
    parser.add_argument('--k', help='number of folds for k-fold cross validation', default=0)
    parser.add_argument('--stratified', help='should make stratified folds, 0 or 1', default=0)
    parser.add_argument('--dataset', help='dataset name; iris, wine, digits, stripes2 or stripes3', default='iris')
    parser.add_argument('--l', help='architecture type; 2, 3 or 4', default='2')
    parser.add_argument('--batch', help='batch size', default=100)
    parser.add_argument('--epochs', help='number of training epochs for backprop', default=1000)
    parser.add_argument('--goal', help='goal accuracy, e.g. 95', default=95)
    parser.add_argument('--lact', help='last activation; relu or sigmoid', default='relu')
    parser.add_argument('--mtype', help='model type; mlp or moe', choices=list(nn.builders), default='moe')
    args = parser.parse_args()

    k = int(args.k)
    should_stratify = bool(int(args.stratified))
    goal = int(args.goal) / 100
    batch_size = int(args.batch)
    n_epochs = int(args.epochs)
    learning_rate = 0.001
    initialization_method = 'he' if args.algo == 'backprop' else 'zeros'

    def initialize():
        return nn.initialize(dataset=args.dataset, architecture_type=args.l, method=initialization_method, last_activation=args.lact, model_type=args.mtype)

    if args.algo == 'backprop':
        train_fn = backprop.train
        train_kwargs = dict(learning_rate=learning_rate, batch_size=batch_size, n_epochs=n_epochs)
    elif args.algo == 'astar':
        train_fn = a_star.train
        train_kwargs = dict(goal=goal, batch_size=batch_size, should_plot_accuracy_vs_depth=k <= 1)
    else:
        raise ValueError(f'Unknown algo: {args.algo}')

    if k > 1:
        kfold.eval(k, should_stratify, args.dataset, args.l, args.algo, train_fn, initialize, **train_kwargs)
    else:
        model = initialize()
        train_inputs, train_labels, test_inputs, test_labels = data.get(dataset=args.dataset)
        train_fn(model, train_inputs, train_labels, test_inputs, test_labels, dataset=args.dataset, **train_kwargs)

if __name__ == '__main__':
    main()
