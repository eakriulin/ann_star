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
    parser.add_argument('--dataset', help='dataset name; iris, wine or digits', default='iris')
    parser.add_argument('--arch', help='architecture type; no or single', default='no')
    parser.add_argument('--batch', help='batch size', default=100)
    parser.add_argument('--epochs', help='number of training epochs for backprop', default=1000)
    parser.add_argument('--goal', help='goal accuracy, e.g. 95', default=95)
    parser.add_argument('--lact', help='last activation; relu or sigmoid', default='relu')
    args = parser.parse_args()

    k = int(args.k)
    goal = int(args.goal) / 100
    batch_size = int(args.batch)
    n_epochs = int(args.epochs)
    learning_rate = 0.001
    initialization_method = 'he' if args.algo == 'backprop' else 'zeros'

    if k > 1:
        if args.algo == 'backprop':
            kfold.eval(k, args.dataset, args.arch, initialization_method, backprop.train, learning_rate=learning_rate, batch_size=batch_size, n_epochs=n_epochs, last_activation=args.lact)
        elif args.algo == 'astar':
            kfold.eval(k, args.dataset, args.arch, initialization_method, a_star.train, goal=goal, batch_size=batch_size, last_activation=args.lact, should_plot_accuracy_vs_depth=False)
    else:
        neural_network = nn.initialize(dataset=args.dataset, architecture_type=args.arch, method=initialization_method)
        train_inputs, train_labels, test_inputs, test_labels = data.get(dataset=args.dataset)

        if args.algo == 'backprop':
            backprop.train(neural_network, train_inputs, train_labels, test_inputs, test_labels, learning_rate=learning_rate, batch_size=batch_size, n_epochs=n_epochs, last_activation=args.lact, dataset=args.dataset)
        elif args.algo == 'astar':
            a_star.train(neural_network, train_inputs, train_labels, test_inputs, test_labels, goal=goal, batch_size=batch_size, dataset=args.dataset, last_activation=args.lact, should_plot_accuracy_vs_depth=True)

if __name__ == '__main__':
    main()