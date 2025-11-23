import argparse
import time
import data
import nn
import backprop
import a_star

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='backprop or a_star')
    parser.add_argument('--dataset', help='iris or mnist')
    parser.add_argument('--epochs', help='number of training epochs for backprop', default=1000)
    parser.add_argument('--batch', help='batch size', default=100)
    args = parser.parse_args()

    train_inputs, train_labels, test_inputs, test_labels = data.get(dataset=args.dataset)

    if args.mode == 'backprop':
        neural_network = nn.initialize(dataset=args.dataset, method='he')
        start_time = time.time()
        backprop.train(neural_network, train_inputs, train_labels, test_inputs, test_labels, learning_rate=0.001, batch_size=int(args.batch), n_epochs=int(args.epochs), dataset=args.dataset)

    if args.mode == 'a_star':
        neural_network = nn.initialize(dataset=args.dataset, method='zeros')
        start_time = time.time()
        a_star.train(neural_network, train_inputs, train_labels, test_inputs, test_labels, batch_size=int(args.batch), dataset=args.dataset)

    print(f'---{time.time() - start_time} seconds---')

if __name__ == '__main__':
    main()