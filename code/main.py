import torch

import time
from data import data_load
from train import train, test
from model import TDG_Mamba
from utils import generate_data
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Setup the training and dataset configuration.")
    parser.add_argument('--data', type=str, default='facebook', help='Name of the dataset to be used.')
    parser.add_argument('--attribute_type', type=str, default='R', choices=['A', 'E', 'G', 'W', 'H'], help='Type of attribute features to be used.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learned rate of model.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight_decay of model.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate of model.')
    parser.add_argument('--Bert', action='store_true', help="Enable Bert if set")
    parser.add_argument('--window_size', type=int, default=1, help='Window size of attribute features to be used.')
    parser.add_argument('--MPNN', type=str, default='GAT', help='Type of internal MPNN to use within DirGNNConv.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of training iterations for the external loop.')
    parser.add_argument('--epoch', type=int, default=50, help='Number of training epochs for the internal loop.')
    parser.add_argument('--split', type=float, default=0.7, help='Ratio for splitting the dataset into train and test sets.')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels; set equal to feature dimensions unless customized.')
    parser.add_argument('--d_model', type=int, default=1, help='Dimensionality of the model.')
    parser.add_argument('--d_state', type=int, default=16, help='Dimensionality of the state space in Mamba.')
    parser.add_argument('--d_conv', type=int, default=4, help='Dimensionality of the convolution in Mamba.')
    parser.add_argument('--expand', type=int, default=1, help='Expansion factor in Mamba.')
    parser.add_argument('--walk_length', type=int, default=30)
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to run the model on: "cuda", "cpu", or "auto" to automatically select based on availability.')
    args = parser.parse_args()
    if args.device == 'auto':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    return args


if __name__ == '__main__':
    start_time = time.time()

    args = get_args()
    print(f"Running with configuration: {args}")
    print(f"Using device: {args.device}")
    graphs, node_attributes = data_load(args)


    num_features = len(next(iter(node_attributes.values())))
    model = TDG_Mamba(args, num_features).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model, node_to_index, x_initial, train_len, input_ids, attention_mask  = train(model, optimizer, args, graphs, node_attributes)


    print('TEST.........')
    accuracy_per_timestamp = []
    auc_per_timestamp = []
    ap_per_timestamp = []
    hit_per_timestamp = []

    test_graphs = dict(list(graphs.items())[train_len:])
    print(len(test_graphs))
    for time_stamp, graph in test_graphs.items():
        X_test, y_test = generate_data(graph)
        x = x_initial #x_initial #+ x_accumulated
        f1, auc, ap, hit, out = test(model, graph, X_test, y_test, x, node_to_index, args, input_ids, attention_mask)
        x_accumulated = out.unsqueeze(-1)


        accuracy_per_timestamp.append(f1)
        auc_per_timestamp.append(auc)
        ap_per_timestamp.append(ap)
        hit_per_timestamp.append(hit)
        print(f"Time {time_stamp}: F1 = {f1}, AUC = {auc}, AP = {ap}, Error = {hit}")

    average_accuracy = sum(accuracy_per_timestamp) / len(accuracy_per_timestamp)
    average_auc = sum(auc_per_timestamp) / len(auc_per_timestamp)
    average_ap = sum(ap_per_timestamp) / len(ap_per_timestamp)
    average_hit = sum(hit_per_timestamp) / len(hit_per_timestamp)
    print(f"Average F1, AUC, AP and HIT@5 over all timestamps: {average_accuracy}, {average_auc}, {average_ap}, {average_hit}")

    end_time = time.time()
    print("Execution time: {:.2f} seconds".format(end_time - start_time))
