from model import TDG_Mamba
import torch
import math
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, zero_one_loss
from utils import generate_data, create_model_filename
from tqdm import tqdm
import networkx as nx

def merge_graphs(graphs):
    """
    合并多个图为一个单一的图
    """
    merged_graph = nx.Graph()
    for graph in graphs:
        merged_graph = nx.compose(merged_graph, graph)
    return merged_graph

def train_link_prediction_model_with_gcn(model, optimizer, graph, X_train, y_train, x, node_to_index, args, input_ids, attention_mask):
    updated_edges = [(node_to_index[u], node_to_index[v]) for u, v in graph.edges()]
    edge_index = torch.tensor(updated_edges, dtype=torch.long).t().contiguous().to(args.device)

    X_train_global = [(node_to_index[u], node_to_index[v]) for u, v in X_train]
    
    edge_index_train = torch.tensor(X_train_global, dtype=torch.long).t().contiguous().to(args.device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float).to(args.device)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()

    #flops, params = profile(model, inputs=(x, input_ids, attention_mask, edge_index))
    # print(f"FLOPs: {flops}, Parameters: {params}")
  
    for epoch in range(args.epoch):  
        optimizer.zero_grad()
        out = model(x, input_ids, attention_mask, edge_index)
        pred = (out[edge_index_train[0]] * out[edge_index_train[1]]).sum(dim=1)
        loss = criterion(pred, y_train_tensor) 
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred_test = (out[edge_index_train[0]] * out[edge_index_train[1]]).sum(dim=1)
        pred_test = torch.sigmoid(pred_test) > 0.5
        f1 = f1_score(y_train_tensor.detach().cpu().numpy(), pred_test.detach().cpu().numpy())
        auc = roc_auc_score(y_train_tensor.detach().cpu().numpy(), pred_test.detach().cpu().numpy())
        ap = average_precision_score(y_train_tensor.detach().cpu().numpy(), pred_test.detach().cpu().numpy())
        hit_rate = zero_one_loss(y_train_tensor.detach().cpu().numpy(), pred_test.detach().cpu().numpy())
    return f1,auc,ap, hit_rate, out.detach(), loss.detach()


def train(model, optimizer, args, graphs, node_attributes):

    num_features = len(next(iter(node_attributes.values())))
    all_nodes = set()
    for graph in graphs.values():
        all_nodes.update(graph.nodes())
    node_to_index = {node: idx for idx, node in enumerate(all_nodes)}
    num_nodes = len(all_nodes)
    num_features = len(next(iter(node_attributes.values())))
    x_initial = torch.zeros((num_nodes, num_features), dtype=torch.float)
    for node, features in node_attributes.items():
        idx = node_to_index[node]
        x_initial[idx] = torch.tensor(features, dtype=torch.float)
    x_initial = x_initial.unsqueeze(-1).to(args.device)

    if args.Bert == True:
        input_ids, attention_mask = model.tokenizers(x_initial)
    else:
        input_ids, attention_mask = None, None

    train_len = math.ceil(len(graphs) * args.split)

    best_auc = 0
    best_model_state = None
    best_model = None
    print('Train.........')
    train_graphs = dict(list(graphs.items())[:train_len])
    print(len(train_graphs))



    for i in tqdm(range(args.iterations), desc="Iterations", unit="iteration"):#range(args.iterations):
        accuracy_per_timestamp = []
        auc_per_timestamp = []
        ap_per_timestamp = []
        hit_per_timestamp = []
        Loss = []

        for time_stamp, graph in train_graphs.items():
            X_train, y_train = generate_data(graph)

            x =  x_initial #+x_accumulated #+
            node_feature = x_initial.clone()
            node_feature[:, len(train_graphs):] = 0

            f1, auc, ap, hit, out, loss = train_link_prediction_model_with_gcn(model, optimizer, graph, X_train, y_train, x, node_to_index, args, input_ids, attention_mask)

            accuracy_per_timestamp.append(f1)
            auc_per_timestamp.append(auc)
            ap_per_timestamp.append(ap)
            hit_per_timestamp.append(hit)
            Loss.append(loss)
            print(f"Time {time_stamp}: F1 = {f1}, AUC = {auc}, AP = {ap}, Error = {hit}")



        average_accuracy = sum(accuracy_per_timestamp) / len(accuracy_per_timestamp)
        average_auc = sum(auc_per_timestamp) / len(auc_per_timestamp)
        average_ap = sum(ap_per_timestamp) / len(ap_per_timestamp)
        average_hit = sum(hit_per_timestamp) / len(hit_per_timestamp)
        average_loss = sum(Loss) / len(Loss)

        tqdm.write(f'Iterations {i}: Loss {average_loss.item():.4f}, F1 Score {average_accuracy:.4f}, AUC {average_auc:.4f}, AP {average_ap:.4f}, Hit Rate {average_hit:.4f}')
        
        # print(f"Average F1, AUC, AP and HIT@5 over all timestamps: {average_accuracy}, {average_auc}, {average_ap}, {average_hit}")


        if average_auc > best_auc:
            # print('Save Model')
            best_auc = average_auc
            best_model_state = model.state_dict()  # Saving the best model state
            best_model = model
            model_filename = create_model_filename(args)
            torch.save(best_model_state, './results/'+model_filename)  # Save the model to a file with all arguments in filename

    return best_model, node_to_index, x_initial, train_len, input_ids, attention_mask



def test(model, graph, X_test, y_test, x, node_to_index, args, input_ids, attention_mask):

    updated_edges = [(node_to_index[u], node_to_index[v]) for u, v in graph.edges()]
    edge_index = torch.tensor(updated_edges, dtype=torch.long).t().contiguous().to(args.device)


    X_test_global = [(node_to_index[u], node_to_index[v]) for u, v in X_test]
    edge_index_test = torch.tensor(X_test_global, dtype=torch.long).t().contiguous().to(args.device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float).to(args.device)
    

    model.eval()
    out = model(x, input_ids, attention_mask, edge_index)  
    pred_test = (out[edge_index_test[0]] * out[edge_index_test[1]]).sum(dim=1)
    pred_test = torch.sigmoid(pred_test) > 0.5
    f1 = f1_score(y_test_tensor.detach().cpu().numpy(), pred_test.detach().cpu().numpy())
    auc = roc_auc_score(y_test_tensor.detach().cpu().numpy(), pred_test.detach().cpu().numpy())
    ap = average_precision_score(y_test_tensor.detach().cpu().numpy(), pred_test.detach().cpu().numpy())
    hit_rate = zero_one_loss(y_test_tensor.detach().cpu().numpy(), pred_test.detach().cpu().numpy())
    return f1,auc,ap, hit_rate, out.detach()





    