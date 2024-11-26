import random
import numpy as np
# random_seed = 2024
# random.seed(random_seed)


def hit_rate_at_k(y_true, pred_probs, k=5):
    sorted_indices = np.argsort(~pred_probs)
    hits = [true_index in sorted_indices[:k] for true_index, _ in enumerate(y_true) if y_true[true_index]]
    return sum(hits) / len(hits) if hits else 0



def generate_negative_samples(graph, num_samples):
    random.seed(2024)
    non_edges = set()
    nodes = list(graph.nodes())
    while len(non_edges) < num_samples:
        u, v = random.sample(nodes, 2)
        if not graph.has_edge(u, v) and not graph.has_edge(v, u):
            non_edges.add((u, v))
    return list(non_edges)




def generate_data(graph):

    positive_edges = [(u, v) for u, v in graph.edges()]
    
    num_positive_samples = len(positive_edges)
    negative_edges = generate_negative_samples(graph, num_positive_samples)

    labels_positive = [1] * len(positive_edges)
    labels_negative = [0] * len(negative_edges)

    X = positive_edges + negative_edges
    y = labels_positive + labels_negative
    return X, y


def create_model_filename(args):
    """
    Create a filename that encapsulates all the important arguments used for training.
    This helps in identifying the model configuration easily from the filename.
    """
    filename = f"model_data-{args.data}_attr-{args.attribute_type}_MPNN-{args.MPNN}.pth"
    return filename