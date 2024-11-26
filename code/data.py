import os
import pickle
from os.path import join

import networkx as nx
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import *
import pandas as pd

import random
import torch
from torch_geometric.data import Data
import numpy as np
from node2vec import Node2Vec

random_seed = 2024
random.seed(random_seed)

def data_load(args):
    if args.data == 'facebook':
        dir_path = f'../data/{args.data}'
        os.makedirs(dir_path, exist_ok=True)  
        path = f'{dir_path}/facebook-wall.txt'
        df = pd.read_table(path, sep = '\t', header = None)
        df.columns = ['source', 'target', 'time']
        # temporal_g = TemporalGraph(data = df, time_granularity = 'months')

        temporal_g = TemporalGraph(data = df, time_granularity = 'months')
        graphs = temporal_g.get_temporal_graphs(min_degree = 0)

    if args.data == 'game_of_thrones':
        with open(join('../data', 'game_of_thrones/gameofthrones_2017_graphs_dynamic.pkl'), 'rb') as f:
            graphs = pickle.load(f)

    if args.data == 'formula':
        with open(join('../data', 'formula/formula_2019_graphs_dynamic.pkl'), 'rb') as f:
            graphs = pickle.load(f)

    if args.data == 'enron':
        dir_path = f'../data/{args.data}'
        os.makedirs(dir_path, exist_ok=True)  
        path = f'{dir_path}/out.enron'
        df = pd.read_table(path, sep=' ', header=None)
        df.columns = ['source', 'target', 'weight', 'time']
        # temporal_g = TemporalGraph(data = df, time_granularity = 'months')
        temporal_g = TemporalGraph(data = df, time_granularity = 'months')
        graphs = temporal_g.get_temporal_graphs(min_degree = 0)


    graphs_to_remove = []
    for time_stamp, graph in graphs.items():
        if len(graph.edges()) <= 150 or len(graph.nodes()) <= 100:
            graphs_to_remove.append(time_stamp)
    for time_stamp in graphs_to_remove:
        del graphs[time_stamp]


    node_attributes = get_arribute_features(args, graphs)
    return graphs, node_attributes


class TemporalGraph():
    def __init__(self, data, time_granularity):
        '''

        :param data: DataFrame- source, target, time, weight columns
        :param time_granularity: 'day', 'week', 'month', 'year' or 'hour'
        '''
        data['day'] = data['time'].apply(lambda timestamp: (datetime.utcfromtimestamp(timestamp)).day)
        data['week'] = data['time'].apply(
            lambda timestamp: (datetime.utcfromtimestamp(timestamp)).isocalendar()[1])
        data['month'] = data['time'].apply(lambda timestamp: (datetime.utcfromtimestamp(timestamp)).month)
        data['year'] = data['time'].apply(lambda timestamp: (datetime.utcfromtimestamp(timestamp)).year)
        data['hour'] = data['time'].apply(lambda timestamp: (datetime.utcfromtimestamp(timestamp)).hour)
        if 'weight' not in data.columns:
            data['weight'] = 1
        self.data = data
        self.time_granularity = time_granularity
        self.time_columns, self.step = self._get_time_columns(time_granularity)
        self.static_graph = self.get_static_graph()

    def get_static_graph(self):
        g = nx.from_pandas_edgelist(self.data, source = 'source', target = 'target', edge_attr = ['weight'],
                                    create_using = nx.Graph())
        self.nodes = g.nodes()
        return g

    def filter_nodes(self, thresh = 5):
        nodes2filter = [node for node, degree in self.static_graph.degree() if degree < thresh]
        return nodes2filter

    def get_temporal_graphs(self, min_degree, mode = 'dynamic'):
        '''
        :param filter_nodes: int.  filter nodes with degree<min_degree in all time steps
        :param mode: if not 'dynamic', add all nodes to the current time step without edges
        :return: dictionary. key- time step, value- nx.Graph
        '''
        G = {}
        for t, time_group in self.data.groupby(self.time_columns):
            time_group = time_group.groupby(['source', 'target'])['weight'].sum().reset_index()
            g = nx.from_pandas_edgelist(time_group, source = 'source', target = 'target', edge_attr = ['weight'],
                                        create_using = nx.Graph())
            if mode != 'dynamic':
                g.add_nodes_from(self.nodes)
            g.remove_nodes_from(self.filter_nodes(min_degree))
            G[self.get_date(t)] = g
        self.graphs = G
        return G
    
    def filter_nodes_by_min_degree(self, edge_index, min_degree):
        num_nodes = max(edge_index.max().item() + 1, edge_index.shape[1])
        edge_count = torch.bincount(edge_index[0], minlength=num_nodes) + torch.bincount(edge_index[1], minlength=num_nodes)
        nodes_to_keep = torch.where(edge_count >= min_degree)[0]
        return nodes_to_keep
    
    def get_date(self, t):
        time_dict = dict(zip(self.time_columns, t))
        if self.time_granularity == 'hours':
            return datetime(year = time_dict['year'], month = time_dict['month'], day = time_dict['day'],
                            hour = time_dict['hour'])
        elif self.time_granularity == 'days':
            return datetime(year = time_dict['year'], month = time_dict['month'], day = time_dict['day'])
        elif self.time_granularity == 'months':
            return datetime(year = time_dict['year'], month = time_dict['month'], day = 1)
        elif self.time_granularity == 'weeks':
            date_year = datetime(year = time_dict['year'], month = 1, day = 1)
            return date_year + timedelta(days = float((time_dict['week'] - 1) * 7))
        elif self.time_granularity == 'years':
            return datetime(year = time_dict['year'], month = 1, day = 1)
        else:
            raise Exception("not valid time granularity")

    @staticmethod
    def _get_time_columns(time_granularity):
        if time_granularity == 'hours':
            group_time = ['year', 'month', 'day', 'hour']
            step = timedelta(hours = 1)
        elif time_granularity == 'days':
            group_time = ['year', 'month', 'day']
            step = timedelta(days = 1)
        elif time_granularity == 'weeks':
            group_time = ['year', 'week']
            step = timedelta(weeks = 1)
        elif time_granularity == 'months':
            group_time = ['year', 'month']
            step = relativedelta(months = 1)
        elif time_granularity == 'years':
            group_time = ['year']
            step = relativedelta(years = 1)
        else:
            raise Exception("not valid time granularity")
        return group_time, step
    


def get_arribute_features(args, graphs):
    time_stamps = list(graphs.keys())
    all_nodes = set()
    for graph in graphs.values():
        all_nodes.update(graph.nodes())
    node_attributes = {node: np.zeros(len(time_stamps)) for node in all_nodes}
    

    if args.attribute_type == 'A':
        for i, time_stamp in enumerate(time_stamps):
            graph = graphs[time_stamp]
            for node in graph.nodes():
                node_attributes[node][i] = 1

    if args.attribute_type == 'E':
        decay_rate = 0.9  
        for i, time_stamp in enumerate(time_stamps):
            weight = decay_rate ** (len(time_stamps) - i - 1)
            graph = graphs[time_stamp]
            for node in graph.nodes():
                node_attributes[node][i] = weight

    if args.attribute_type == 'G':
        for i, time_stamp in enumerate(time_stamps):
            graph = graphs[time_stamp]
            for node in graph.nodes():
                node_attributes[node][i] = len(graph.edges(node))

    if args.attribute_type == 'W':
        for i, time_stamp in enumerate(time_stamps):
            graph = graphs[time_stamp]
            for node in graph.nodes():
                node_attributes[node][i] = 1
        cumulative_sums = {node: np.cumsum([0] + node_attributes[node]) for node in node_attributes}
        window_size = args.window_size  
        for i, time_stamp in enumerate(time_stamps):
            graph = graphs[time_stamp]
            for node in graph.nodes():
                window_start = max(0, i - window_size + 1)
                if i + 1 < len(cumulative_sums[node]):
                    window_sum = cumulative_sums[node][i + 1] - cumulative_sums[node][window_start]
                    node_count = i + 1 - window_start 
                    node_attributes[node][i] = window_sum / node_count

    if args.attribute_type == 'R':
        filename = f'{args.data}_attr-{args.attribute_type}-node_attributes.pkl'

        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                node_attributes = pickle.load(f)
            return node_attributes
        for i, time_stamp in enumerate(time_stamps):
            print(time_stamp)
            graph = graphs[time_stamp]
            node2vec = Node2Vec(graph, dimensions=64, walk_length=args.walk_length, num_walks=1, workers=1)
            model = node2vec.fit(window=args.window, min_count=1, batch_words=4)
            
            for node in graph.nodes():
                if node in model.wv:
                    node_attributes[node][i] = np.mean(model.wv[node]) 
                else:
                    node_attributes[node][i] = 1  
        with open(f'{args.data}_attr-{args.attribute_type}-node_attributes_hyperparm.pkl', 'wb') as f:
            pickle.dump(node_attributes, f)



    if args.attribute_type == 'H':
        window_size = args.window_size
        time_stamps = list(graphs.keys())
        num_features_per_timestamp = 4  
        num_timestamps = len(time_stamps)
        node_feature_vectors = {node: np.zeros((num_timestamps, num_features_per_timestamp)) for node in all_nodes}
        for i, time_stamp in enumerate(time_stamps):
            graph = graphs[time_stamp]
            decay_rate = 0.9
            weight = decay_rate ** (num_timestamps - i - 1)
            
            for node in graph.nodes():
                presence_feature = 1
                weighted_presence_feature = weight
                cumulative_activity_feature = len(graph.edges(node))
                window_start = max(0, i - window_size + 1)
                window_presence_feature = sum(node_feature_vectors[node][window_start:i+1, 0]) / window_size
                node_attributes[node][i] = (torch.sum(torch.Tensor([
                    presence_feature,
                    weighted_presence_feature,
                    cumulative_activity_feature,
                    window_presence_feature
                ]),dim=-1))/4
    return node_attributes