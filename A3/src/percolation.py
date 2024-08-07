import numpy as np
import networkx as nx
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

class Graph:
    def __init__(self, num_graphs_per_setting, n, radius, p):
        self.num_graphs_per_setting=num_graphs_per_setting
        self.n=n
        self.radius=radius
        self.p=p
        self.graphs = []
        self.graphs_radi=[]
        self.graphs_p=[]
        self.graphs_n=[]
        self.threshold=0.5
        self.generate_graph()

    def _generate_hybrid_graph(self,n,radius,p):
        graphs = []
        graphs_radi=[]
        graphs_p=[]
        graphs_n=[]
        for _ in range(self.num_graphs_per_setting):
            G = nx.random_geometric_graph(n, radius)
            for i in range(n):
                for j in range(i + 1, n):
                    if G.has_edge(i, j) and np.random.rand() > p:
                        G.remove_edge(i, j)
            graphs.append(G)
            graphs_radi.append(radius)
            graphs_p.append(p)
            graphs_n.append(n)
        return graphs,graphs_radi,graphs_p,graphs_n
        
    
    def generate_graph(self):
        for n in self.n:
            for p in self.p:
                for radius in self.radius:
                    one_set_of_graph,one_set_of_radi,one_set_of_p,one_set_of_n=self._generate_hybrid_graph(n, radius, p) 
                    self.graphs += one_set_of_graph
                    self.graphs_radi += one_set_of_radi
                    self.graphs_p += one_set_of_p
                    self.graphs_n += one_set_of_n
                    # one_set_of_result= compute_percolation_status(one_set_of_graph)
                    # result.append(np.sum(one_set_of_result)/num_graphs)
        
        # self.graphs=graphs
        # self.graph_radi=graphs_radi
        # self.graph_p=graphs_p
        

    def _compute_percolation_status(self):
        percolation_status = []
        for G in self.graphs:
            largest_cc = max(nx.connected_components(G), key=len)
            fraction_largest_cc = len(largest_cc) / len(G)
            percolation_status.append(int(fraction_largest_cc >= self.threshold))
        return np.array(percolation_status).reshape(-1, 1)

    def _extract_features(self,modify_feature=False):
        avg_degree = []
        for G in self.graphs:
            avg_degree_one_set = np.mean(list(dict(G.degree()).values()))
            avg_degree.append(avg_degree_one_set)
        
        avg_degree=np.array(avg_degree)[:,np.newaxis]

        avg_n=np.array(self.graphs_n)[:,np.newaxis]
        avg_radi=np.array(self.graphs_radi)[:,np.newaxis]
        avg_p=np.array(self.graphs_p)[:,np.newaxis]

        if modify_feature:
            return avg_degree
        else:
            feature=np.concatenate((avg_degree,avg_n,avg_radi,avg_p),axis=1)

        return feature
    
    def dataset(self, modify_feature=False):
        X=self._extract_features(modify_feature)
        y=self._compute_percolation_status()
        return np.array(X),np.array(y).flatten()
    
    

