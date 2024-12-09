from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import (
    TSNE,
)
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from time import time
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.sparse import csr_matrix
import plotly.express as px
from sklearn.compose import ColumnTransformer

import os

import faiss

class data:
    def __init__(self, path, graph_type='whole'):
        self.datapath = path
        self.data = self.load_data(500)
        self.data, self.labels = self.load_labels()
        self.class_labels = {'class': {'normal':0, 'anomaly':1}}
        #self.similarity_matrix = None
        if graph_type == 'stratified':
            self.stratified_data, self.stratified_labels = self.stratified_data(.95, 25)
            #print(self.stratified_data)
            #print(self.stratified_labels)
        #self.stratified_graph = None
        self.graph = None

    def stratified_data(self, sample_size, splits):
        stratified_split = StratifiedShuffleSplit(n_splits=splits, test_size=sample_size)

        indices =  stratified_split.split(self.data, self.labels)     

        strat_data = []
        strat_labels = []

        for idx, (train, test) in enumerate(indices):
            
            strat_data.append(self.data.iloc[train])
            strat_labels.append(self.labels.iloc[train])
    
            strat_data[idx] = strat_data[idx].reset_index(drop=True)
            strat_labels[idx] = strat_labels[idx].reset_index(drop=True)

        #print(len(strat_data))

        return strat_data, strat_labels

    def scale_data(self, scaling):
        '''
        cols = self.data.loc[:, ~self.data.columns.isin(['flag',
                                                                     'land', 'wrong_fragment', 'urgent',
                                                                     'num_failed_logins', 'logged_in',
                                                                     'root_shell', 'su_attempted', 'num_shells',
                                                                     'num_access_files', 'num_outbound_cmds',
                                                                     'is_host_login', 'is_guest_login', 'serror_rate',                                                                     'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',                                                                  'same_srv_rate', 'diff_srv_rate',
                                                                     'srv_diff_host_rate', 'dst_host_same_srv_rate',
                                                                     'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',                                                              'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',                                                                'dst_host_srv_serror_rate', 'dst_host_rerror_rate',                                                                   'dst_host_srv_rerror_rate', 'protocol_type ', 'service ' ])].columns
        '''
        cols = self.data.columns
        cols = np.asarray(cols)
        if scaling == 'standard':
            ct = ColumnTransformer([('normalize', StandardScaler(), cols)],
                                    remainder='passthrough' 
                                  )
            
            transformed_cols = ct.fit_transform(self.data)
            self.data = pd.DataFrame(transformed_cols, columns = self.data.columns)

        elif scaling == 'min_max':
            ct = ColumnTransformer([('normalize', MinMaxScaler(), cols)],
                                    remainder='passthrough'  
                                  ) 

            transformed_cols = ct.fit_transform(self.data)
            self.data = pd.DataFrame(transformed_cols, columns = self.data.columns)

        elif scaling == 'robust':
            ct = ColumnTransformer([('scaler', RobustScaler(), cols)],
                                    remainder='passthrough'
                                  )
            transformed_cols = ct.fit_transform(self.data)
            self.data = pd.DataFrame(transformed_cols, columns = self.data.columns)
        else:
            print("Scaling arg not supported")
        
    def encode_categorical(self, column_name, target_set):
        label_encoder = LabelEncoder()

        if target_set == 'data': 
            label_encoder = label_encoder.fit(self.data[column_name])
            self.data[column_name] = label_encoder.transform(self.data[column_name])
        elif target_set == 'labels':
            #print(self.labels)
            self.labels = self.labels.replace(self.class_labels)

    def load_data(self, sample_size=None):
        data = pd.read_csv(self.datapath)
        if sample_size != None:
            return data.sample(sample_size)
        else:
            return data

    def load_labels(self):
        labels = pd.DataFrame(self.data['class'], columns=['class'])
        data = self.data.loc[:, self.data.columns != 'class']
        return self.reset_indices(data, labels)

    def reset_indices(self, data, labels):
        data = data.reset_index(drop=True)
        labels = labels.reset_index(drop=True)
        return data, labels

    def generate_graphs(self, num_neighbors, rep=None, mode='distance', metric='euclidean', data_type='whole'):
        '''
        Generate the k-NN graph using FAISS
        
        Parameters:
            num_neighbors (int): The number of neighbors for each point.
            mode (str): 'distance' for distance values, 'connectivity' for binary connectivity.
            metric (str): Metric to use for distance ('euclidean' is assumed here).
        '''
        if data_type == 'whole':
        # Ensure the data is in dense format (FAISS requires dense arrays)
            data_matrix = np.array(self.data, dtype=np.float32)
            x_len = self.data.shape[0]
            y_len = self.data.shape[0]
        elif data_type == 'stratified':
            #print(self.stratified_data)
            data_matrix = np.array(self.stratified_data[rep], dtype=np.float32)
            x_len = self.stratified_data[rep].shape[0]
            y_len = self.stratified_data[rep].shape[0]

        # Initialize FAISS index for L2 distance (Euclidean)
        if metric == 'euclidean':
            index = faiss.IndexFlatL2(data_matrix.shape[1])
        else:
            raise ValueError("Currently, only 'euclidean' metric is supported.")

        # Add the data to the FAISS index
        index.add(data_matrix)

        # Perform the k-NN search to get the indices of neighbors and distances
        distances, indices = index.search(data_matrix, num_neighbors)  # distances and indices of neighbors

        # Construct the graph
        # Create an empty sparse matrix to hold the graph data
        if mode == 'distance':
            # Use the distances to create the graph (edges weighted by distance)
            graph_data = np.zeros((x_len, y_len))
            for i in range(len(indices)):
                for j in range(num_neighbors):
                    graph_data[i, indices[i, j]] = distances[i, j]
        elif mode == 'connectivity':
            # Use binary connectivity (1 if neighbor, 0 otherwise)
            graph_data = np.zeros((x_len, y_len))
            for i in range(len(indices)):
                for j in range(num_neighbors):
                    graph_data[i, indices[i, j]] = 1
        else:
            raise ValueError("Invalid mode. Use 'distance' or 'connectivity'.")

        # Convert the graph data to a sparse matrix
        graph = csr_matrix(graph_data)
       
        #mm_file = './mmap_file'
        #graph = np.memmap(mm_file + 'knn_graph', dtype='float32', mode='w+', shape=graph.shape)
        self.graph = graph
        #return graph

    '''
    def generate_graphs(self, num_neighbors, mode='distance', metric='euclidean'):
        nn = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree', metric=metric, p=2, n_jobs=-1)
        nn.fit(self.data)
        self.graph = nn.kneighbors_graph(self.data, mode=mode)
        #self.graph = kneighbors_graph(self.data, n_neighbors=num_neighbors, algorithm='ball_tree', mode=mode, metric=metric, p=2, include_self=True, n_jobs=-1)
    '''

class visualizer():
    def __init__(self, labels, dims):

        if isinstance(labels, np.ndarray):
            self.labels = pd.DataFrame(labels, columns=['class'])
        elif isinstance(labels, pd.DataFrame):
            self.labels = labels

        self.dims = dims

    def get_embeddings(self, num_components, embedding_subset = None):
        embeddings = {
            #"Truncated SVD embedding": TruncatedSVD(n_components=num_components),
            #"Standard LLE embedding": LocallyLinearEmbedding(
            #    n_neighbors=n_neighbors, n_components=num_components, method="standard", 
            #    eigen_solver='dense', n_jobs=-1
            #),
            #"Random Trees embedding": make_pipeline(
            #    RandomTreesEmbedding(n_estimators=200, max_depth=5, random_state=0, n_jobs=-1),
            #    TruncatedSVD(n_components=num_components),
            #),
            #"t-SNE embedding": TSNE(
            #    n_components=num_components,
            #    max_iter=500,
            #    n_iter_without_progress=150,
            #    n_jobs=-1,
            #    init='random',
            #    random_state=0,
            #),
            "PCA": PCA(n_components=num_components),
        }
        if embedding_subset == None:
            return embeddings
        else:
            out_dict = {}
            for key, value in enumerate(embeddings):
                out_dict[key] = value
            return out_dict

    def downsize_data(self, data):
        embeddings = self.get_embeddings(self.dims)

        projections, timing = {}, {}
        for name, transformer in embeddings.items():
            print(f"Computing {name}...")
            start_time = time()
            projections[name] = transformer.fit_transform(data, self.labels)
            timing[name] = time() - start_time

        return projections, timing 

    def lower_dimensional_embedding(self, data, title, path, downsize=False):
        embeddings = self.get_embeddings(self.dims)
        if downsize:
            projections, timing = self.downsize_data(data) 
        
            for name in timing:
                #title = f"{name} (time {timing[name]:.3f}s  {passed_title})"
                #file_path = str(path) + str(name) + '.html'
                self.plot_embedding(projections[name], title, path)
        else:
            self.plot_embedding(data, title, path)


    def plot_embedding(self, data, title, path):
        cdict = { 0: 'blue', 1: 'red'}

        #print(data)

        if self.dims == 2:
            self.plot_2d(data, title, path, cdict)
        elif self.dims == 3:
            self.plot_3d(data, title, path, cdict)
        
    def plot_2d(self, data, title, path, cdict):
        df = pd.DataFrame({ 'x1': data[:,0],
                            'x2': data[:,1],
                            'label': np.asarray(self.labels['defects']) })

        for label in np.unique(self.labels):
            idx = np.where(self.labels == label)
            fig = px.scatter(df, x='x1', y='x2', 
                                color='label', color_discrete_map=cdict,
                                opacity=.4)

            fig.update_layout(
                title = title
            )
            
        file_name = path + title
        print(path)
        fig.write_html(file_name, div_id = title)

    def plot_3d(self, data, title, path, cdict):
        df = pd.DataFrame({ 'x1': data[:,0],
                            'x2': data[:,1],
                            'x3': data[:,2],
                            'label': np.asarray(self.labels['defects'])})

        for label in np.unique(self.labels):
            idx = np.where(self.labels == label)
            fig = px.scatter_3d(df, x='x1', y='x2', z='x3',
                                color='label', color_discrete_map=cdict,
                                opacity=.4)

            fig.update_layout(
                title = title
            )

        file_name = path + title 
        print(path)
        fig.write_html(file_name, div_id = title)

    
