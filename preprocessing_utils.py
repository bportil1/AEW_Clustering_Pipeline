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
import plotly.express as px

from sklearn.compose import ColumnTransformer

class data():
    def __init__(self, data = None, labels = None):
        self.data = train_data
        self.graph = None
        self.labels = train_labels
        self.projection = None
        self.class_labels = {'class': {'normal': 0, 'anomaly':1}}
        self.similarity_matrix = None

    def scale_data(self, scaling):
        cols = self.data.loc[:, ~self.data.columns.isin(['flag',
                                                                     'land', 'wrong_fragment', 'urgent',
                                                                     'num_failed_logins', 'logged_in',
                                                                     'root_shell', 'su_attempted', 'num_shells',
                                                                     'num_access_files', 'num_outbound_cmds',
                                                                     'is_host_login', 'is_guest_login', 'serror_rate',                                                                     'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',                                                                  'same_srv_rate', 'diff_srv_rate',
                                                                     'srv_diff_host_rate', 'dst_host_same_srv_rate',
                                                                     'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',                                                              'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',                                                                'dst_host_srv_serror_rate', 'dst_host_rerror_rate',                                                                   'dst_host_srv_rerror_rate', 'protocol_type ', 'service ' ])].columns
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
            label_encoder.fit(list(self.data[column_name])+list(self.data[column_name])) 
            self.data[column_name] = label_encoder.transform(self.data[column_name])
            self.data[column_name] = label_encoder.transform(self.data[column_name])
        elif target_set == 'labels':
            self.labels = self.labels.replace(self.class_labels)
            self.labels = self.labels.replace(self.class_labels)

    def load_data(self, sample_size=None):
        self.train_data = pd.read_csv(datapath)
        if sample_size != None:
            self.data = self.data.sample(sample_size)
    
    def load_labels(self):
        self.labels = pd.DataFrame(self.data['class'], columns=['class'])
        self.data = self.data.loc[:, self.data.columns != 'class']
        self.reset_indices

    def reset_indices(self):
        self.data = self.data.reset_index(drop=True)
        self.labels = self.labels.reset_index(drop=True)

    def generate_graphs(self, num_neighbors, mode='distance', metric='euclidean'):
        self.graph = kneighbors_graph(self.data, n_neighbors=num_neighbors, mode=mode, metric=metric, p=2, include_self=True, n_jobs=-1)

class visualization():
    def __init__(self, orig_data, aew_embedding, labels):
        self.orig_data = orig_data
        self.aew_embedding = aew_embedding
        self.labels = labels

    def get_embeddings(self, num_components, embedding_subset = None):
        embeddings = {
            "Truncated SVD embedding": TruncatedSVD(n_components=num_components),
            #"Standard LLE embedding": LocallyLinearEmbedding(
            #    n_neighbors=n_neighbors, n_components=num_components, method="standard", 
            #    eigen_solver='dense', n_jobs=-1
            #),
            "Random Trees embedding": make_pipeline(
                RandomTreesEmbedding(n_estimators=200, max_depth=5, random_state=0, n_jobs=-1),
                TruncatedSVD(n_components=num_components),
            ),
            #"t-SNE embedding": TSNE(
            #        n_components=num_components,
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

    def downsize_data(self, data, data_type, num_components):
        if data_type == 'train':
            labels = self.train_labels
        elif data_type == 'test':
            labels = self.test_labels

        embeddings = self.get_embeddings(num_components)

        projections, timing = {}, {}
        for name, transformer in embeddings.items():
            print(f"Computing {name}...")
            start_time = time()
            projections[name] = transformer.fit_transform(data, labels)
            timing[name] = time() - start_time

        return projections, timing 

    def lower_dimensional_embedding(self, data, data_type, passed_title, path):
        if data_type == 'train':
            labels = self.train_labels
        elif data_type == 'test': 
            labels = self.test_labels

        embeddings = self.get_embeddings(3)
        projections, timing = self.downsize_data(data, 'train', 3) 
        
        for name in timing:
            title = f"{name} (time {timing[name]:.3f}s  {passed_title})"
            file_path = str(path) + str(name) + '.html'
            self.plot_embedding(projections[name], labels, title, file_path)

    def plot_embedding(self, data, labels, title, path):
        cdict = { 0: 'blue', 1: 'red'}

        df = pd.DataFrame({ 'x1': data[:,0],
                            'x2': data[:,1],
                            'x3': data[:,2],
                            'label': labels['class'] })
    
        for label in np.unique(labels):
            idx = np.where(labels == label)
            fig = px.scatter_3d(df, x='x1', y='x2', z='x3',
                                color='label', color_discrete_map=cdict,
                                opacity=.4)

            fig.update_layout(
                title = title
            )

        fig.write_html(path, div_id = title)
        #fig.show()
