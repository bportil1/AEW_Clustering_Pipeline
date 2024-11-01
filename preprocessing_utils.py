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
    def __init__(self, train_data = None, train_labels = None, test_data = None, test_labels = None, output_path = None):
        self.train_data = train_data
        self.train_graph = None
        self.train_labels = train_labels
        self.train_projection = None
        self.test_data = pd.DataFrame()
        self.test_graph = None
        self.test_labels = test_labels
        self.test_projection = None
        self.class_labels = {'class': {'normal': 0, 'anomaly':1}}
        self.similarity_matrix = None

    def scale_data(self, scaling):
        cols = self.train_data.loc[:, ~self.train_data.columns.isin(['flag',
                                                                     'land', 'wrong_fragment', 'urgent',
                                                                     'num_failed_logins', 'logged_in',
                                                                     'root_shell', 'su_attempted', 'num_shells',
                                                                     'num_access_files', 'num_outbound_cmds',
                                                                     'is_host_login', 'is_guest_login', 'serror_rate',                                                                     'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',                                                                  'same_srv_rate', 'diff_srv_rate',
                                                                     'srv_diff_host_rate', 'dst_host_same_srv_rate',
                                                                     'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',                                                              'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',                                                                'dst_host_srv_serror_rate', 'dst_host_rerror_rate',                                                                   'dst_host_srv_rerror_rate', 'protocol_type ', 'service ' ])].columns
        cols = np.asarray(cols)
        #print(self.train_data.columns)
        #print(cols)
        if scaling == 'standard':
            ct = ColumnTransformer([('normalize', StandardScaler(), cols)],
                                    remainder='passthrough' 
                                  )
            
            #self.train_data[[col for col in self.train_data]] = StandardScaler().fit_transform(self.train_data[[col for col in self.train_data]])
            
            #self.train_data.loc[:, self.train_data.columns not in ['protocol_type', 'service', 'flag']] = StandardScaler().fit_transform(self.train_data.loc[:, self.train_data.columns not in ['protocol_type', 'service', 'flag']])
            transformed_cols = ct.fit_transform(self.train_data)

            self.train_data = pd.DataFrame(transformed_cols, columns = self.train_data.columns)
            if not self.test_data.empty:
                #self.test_data[[col for col in self.test_data]] = StandardScaler().fit_transform(self.test_data[[col for col in self.test_data]])
                
                #self.test_data.loc[:, self.test_data.columns not in ['protocol_type', 'service', 'flag']] = StandardScaler().fit_transform(self.test_data.loc[:, self.test_data.columns not in ['protocol_type', 'service', 'flag']]) 
        
                self.test_data = pd.DataFrame(ct.fit_transform(self.test_data), columns = self.test_data.columns)


        elif scaling == 'min_max':
            #min_max_scaler = MinMaxScaler()
            ct = ColumnTransformer([('normalize', MinMaxScaler(), cols)],
                                    remainder='passthrough'  
                                  ) 


            #self.train_data.loc[:, self.train_data.columns not in ['protocol_type', 'service', 'flag']] = StandardScaler().fit_transform(self.train_data.loc[:, self.train_data.columns not in ['protocol_type', 'service', 'flag']])   
            #self.train_data[[col for col in self.train_data]] = min_max_scaler.fit_transform(self.train_data[[col for col in self.train_data]])
            
            transformed_cols = ct.fit_transform(self.train_data)
            self.train_data = pd.DataFrame(transformed_cols, columns = self.train_data.columns)

            #print(self.test_data.empty)

            if not self.test_data.empty:
                #self.test_data[[col for col in self.test_data]] = min_max_scaler.fit_transform(self.test_data[[col for col in self.test_data]])
                transformed_cols = ct.fit_transform(self.test_data)

                self.test_data = pd.DataFrame(transformed_cols, columns = self.test_data.columns)


        #self.test_data.loc[:, self.test_data.columns not in ['protocol_type', 'service', 'flag']] = StandardScaler().fit_transform(self.test_data.loc[:, self.test_data.columns not in ['protocol_type', 'service', 'flag']])
        
        elif scaling == 'robust':
            ct = ColumnTransformer([('scaler', RobustScaler(), cols)],
                                    remainder='passthrough'
                                  )

            self.train_data = pd.DataFrame(ct.fit_transform(self.train_data), columns = self.train_data.columns)
            #self.train_data.loc[:, self.train_data.columns not in ['protocol_type', 'service', 'flag']] = StandardScaler().fit_transform(self.train_data.loc[:, self.train_data.columns not in ['protocol_type', 'service', 'flag']])   
            #self.train_data[[col for col in self.train_data]] = RobustScaler().fit_transform(self.train_data[[col for col in self.train_data]])
            if not self.test_data.empty:
                #self.test_data.loc[:, self.test_data.columns not in ['protocol_type', 'service', 'flag']] = StandardScaler().fit_transform(self.test_data.loc[:, self.test_data.columns not in ['protocol_type', 'service', 'flag']])
                #self.test_data[[col for col in self.test_data]] = RobustScaler().fit_transform(self.test_data[[col for col in self.test_data]])
                self.test_data = pd.DataFrame(ct.fit_transform(self.test_data), columns = self.test_data.columns)
        else:
            print("Scaling arg not supported")
        
    def encode_categorical(self, column_name, target_set):
        label_encoder = LabelEncoder()

        if target_set == 'data': 
            label_encoder.fit(list(self.train_data[column_name])+list(self.test_data[column_name])) 
            self.train_data[column_name] = label_encoder.transform(self.train_data[column_name])
            self.test_data[column_name] = label_encoder.transform(self.test_data[column_name])
        elif target_set == 'labels':
            self.train_labels = self.train_labels.replace(self.class_labels)
            self.test_labels = self.test_labels.replace(self.class_labels)

    def load_data(self, datapath, data_type):
        if data_type == 'train':
            self.train_data = pd.read_csv(datapath)
            self.train_data = self.train_data.sample(800)
            #self.train_data = self.train_data
        elif data_type == 'test':
            self.test_data = pd.read_csv(datapath)

    def load_labels(self, data_type, datapath=None, from_data = False):
    
        if from_data:
            if data_type == 'train':
                self.train_labels = pd.DataFrame(self.train_data['class'], columns=['class'])
                #self.train_labels.columns = ['class']
                self.train_data = self.train_data.loc[:, self.train_data.columns != 'class']
            elif data_type == 'test':
                self.test_labels['class'] = pd.DataFrame(self.test_data['class'], columns=['class'])
                #self.test_labels.columns = ['class']
                self.test_data = self.test_data.loc[:, self.test_data.columns != 'class']
        else:
            if data_type == 'train':
                self.train_labels = pd.read_csv(datapath)
            elif data_type == 'test':
                self.test_labels = pd.read_csv(datapath)

    def split_data(self, split_size):
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(self.train_data, self.train_labels, test_size = split_size)

        self.reset_indices()

    def reset_indices(self):
        self.train_data = self.train_data.reset_index(drop=True)
        self.train_labels = self.train_labels.reset_index(drop=True)
        self.test_data = self.test_data.reset_index(drop=True)
        self.test_labels = self.test_labels.reset_index(drop=True)


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

    def remove_disconnections(self):

        self.train_data = self.train_data[(self.train_data != 0).any(axis=1)]
        self.test_data = self.test_data.loc[:, (self.test_data != 0).any(axis=0)]
        self.test_data = self.test_data.loc[:, (self.test_data != 0).any(axis=0)]

    def generate_graphs(self, data_type):
        if data_type == 'train':
            self.train_graph = kneighbors_graph(self.train_data, n_neighbors=150, mode='distance', metric='euclidean', p=2, include_self=True, n_jobs=-1)
        elif data_type == 'test':
            self.test_graph = kneighbors_graph(self.test_data, n_neighbors=150, mode='distance', metric='euclidean', p=2, include_self=True, n_jobs=-1)
            
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
