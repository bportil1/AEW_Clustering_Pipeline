import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import contingency_matrix
from sklearn import metrics
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from preprocessing_utils import *
from aew_gpu import *
import warnings
import time
from itertools import cycle, islice
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn import mixture

warnings.resetwarnings()

class clustering():
    '''
    Class to cluster data and collect results
    '''
    def __init__(self, base_data=None, data=None, labels=None, path_name = None, name_append = None, workers = 1):
        self.base_data=base_data
        self.data = data
        self.labels = self.flatten_labels(labels)
        self.pred_labels = None
        self.base_path = path_name 
        self.workers = workers
        self.fin_df = pd.DataFrame(columns=[ 'test_type','clustering', 'hyperparameters', 'label_accuracy', 'Silhoutte', 'Calinski_Harbasz', 'Davies_Bouldin', 'RAND', 'ARAND','MIS', 'AMIS', 'NMIS', 'Hmg', 'Cmplt', 'V_meas', 'FMs'])
        self.name_append = name_append
        
    def flatten_labels(self, labels):
        '''
        Class to convert labels to a one-dimensional vector
        '''
        return labels['defects'].tolist()

    def get_clustering_hyperparams(self, cluster_alg):
        '''
        Utility to collect clustering methods parameters for a grid search
        '''
        clustering_params = {
                'gaussianmixture': {'n_components' : [2,3,4,5],
                                         'covariance_type': ('full', 'tied', 'diag', 'spherical'),
                                         'init_params': ('kmeans', 'k-means++', 'random', 'random_from_data')
                },
                'kmeans': {'k_means_alg': ('lloyd', 'elkan'),
                           'n_clusters' : [2, 3, 4, 5, 6, 7, 8] # , 9, 10, 15, 20]
                },
                'spectral': {'n_clusters': [2], 
                             'affinity': ('nearest_neighbors', 'rbf'),
                             'assign_labels': ('kmeans', 'discretize', 'cluster_qr'),
                             'workers': self.workers
                }
        }
        return clustering_params[cluster_alg]

    def generate_gaussianmixture(self):
        '''
        Gaussian Mixture Model grid search and data collection utility 
        '''
        print("Computing GaussianMixture")
        hyperparams = self.get_clustering_hyperparams('gaussianmixture')
        outpath = self.base_path + "gaussianmixture/"
        for cov_type in hyperparams['covariance_type']:
            for init_par in hyperparams['init_params']:
                for n_comp in hyperparams['n_components']:
                    clustering = mixture.GaussianMixture(n_components=n_comp, covariance_type=cov_type, init_params=init_par)
                    self.cluster_evaluation('gaussianmixture', (cov_type, init_par, n_comp), clustering)
        fin_path = self.base_path + self.name_append +'_concat_data.csv'
        self.fin_df.to_csv(fin_path, index=False)


    def generate_kmeans(self):
        '''
        K-means Model grid search and data collection utility 
        '''
        print("Computing K-means")
        hyperparams = self.get_clustering_hyperparams('kmeans')
        outpath = self.base_path + "kmeans/"
        for aff in hyperparams['k_means_alg']:
            for num_clust in hyperparams['n_clusters']:
                clustering = KMeans(n_clusters=num_clust, algorithm=aff)
                self.cluster_evaluation('kmeans', (aff, num_clust), clustering) 
        fin_path = self.base_path + self.name_append +'_concat_data.csv'
        self.fin_df.to_csv(fin_path, index=False)


    def generate_spectral(self):
        '''
        Spectral Model grid search and data collection utility 
        '''
        print("Computing Spectral")
        hyperparams = self.get_clustering_hyperparams('spectral')
        outpath = self.base_path + "spectral/"
        for alg in hyperparams['assign_labels']:
            for aff in hyperparams['affinity']:
                for num_clust in hyperparams['n_clusters']:
                    clustering = SpectralClustering(n_clusters=num_clust, affinity=aff, assign_labels=alg, n_jobs=hyperparams['workers'])
                    self.cluster_evaluation('spectral', (alg, aff, num_clust), clustering) 
        fin_path = self.base_path + self.name_append +'_concat_data.csv'
        self.fin_df.to_csv(fin_path, index=False)

    def cluster_evaluation(self, alg, hyperparameters, model):
        '''
        Model evalutation and data collection utility 
        '''
        ctg_matrices_path = self.base_path + alg.lower() + '/ctg_matrices'  
        visualizations_path = self.base_path + alg.lower() + '/visualizations'
        results_path = self.base_path + alg.lower() + '/results'  
        os.makedirs(ctg_matrices_path, exist_ok=True)
        os.makedirs(visualizations_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)
        labels_pred = model.fit_predict(self.data)
        self.pred_labels = labels_pred
        label_acc = accuracy_score(labels_pred, self.labels)
        silhoutte = metrics.silhouette_score(self.data, labels_pred, metric='euclidean')    
        calinski_harabasz = metrics.calinski_harabasz_score(self.data, labels_pred)  
        davies_bouldin = metrics.davies_bouldin_score(self.data, labels_pred)   
        ri = metrics.rand_score(self.labels, labels_pred)   
        ari = metrics.adjusted_rand_score(self.labels, labels_pred) 
        mis = metrics.mutual_info_score(self.labels, labels_pred)  
        amis = metrics.adjusted_mutual_info_score(self.labels, labels_pred)   
        nmis = metrics.normalized_mutual_info_score(self.labels, labels_pred)  
        hmg = metrics.homogeneity_score(self.labels, labels_pred)   
        cmplt = metrics.completeness_score(self.labels, labels_pred)   
        v_meas = metrics.v_measure_score(self.labels, labels_pred)   
        fowlkes_mallows = metrics.fowlkes_mallows_score(self.labels, labels_pred)  
        cntg_mtx = contingency_matrix(self.labels, labels_pred)
        d = { 'test_type': self.name_append, 'clustering': alg, 'hyperparameters': str(hyperparameters), 'label_accuracy': label_acc, 
              'Silhoutte' : silhoutte, 'Calinski_Harbasz' : calinski_harabasz, 'Davies_Bouldin' : davies_bouldin,
              'RAND' : ri , 'ARAND': ari, 'MIS' : mis, 'AMIS' : amis, 'NMIS' : nmis, 'Hmg' : hmg, 'Cmplt' : cmplt,
              'V_meas' : v_meas, 'FMs' : fowlkes_mallows}
        df = pd.DataFrame([d])
        self.fin_df = pd.concat([self.fin_df, df], ignore_index=True)
        filename_base = '/' + alg   
        for hyper_param in hyperparameters:
            filename_base += "_" + str(hyper_param) 
        cntg_mtx_name = ctg_matrices_path + filename_base + ".csv"
        f = open(cntg_mtx_name, 'a')
        f.close()
        np.savetxt(cntg_mtx_name, cntg_mtx, delimiter=',', fmt='%d')
        results_file_name = results_path + filename_base + ".csv"
        f = open(results_file_name, 'a')
        f.close()
        df.to_csv(results_file_name, index=False)
        #vis_file_name = filename_base + "_eigen.html"
        #labels_pred = pd.DataFrame(labels_pred, columns=['defects'])
        #if isinstance(self.data, np.ndarray):
        #    dims = self.data.shape[1]  
        #elif isinstance(self.data, pd.DataFrame):
        #    dims = len(self.data.columns)
        #visualizer_obj = visualizer(labels=labels_pred, dims = dims)
        #visualizer_obj.lower_dimensional_embedding(self.data, vis_file_name, visualizations_path)
        #whole_data_name = filename_base + "_whole.html"
        #visualizer_obj.lower_dimensional_embedding(self.base_data, whole_data_name, visualizations_path)

