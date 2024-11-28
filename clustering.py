import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.semi_supervised import LabelPropagation
from sklearn.cluster import HDBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch
from sklearn.cluster import BisectingKMeans
from sklearn.metrics.cluster import contingency_matrix
from sklearn import metrics
import os
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from preprocessing_utils import *
from aew_sm import *
#from aew_gpu_3 import *

import warnings

import time
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np

from sklearn import cluster

from sklearn import datasets

from sklearn import mixture

warnings.resetwarnings()

class clustering():
    def __init__(self, base_data=None, data=None, labels=None, path_name = None, name_append = None, workers = 1):
        self.base_data=base_data
        self.data = data
        #if labels.empty:
        self.labels = self.flatten_labels(labels)
        #if labels != None:
        #    self.labels = self.flatten_labels(labels)
        self.pred_labels = None
        self.base_path = path_name 
        self.workers = workers
        self.fin_df = pd.DataFrame(columns=[ 'test_type','clustering', 'hyperparameters', 'label_accuracy', 'Silhoutte', 'Calinski_Harbasz', 'Davies_Bouldin', 'RAND', 'ARAND','MIS', 'AMIS', 'NMIS', 'Hmg', 'Cmplt', 'V_meas', 'FMs'])
        self.name_append = name_append
        

    def flatten_labels(self, labels):
        return labels['defects'].tolist()

    def get_clustering_hyperparams(self, cluster_alg):

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

        ctg_matrices_path = self.base_path + alg.lower() + '/ctg_matrices'  
        visualizations_path = self.base_path + alg.lower() + '/visualizations'
        results_path = self.base_path + alg.lower() + '/results'  

        os.makedirs(ctg_matrices_path, exist_ok=True)
        os.makedirs(visualizations_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)

        labels_pred = model.fit_predict(self.data)
        self.pred_labels = labels_pred
        label_acc = accuracy_score(labels_pred, self.labels)

        silhoutte = metrics.silhouette_score(self.data, labels_pred, metric='euclidean')     # Silhouette Coefficient
        calinski_harabasz = metrics.calinski_harabasz_score(self.data, labels_pred)  # Calinski-Harabasz Index
        davies_bouldin = metrics.davies_bouldin_score(self.data, labels_pred)    # Davies-Bouldin Index

        ri = metrics.rand_score(self.labels, labels_pred)   # RAND score
        ari = metrics.adjusted_rand_score(self.labels, labels_pred) # Adjusted RAND score

        mis = metrics.mutual_info_score(self.labels, labels_pred)  # mutual info score
        amis = metrics.adjusted_mutual_info_score(self.labels, labels_pred)    # adjusted mutual information score
        nmis = metrics.normalized_mutual_info_score(self.labels, labels_pred)  # normalized mutual info score

        hmg = metrics.homogeneity_score(self.labels, labels_pred)   # homogeneity
        cmplt = metrics.completeness_score(self.labels, labels_pred)    # completeness
        v_meas = metrics.v_measure_score(self.labels, labels_pred)   # v_measure score

        fowlkes_mallows = metrics.fowlkes_mallows_score(self.labels, labels_pred)   # Fowlkes-Mallows scores

        cntg_mtx = contingency_matrix(self.labels, labels_pred)     # Contingency Matrix

        d = { 'test_type': self.name_append, 'clustering': alg, 'hyperparameters': str(hyperparameters), 'label_accuracy': label_acc, 
              'Silhoutte' : silhoutte, 'Calinski_Harbasz' : calinski_harabasz, 'Davies_Bouldin' : davies_bouldin,
              'RAND' : ri , 'ARAND': ari, 'MIS' : mis, 'AMIS' : amis, 'NMIS' : nmis, 'Hmg' : hmg, 'Cmplt' : cmplt,
              'V_meas' : v_meas, 'FMs' : fowlkes_mallows}

        df = pd.DataFrame([d])

        self.fin_df = pd.concat([self.fin_df, df], ignore_index=True)

        filename_base = '/spectral'  

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

        vis_file_name = filename_base + "_eigen.html"

        labels_pred = pd.DataFrame(labels_pred, columns=['defects'])

        if isinstance(self.data, np.ndarray):
            dims = self.data.shape[1]  # Returns number of columns in ndarray
        elif isinstance(self.data, pd.DataFrame):
            dims = len(self.data.columns)

        visualizer_obj = visualizer(labels=labels_pred, dims = dims)

        visualizer_obj.lower_dimensional_embedding(self.data, vis_file_name, visualizations_path)

        whole_data_name = filename_base + "_whole.html"

        visualizer_obj.lower_dimensional_embedding(self.base_data, whole_data_name, visualizations_path)

def synthetic_data_tester(rep, iterations):

    from sklearn import cluster, datasets, mixture

    n_samples = 500
    seed = 30
    noisy_circles = datasets.make_circles(
        n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
    )
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
    rng = np.random.RandomState(seed)
    no_structure = rng.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std= [ 1.5, 2.5, 0.5], random_state=random_state
    )

    # ============
    # Set up cluster parameters
    # ============
    plt.figure(figsize=(9 * 2 + 3, 13))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
    )

    plot_num = 1

    default_base = {
            "quantile": 0.3,
            "eps": 0.3,
            "damping": 0.9,
            "preference": -200,
            "n_neighbors": 3,
            "n_clusters": 3,
            "min_samples": 7,
            "xi": 0.05,
            "min_cluster_size": 0.1,
            "allow_single_cluster": True,
            "hdbscan_min_cluster_size": 15,
            "hdbscan_min_samples": 3,
            "random_state": 42,
    }

    datasets = [
        (
            noisy_circles,
            {
                            "damping": 0.77,
                            "preference": -240,
                            "quantile": 0.2,
                            "n_clusters": 2,
                            "min_samples": 7,
                            "xi": 0.08,
            },
            ),
        (
            noisy_moons,
            {
            "damping": 0.75,
            "preference": -220,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.1,
            },
        ),
        (
            varied,
            {
                "eps": 0.18,
                "n_neighbors": 2,
                "min_samples": 7,
                "xi": 0.01,
                "min_cluster_size": 0.2,
            },
        ),
        (
            aniso,
            {
                "eps": 0.15,
                "n_neighbors": 2,
                "min_samples": 7,
                "xi": 0.1,
                "min_cluster_size": 0.2,
            },
        ),
        (blobs, {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2})
    ]
        
    for i_dataset, (dataset, algo_params) in enumerate(datasets):
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)

        X, y = dataset

        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)
            
        #print(X)

        X_df = pd.DataFrame(X)

        #print(X)

        X_data = data(data=X_df, labels=y, datapath=".")

        X_data.generate_graphs(150)

        X_graph = X_data.graph

        #print(y[:5])
        y_df = pd.DataFrame(y)

        #prec_gamma = np.var(X_data.train_data, axis=0).values

        X_obj = aew(X_graph.toarray(), X_df, y_df, 'var')

        y = y_df[y_df.columns[0]].values

        #print(y[:5])

        X_obj.generate_optimal_edge_weights(iterations)

        #X_obj.generate_edge_weights()

        X_obj.get_eigenvectors(2, .90)

        X_aew = X_obj.eigenvectors
                   
        #print(X)
    
        bandwidth = cluster.estimate_bandwidth(X_aew, quantile=params["quantile"])

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
             X_aew, n_neighbors=params["n_neighbors"], include_self=False
        )
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # ============
        # Create cluster objects
        # ============
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        two_means = cluster.MiniBatchKMeans(
                n_clusters=params["n_clusters"],
                random_state=params["random_state"],
        )
        ward = cluster.AgglomerativeClustering(
                n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
        )
        spectral = cluster.SpectralClustering(
                n_clusters=params["n_clusters"],
                eigen_solver="arpack",
                affinity="rbf",
                random_state=params["random_state"],
        )
        dbscan = cluster.DBSCAN(eps=params["eps"])
        hdbscan = cluster.HDBSCAN(
        min_samples=params["hdbscan_min_samples"],
        min_cluster_size=params["hdbscan_min_cluster_size"],
        allow_single_cluster=params["allow_single_cluster"],
        )
        optics = cluster.OPTICS(
                min_samples=params["min_samples"],
                xi=params["xi"],
                min_cluster_size=params["min_cluster_size"],
        )
        affinity_propagation = cluster.AffinityPropagation(
                damping=params["damping"],
                preference=params["preference"],
                random_state=params["random_state"],
        )
        average_linkage = cluster.AgglomerativeClustering(
                linkage="average",
                metric="cityblock",
                n_clusters=params["n_clusters"],
                connectivity=connectivity,
        )
        birch = cluster.Birch(n_clusters=params["n_clusters"])
        gmm = mixture.GaussianMixture(
                n_components=params["n_clusters"],
                covariance_type="full",
                random_state=params["random_state"],
        )

        clustering_algorithms = (
                ("MiniBatch\nKMeans", two_means),
                ("Affinity\nPropagation", affinity_propagation),
                ("MeanShift", ms),
                ("Spectral\nClustering", spectral),
                ("Ward", ward),
                ("Agglomerative\nClustering", average_linkage),
                ("DBSCAN", dbscan),
                ("HDBSCAN", hdbscan),
                ("OPTICS", optics),
                ("BIRCH", birch),
                ("Gaussian\nMixture", gmm),
        )

        for name, algorithm in clustering_algorithms:
            t0 = time.time()
                        
        for name, algorithm in clustering_algorithms:
            t0 = time.time()

            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                        "ignore",
                        message="the number of connected components of the "
                        + "connectivity matrix is [0-9]{1,2}"
                            + " > 1. Completing it to avoid stopping the tree early.",
                            category=UserWarning,
                )
                warnings.filterwarnings(
                            "ignore",
                            message="Graph is not fully connected, spectral embedding"
                            + " may not work as expected.",
                            category=UserWarning,
                )
                    #algorithm = self.get_clustering_funcs(name)
                algorithm.fit(X_aew)

            t1 = time.time()
            if hasattr(algorithm, "labels_"):
                y_pred = algorithm.labels_.astype(int)
            else:
                y_pred = algorithm.predict(X_aew)
            #print(X)

            plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                name1 = name + str(accuracy_score(y,y_pred))
                plt.title(name1, size=18)


            colors = np.array(
                        list(
                            islice(
                                cycle(
                                    [
                                         "#377eb8",
                                         "#ff7f00",
                                         "#4daf4a",
                                         "#f781bf",
                                         "#a65628",
                                         "#984ea3",
                                         "#999999",
                                         "#e41a1c",
                                         "#dede00",
                                    ]
                                    ),
                                int(max(y_pred) + 1),
                            )
                        )
            )
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

            plt.title(accuracy_score(y,y_pred))
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            plt.xticks(())
            plt.yticks(())
            plt.text(
                        0.99,
                        0.01,
                        ("%.2fs" % (t1 - t0)).lstrip("0"),
                        transform=plt.gca().transAxes,
                        size=15,
                        horizontalalignment="right",
            )
            #plt_name = "synthetic_data_" + str(plot_num) + ".png"
            #plt.savefig(plt_name) 
            plot_num += 1

    plt_name = "plot_" +str(rep)+".png"
    plt.savefig(plt_name, bbox_inches = 'tight')

    plt_name = "plot_" +str(rep) + "_" + str(cycle) + ".png"
    plt.savefig(plt_name, bbox_inches = 'tight')

