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
from spread_opt import *

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
    def __init__(self, train_data=None, train_labels=None, test_data=None, test_labels=None, projection_type=None, dims=None, clustering_methods=None, workers = 1):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.base_path = './results/'
        self.clustering_methods = clustering_methods
        self.clustering_funcs = None
        self.projection_type = projection_type
        self.workers = workers
        self.dims=dims

        if self.clustering_methods == None:
            self.get_clustering_methods()

        #self.get_clustering_funcs()

    def get_clustering_methods(self):
        self.clustering_methods = ['Kmeans',
                                   'Spectral', 
                                   'Agglomerative',
                                   #'LabelProp',
                                   #'DBSCAN', 
                                   'HDBSCAN', 
                                   'MeanShift',
                                   'OPTICS', 
                                   'Birch', 
                                   'BisectingKmeans'
                                   ]

    def get_clustering_funcs(self, meth):
        avail_clustering_funcs = {
                'Kmeans': (lambda : self.generate_kmeans(self.get_clustering_hyperparams('Kmeans'))),
                'Spectral': (lambda : self.generate_spectral(self.get_clustering_hyperparams('Spectral'))),
                'Agglomerative': (lambda : self.generate_agglomerative(self.get_clustering_hyperparams('Agglomerative'))),
                #'LabelProp': (lambda x: self.generate_labelprop(self.get_clustering_hyperparams('LabelProp'))),
                #'DBSCAN': (lambda x: self.generate_dbscan(self.get_clustering_hyperparams('DBSCAN'))),
                'HDBSCAN': (lambda : self.generate_hdbscan(self.get_clustering_hyperparams('HDBSCAN'))),
                'MeanShift': (lambda : self.generate_meanshift(self.get_clustering_hyperparams('MeanShift'))),
                'OPTICS': (lambda: self.generate_optics(self.get_clustering_hyperparams('OPTICS'))),
                'Birch': (lambda : self.generate_birch(self.get_clustering_hyperparams('Birch'))),
                'BisectingKmeans': (lambda : self.generate_bisectingkmeans(self.get_clustering_hyperparams('BisectingKmeans'))),
                'GaussianMixture': (lambda : self.generate_gaussianmixture(self.get_clustering_hyperparams('GaussianMixture'))) 
        }
        
        return avail_clustering_funcs[meth]()

    def get_clustering_hyperparams(self, cluster_alg):

        clustering_params = {
                'Kmeans': {'k_means_alg': ('lloyd', 'elkan'),
                           'n_clusters' : [2, 3, 4, 5], #, 6, 7, 8, 9, 10, 15, 20]
                },
                'Spectral': {'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9], #, 10, 15, 20],
                             'affinity': ('nearest_neighbors', 'rbf', 'precomputed', 'precomputed_nearest_neighbors'),
                             'assign_labels': ('kmeans', 'discretize', 'cluster_qr'),
                             'workers': self.workers
                },
                'Agglomerative': {'n_clusters': [2, 3, 4, 5], #, 6, 7, 8, 9, 10, 15, 20],
                                  'linkage': ('ward', 'complete', 'average', 'single'),
                                  'metric': ('euclidean', 'manhattan')
                },
                'LabelProp': {'kernel': ('rbf', 'knn'),
                              'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                              'gamma': [15, 18, 20, 23, 25],
                              'workers': self.workers
                },
                #'DBSCAN': {'eps': np.arange(5,150 , step = 5),
                #           'min_samples': [5, 10, 15, 20, 25, 30]   
                #},
                'HDBSCAN': {"min_cluster_size": [5, 10, 15], #, 20, 25, 30]
                },
                'MeanShift': {'workers': self.workers
                },
                'OPTICS': {'min_samples': [5, 10, 15], #20, 25, 30]
                },
                'Birch': {'n_clusters': [2, 3, 4, 5, 6], #, 7, 8, 9, 10, 15, 20],
                          'threshold': [.03, .05]
                },
                'BisectingKmeans': {'k_means_alg': ('lloyd', 'elkan'),
                                    'n_clusters' : [2, 3, 4, 5], #, 7, 8, 9, 10, 15, 20],
                                    'bisecting_strategy': ('biggest_inertia', 'largest_cluster')
                },
                'GaussianMixture': {'n_components' : [2,3,4,5],
                                         'covariance_type': ('full', 'tied', 'diag', 'spherical'),
                                         'init_params': ('kmeans', 'k-means++', 'random', 'random_from_data')
                }
        }
        return clustering_params[cluster_alg]

    def generate_clustering(self):
        for alg in self.clustering_methods:
            ctg_matrices_path = self.base_path + alg.lower() + '/ctg_matrices'  
            visualizations_path = self.base_path + alg.lower() + '/visualizations' 
            results_path = self.base_path + alg.lower() + '/results' 
     
            os.makedirs(ctg_matrices_path, exist_ok=True)
            os.makedirs(visualizations_path, exist_ok=True)
            os.makedirs(results_path, exist_ok=True)

            clustering = self.get_clustering_funcs(alg)

            #clustering = self.synthetic_data_tester()

    def generate_kmeans(self, hyperparams):
        print("Computing Kmeans")
        outpath = self.base_path + "kmeans/"
        for alg in hyperparams['k_means_alg']:
            for num_clust in hyperparams['n_clusters']:
                clustering = KMeans(n_clusters=num_clust, algorithm=alg)
                self.cluster_evaluation('kmeans', (alg, num_clust), clustering) 

        return clustering

    def generate_spectral(self, hyperparams):
        print("Computing Spectral")
        outpath = self.base_path + "spectral/"
        for alg in hyperparams['assign_labels']:
            #for aff in hyperparams['affinity']:
            for num_clust in hyperparams['n_clusters']:
                clustering = SpectralClustering(n_clusters=num_clust, affinity='rbf', assign_labels=alg, n_jobs=hyperparams['workers'])
                self.cluster_evaluation('spectral', (alg, 'rbf', num_clust), clustering) 

        return clustering

    def generate_agglomerative(self, hyperparams):
        print("Computing Agglomerative")
        outpath = self.base_path + "agglomerative/"
        for alg in hyperparams['linkage']:
            for metric in hyperparams['metric']:
                if alg == 'ward' and metric != 'euclidean':
                    continue
                
                for num_clust in hyperparams['n_clusters']:
                    clustering = AgglomerativeClustering(n_clusters=num_clust, metric=metric, linkage=alg)
                    self.cluster_evaluation('agglomerative', (alg, metric, num_clust), clustering)

        return clustering

    def generate_labelprop(self, hyperparams):
        print("Computing Labelprop")
        outpath = self.base_path + "labelprop/"
        for kernel in hyperparams['kernel']:
            if kernel ==  'rbf':
                for gamma in hyperparams['gamma']:
                    clustering = LabelPropagation(kernel=kernel, gamma=gamma, n_jobs=hyperparams['workers'])
                    self.cluster_evaluation('labelprop', (kernel, gamma), clustering)

            elif kernel == 'knn':
                for num_neighs in hyperparams['n_neighbors']:
                    clustering = LabelPropagation(kernel=kernel, n_neighbors=num_neighs, n_jobs=hyperparams['workers'])
                    self.cluster_evaluation('labelprop', (kernel, num_neighs), clustering)

    def generate_dbscan(self, hyperparams):
        print("Computing DBSCAN")
        outpath = self.base_path + "dbscan/"
        for min_samps in hyperparams['min_samples']:
            for eps in hyperparams['eps']:
                clustering = DBSCAN(eps=eps/100, min_samples=min_samps)
                self.cluster_evaluation('dbscan', (eps, min_samps), clustering) 

    def generate_hdbscan(self, hyperparams):
        print("Computing HDBSCAN")
        outpath = self.base_path + "hdbscan/"
        for min_size in hyperparams['min_cluster_size']:
            clustering = HDBSCAN(min_cluster_size=min_size)
            self.cluster_evaluation('hdbscan', [min_size], clustering)         

    def generate_meanshift(self, hyperparams):
        print("Computing MeanShift")
        outpath = self.base_path + "meanshift/"
        clustering = MeanShift(n_jobs=hyperparams['workers'])
        self.cluster_evaluation('meanshift', ["no params"], clustering)  

    def generate_optics(self, hyperparams):
        print("Computing OPTICS")
        outpath = self.base_path + "optics/"
        for min_samps in hyperparams['min_samples']:
            clustering = OPTICS(min_samples=min_samps)
            self.cluster_evaluation('optics', [min_samps], clustering)

    def generate_birch(self, hyperparams):
        print("Computing Birch")
        outpath = self.base_path + "birch/"
        for thresh in hyperparams['threshold']:
            for n_clusts in hyperparams['n_clusters']:
                clustering = Birch(threshold=thresh, n_clusters=n_clusts)
                self.cluster_evaluation('birch', (n_clusts, thresh), clustering) 

    def generate_bisectingkmeans(self, hyperparams):
        print("Computing BisectingKmeans")
        outpath = self.base_path + "bisectingkmeans/"
        for alg in hyperparams['k_means_alg']:
            for strat in hyperparams['bisecting_strategy']:
                for n_clusts in hyperparams['n_clusters']:
                    clustering = BisectingKMeans(n_clusters=n_clusts, algorithm=alg, bisecting_strategy=strat)
                    self.cluster_evaluation('bisectingkmeans', (n_clusts, alg, strat), clustering)   

    def generate_gaussianmixture(self, hyperparams):
        print("Computing GaussianMixture")
        outpath = self.base_path + "gaussianmixture/"
        for cov_type in hyperparams['covariance_type']:
            for init_par in hyperparams['init_params']:
                for n_comp in hyperparams['n_components']:
                    clustering = mixture.GaussianMixture(n_components=n_comp, covariance_type=cov_type, init_params=init_par)
                    self.cluster_evaluation('gaussianmixture', (cov_type, init_par, n_comp), clustering)
                                
    def cluster_evaluation(self, alg, hyperparameters, model):

        ctg_matrices_path = self.base_path + alg.lower() + '/ctg_matrices'  
        visualizations_path = self.base_path + alg.lower() + '/visualizations'
        results_path = self.base_path + alg.lower() + '/results'  

        os.makedirs(ctg_matrices_path, exist_ok=True)
        os.makedirs(visualizations_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)

        if alg in ('spectral', 'agglomerative', 'dbscan',
                   'hdbscan', 'meanshift', 'optics',
                   'birch'):
            labels = model.fit_predict(self.train_data)
            avg_train_acc = accuracy_score(labels, self.train_labels)
            avg_test_acc = 'null'
            avg_fit_time = 'null'
            labels_pred = model.fit_predict(self.test_data)
        elif alg == 'labelprop':
            model.fit(self.train_data, self.train_labels)
            labels = model.predict(self.train_data)
            avg_train_acc = accuracy_score(labels, self.train_labels)
            avg_test_acc = 'null'
            avg_fit_time = 'null'
            labels_pred = model.predict(self.test_data)
        else:
            cv_res = cross_validate(model, self.train_data, self.train_labels, cv = 5, return_train_score = True)
            avg_train_acc = np.average(cv_res['train_score'])
            avg_test_acc = np.average(cv_res['test_score'])
            avg_fit_time = np.average(cv_res['fit_time'])
            labels_pred = model.fit_predict(self.test_data)
            labels = model.fit_predict(self.train_data)

        labels_true = self.test_labels.values.flatten()

        test_set_acc = accuracy_score(labels_pred, labels_true)

        #silhoutte = metrics.silhouette_score(self.train_data, labels, metric='euclidean')     # Silhouette Coefficient
        #calinski_harabasz = metrics.calinski_harabasz_score(self.train_data, labels)  # Calinski-Harabasz Index
        #davies_bouldin = metrics.davies_bouldin_score(self.train_data, labels)    # Davies-Bouldin Index

        ri = metrics.rand_score(labels_true, labels_pred)   # RAND score
        ari = metrics.adjusted_rand_score(labels_true, labels_pred) # Adjusted RAND score

        mis = metrics.mutual_info_score(labels_true, labels_pred)  # mutual info score
        amis = metrics.adjusted_mutual_info_score(labels_true, labels_pred)    # adjusted mutual information score
        nmis = metrics.normalized_mutual_info_score(labels_true, labels_pred)  # normalized mutual info score

        hmg = metrics.homogeneity_score(labels_true, labels_pred)   # homogeneity
        cmplt = metrics.completeness_score(labels_true, labels_pred)    # completeness
        v_meas = metrics.v_measure_score(labels_true, labels_pred)   # v_measure score

        fowlkes_mallows = metrics.fowlkes_mallows_score(labels_true, labels_pred)   # Fowlkes-Mallows scores

        cntg_mtx = contingency_matrix(labels_true, labels_pred)     # Contingency Matrix

        d = { 'clustering': alg, 'hyperparameters': hyperparameters, 'train_score_avg': avg_train_acc, 
                'test_score_avg_cv': avg_test_acc, 'avg_fit_time': avg_fit_time, 'test_set_acc': test_set_acc, ''''Silhoutte' : silhoutte, 'Calinski_Harbasz' : calinski_harabasz, 'Davies_Bouldin' : davies_bouldin,'''
             'RAND' : ri , 'ARAND': ari, 'MIS' : mis, 'AMIS' : amis, 'NMIS' : nmis, 'Hmg' : hmg, 'Cmplt' : cmplt,
             'V_meas' : v_meas, 'FMs' : fowlkes_mallows}

        df = pd.DataFrame.from_dict(d)
        
        filename_base = '/alg_' + str(self.dims)  

        for hyper_param in hyperparameters:

            filename_base += "_" + str(hyper_param) 

        cntg_mtx_name = ctg_matrices_path + filename_base + ".csv"

        f = open(cntg_mtx_name, 'a')

        f.close()

        np.savetxt(cntg_mtx_name, cntg_mtx, delimiter=',', fmt='%d')

        results_file_name = results_path + filename_base + "_" + str(self.projection_type)  +".csv"

        f = open(results_file_name, 'a')

        f.close()

        df.to_csv(results_file_name, index=False)

        vis_file_name = visualizations_path + filename_base + ".html"

        labels_pred = pd.DataFrame(labels_pred, columns=['class'])

        #data_obj = data(test_labels=labels_pred)

        #data_obj.lower_dimensional_embedding(self.test_data, 'test', filename_base, vis_file_name)

    def synthetic_data_tester(self):

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
            "n_clusters": 2,
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

            X_data = data(train_data = X_df)

            X_data.generate_graphs('train')

            X_graph = X_data.train_graph

            #print(y[:5])
            y_df = pd.DataFrame(y)

            prec_gamma = np.var(X_data.train_data, axis=0).values

            X_obj = aew(X_graph, X_df, y_df, prec_gamma)

            y = y_df[y_df.columns[0]].values

            #print(y[:5])

            X_obj.generate_optimal_edge_weights(10)

            #X_obj.generate_edge_weights()

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
            

            '''
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
                    algorithm.fit(X)

                t1 = time.time()
                if hasattr(algorithm, "labels_"):
                    y_pred = algorithm.labels_.astype(int)
                else:
                    y_pred = algorithm.predict(X)
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

                #plt.title(accuracy_score(y,y_pred))
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
                plot_num += 1

            plt.show()
            '''
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
                plot_num += 1
        plt.show()

