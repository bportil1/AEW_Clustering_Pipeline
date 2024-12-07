import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from math import isclose
from sklearn.decomposition import PCA
import scipy.sparse as sp
import warnings

import cupy as cp

from optimizers_sm import *

warnings.filterwarnings("ignore")

class aew():
    def __init__(self, similarity_matrix, data, labels, gamma_init=None):
        '''
        Initialize class attributes with sparse matrix handling
        '''
        self.data = data
        self.labels = labels
        self.eigenvectors = None
        self.gamma = self.gamma_initializer(gamma_init)
        self.similarity_matrix = self.correct_similarity_matrix_diag(similarity_matrix)


    def correct_similarity_matrix_diag(self, similarity_matrix):
        '''
        Correct diagonals of precomputed similarity matrix, if necessary
        '''
        if not sp.issparse(similarity_matrix):
            similarity_matrix = sp.csr_matrix(similarity_matrix)
        identity = sp.lil_matrix((self.data.shape[0], self.data.shape[0]))
        identity.setdiag(cp.ones(self.data.shape[0]))
        similarity_matrix = similarity_matrix.tolil()  # Convert to LIL format for easy manipulation
        similarity_matrix.setdiag(cp.ones(self.data.shape[0]))  # Correct the diagonal
        return similarity_matrix

    def gamma_initializer(self, gamma_init=None):
        '''
        Initialize gamma by user chosen option
        '''
        if gamma_init is None:
            gamma = cp.ones(self.data.shape[1])
        elif gamma_init == 'var':
            gamma = cp.var(self.data, axis=0).values
        elif gamma_init == 'random_int':
            gamma = cp.random.randint(0, 1000, self.data.shape[1])
        elif gamma_init == 'random_float':
            rng = cp.random.default_rng()
            gamma = rng.random(self.data.shape[1])
        return gamma

    def similarity_function(self, pt1_idx, pt2_idx, gamma):
        '''
        Compute similarity between two points using sparse matrix operations
        '''
        point1 = cp.asarray(self.data.loc[[pt1_idx]])[0]
        point2 = cp.asarray(self.data.loc[[pt2_idx]])[0]
        deg_pt1 = cp.sum(self.similarity_matrix[pt1_idx, :])
        deg_pt2 = cp.sum(self.similarity_matrix[pt2_idx, :])
        
        # Compute squared difference and apply gamma
        similarity_measure = cp.sum(cp.where(cp.abs(gamma) > 1e-5, (((point1 - point2)**2) / (gamma**2)), 0))
        similarity_measure = cp.exp(-similarity_measure)
        
        degree_normalization_term = cp.sqrt(cp.abs(deg_pt1 * deg_pt2))
        
        if degree_normalization_term != 0 and not isclose(degree_normalization_term, 0, abs_tol=1e-100):
            return similarity_measure / degree_normalization_term
        else:
            return 0

    def objective_computation(self, section, adj_matrix, gamma):
        '''
        Compute reconstruction error for a section, adjusted for sparse matrix usage
        '''
        approx_error = 0
        for idx in section:
            degree_idx = cp.sum(adj_matrix[idx, :].toarray())
            xi_reconstruction = cp.sum([adj_matrix[idx, y] * cp.asarray(self.data.loc[[y]])[0] for y in range(len(self.gamma)) if idx != y], axis=0)
            if degree_idx != 0 and not isclose(degree_idx, 0, abs_tol=1e-100):
                xi_reconstruction /= degree_idx
            else:
                xi_reconstruction = cp.zeros(len(self.gamma))
            approx_error += cp.sum((cp.asarray(self.data.loc[[idx]])[0] - xi_reconstruction)**2)
        return approx_error

    def objective_function(self, adj_matr, gamma):
        '''
        Parallelization of error computation
        '''
        split_data = self.split(range(self.data.shape[0]), cpu_count())
        with Pool(processes=cpu_count()) as pool:
            errors = [pool.apply_async(self.objective_computation, (section, adj_matr, gamma)) for section in split_data]
            error = [error.get() for error in errors]
        return cp.sum(error)

    def gradient_computation(self, section, similarity_matrix, gamma):
        '''
        Compute gradient for a section, adjusted for sparse matrix usage
        '''
        gradient = cp.zeros(len(gamma))
        for idx in section:
            dii = cp.sum(similarity_matrix[idx, :])
            xi_reconstruction = cp.sum([similarity_matrix[idx, y] * cp.asarray(self.data.loc[[y]])[0] for y in range(len(similarity_matrix[idx])) if idx != y], axis=0)
            if dii != 0 and not isclose(dii, 0, abs_tol=1e-100):
                xi_reconstruction /= dii
                first_term = (cp.asarray(self.data.loc[[idx]])[0] - xi_reconstruction) / dii
            else:
                first_term = cp.zeros_like(xi_reconstruction)
            cubed_gamma = cp.where(cp.abs(gamma) > 1e-7, gamma**(-3), 0)
            dw_dgamma = cp.sum([(2 * similarity_matrix[idx, y] * (((cp.asarray(self.data.loc[[idx]])[0] - cp.asarray(self.data.loc[[y]])[0])**2) * cubed_gamma) * cp.asarray(self.data.loc[[y]])[0]) for y in range(self.data.shape[0]) if idx != y])
            dD_dgamma = cp.sum([(2 * similarity_matrix[idx, y] * (((cp.asarray(self.data.loc[[idx]])[0] - cp.asarray(self.data.loc[[y]])[0])**2) * cubed_gamma) * xi_reconstruction) for y in range(self.data.shape[0]) if idx != y])
            gradient += first_term * (dw_dgamma - dD_dgamma)
            gradient = cp.nan_to_num(gradient, nan=0)
        return gradient

    def split(self, a, n):
        '''
        Split an array a into n pieces
        '''
        k, m = divmod(len(a), n)
        return [a[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]

    def gradient_function(self, similarity_matrix, gamma):
        '''
        Parallelization of gradient computation using sparse matrix operations
        '''
        gradient = []
        split_data = self.split(range(self.data.shape[0]), cpu_count())
        with Pool(processes=cpu_count()) as pool:
            gradients = [pool.apply_async(self.gradient_computation, (section, similarity_matrix, gamma)) for section in split_data]
            gradients = [gradient.get() for gradient in gradients]
        return cp.sum(gradients, axis=0)

    def optimize_gamma(self, optimizer, num_iterations=100):
        if optimizer == 'adam':
            opt_obj = AdamOptimizer(self.similarity_matrix, self.gamma, self.generate_edge_weights, self.objective_function, self.gradient_function, num_iterations)
        elif optimizer == 'simulated_annealing':
            opt_obj = SimulatedAnnealingOptimizer(self.similarity_matrix, self.gamma, self.generate_edge_weights, self.objective_function, num_iterations, cooling_rate=.95)
        elif optimizer == 'particle_swarm':
            opt_obj = ParticleSwarmOptimizer(self.similarity_matrix, self.gamma, self.objective_function,  self.generate_edge_weights, 3, len(self.gamma), num_iterations)
        elif optimizer == 'swarm_based_annealing':
            opt_obj = SwarmBasedAnnealingOptimizer(self.similarity_matrix, self.gamma, self.objective_function, self.gradient_function, self.generate_edge_weights, 3, len(self.gamma), num_iterations)
        
        self.gamma = opt_obj.optimize()
        print("Optimized Gamma: ", self.gamma)

    def generate_optimal_edge_weights(self, num_iterations):
        '''
        Function to optimize gamma and set the resulting similarity matrix
        '''
        print("Generating Optimal Edge Weights")
        self.similarity_matrix = self.generate_edge_weights(self.gamma)
        self.optimize_gamma('simulated_annealing', num_iterations)
        self.similarity_matrix = self.generate_edge_weights(self.gamma)

    def edge_weight_computation(self, section, gamma):
        '''
        Compute edge weights for a section using sparse matrix operations
        '''
        res = []
        for idx in section:
            for vertex in range(self.data.shape[0]):
                if vertex != idx:
                    res.append((idx, vertex, self.similarity_function(idx, vertex, gamma)))
        return res

    def generate_edge_weights(self, gamma):
        '''
        Parallelization of edge weight computation using sparse matrix operations
        '''
        print("Generating Edge Weights")
        curr_sim_matr = sp.lil_matrix(self.similarity_matrix.shape)
        mm_file = './mmap_file'
        curr_sim_matr = cp.memmap(mm_file + 'curr_sim_matr', dtype='float32', mode='w+', shape=curr_sim_matr.shape)

        split_data = self.split(range(self.data.shape[0]), cpu_count())
        with Pool(processes=cpu_count()) as pool:
            edge_weight_res = [pool.apply_async(self.edge_weight_computation, (section, gamma)) for section in split_data]
            edge_weights = [edge_weight.get() for edge_weight in edge_weight_res]
        for section in edge_weights:
            for weight in section:
                if weight[0] != weight[1]:
                    curr_sim_matr[weight[0], weight[1]] = weight[2]
                    curr_sim_matr[weight[1], weight[0]] = weight[2]
        curr_sim_matr = self.subtract_identity(curr_sim_matr)
        print("Edge Weight Generation Complete")
        return curr_sim_matr

    def subtract_identity(self, adj_matrix):
        '''
        Subtract matrix by identity for normalized symmetric laplacian
        '''
        identity = sp.lil_matrix(adj_matrix.shape)
        identity.setdiag(2)  # Set diagonal elements to 2
        adj_matrix = identity - adj_matrix
        return adj_matrix

    def unit_normalization(self, matrix):
        '''
        Normalize matrix to unit length
        '''
        norms = cp.sqrt(cp.sum(matrix.multiply(matrix), axis=1))
        norms = norms.A.flatten()  # Convert from matrix to 1D array
        matrix = matrix.multiply(1 / norms[:, None])  # Normalize each row
        return matrix

    def get_eigenvectors(self, num_components, min_variance):
        '''
        Cast similarity matrix to lower dimensional representation using PCA
        '''
        pca = PCA()
        if num_components == 'lowest_var':
            pca.fit(self.similarity_matrix.toarray())  # Convert sparse to dense for PCA fitting
            expl_var = pca.explained_variance_ratio_
            cum_variance = expl_var.cumsum()
            num_components = (cum_variance <= min_variance).sum() + 1
        pca = PCA(n_components=num_components)
        pca_result = pca.fit_transform(self.similarity_matrix.toarray())
        pca_normalized = self.unit_normalization(sp.csr_matrix(pca_result))
        self.eigenvectors = pca_normalized.toarray()
