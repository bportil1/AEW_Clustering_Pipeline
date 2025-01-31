import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from math import isclose, sqrt, exp
from sklearn.decomposition import PCA
import scipy.sparse as sp
import warnings

import numba
from numba import vectorize, cuda

from optimizers import *

import time

warnings.filterwarnings("ignore")

class aew():

    def __init__(self, similarity_matrix, data, comp_data, labels, test_rep=None, threads=32, blocks=128, gamma_init=None):
        '''
        Initialize class attributes with sparse matrix handling
        '''
        self.data = data
        self.labels = labels
        self.eigenvectors = None
        self.gamma = self.gamma_initializer(gamma_init)
        #self.similarity_matrix = self.correct_similarity_matrix_diag(similarity_matrix)
        self.similarity_matrix = similarity_matrix #.toarray()
        self.threads = (16, 16)
        self.blocks = (len(self.similarity_matrix[0]) + self.threads[0] - 1) // self.threads[0], \
                      (len(self.similarity_matrix[0]) + self.threads[1] - 1) // self.threads[1]
        print("Init sim matr: ", self.similarity_matrix)

    #@vectorize
    def correct_similarity_matrix_diag(self, similarity_matrix):
        '''
        Correct diagonals of precomputed similarity matrix, if necessary
        '''
        if not sp.issparse(similarity_matrix):
            similarity_matrix = sp.csr_matrix(similarity_matrix)
        identity = sp.lil_matrix((self.data.shape[0], self.data.shape[0]))
        identity.setdiag(np.ones(self.data.shape[0]))
        similarity_matrix = similarity_matrix.tolil()  
        return similarity_matrix.toarray()

    def gamma_initializer(self, gamma_init=None):
        '''
        Initialize gamma by user chosen option
        '''
        if gamma_init is None:
            gamma = np.ones(self.data.shape[1])
        elif gamma_init == 'var':
            gamma = np.var(self.data, axis=0).values
        elif gamma_init == 'random_int':
            gamma = np.random.randint(0, 1000, self.data.shape[1])
        elif gamma_init == 'random_float':
            rng = np.random.default_rng()
            gamma = rng.random(self.data.shape[1])
        return gamma

    def objective_function(self, adj_matr, gamma):
        '''
        Parallelization of error computation
        '''
        matr_d = cuda.to_device(adj_matr)
        data_d = cuda.to_device(self.data.to_numpy())
        gamma_d = cuda.to_device(gamma)
        degree_d = cuda.device_array(shape=(self.data.shape[0],), dtype=np.float32)
        error_arr_d = cuda.device_array(shape=(self.data.shape[0],), dtype=np.float32)
        xi_reconstruction = cuda.device_array(shape=(self.data.shape[0], self.data.shape[0]), dtype=np.float32)
        out_d = cuda.device_array(shape=(self.data.shape[0],), dtype=np.float32)
        cuda.synchronize()
        objective_computation[self.blocks, self.threads](matr_d, data_d, gamma_d, degree_d, error_arr_d, xi_reconstruction, out_d)
        cuda.synchronize()
        error = out_d.copy_to_host()
        error = np.nan_to_num(error, nan=0, posinf=1e10, neginf=-1e10)
        #print("Degree Matr: ", degree_d.copy_to_host())
        #print("Error Sum: ", np.sum(error))
        return np.sum(error)

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
        d_sim_matr = cuda.to_device(similarity_matrix)
        d_data = cuda.to_device(self.data.to_numpy())
        d_gamma = cuda.to_device(gamma)
        d_degrees = cuda.device_array(shape=(self.data.shape[0],), dtype=np.float32)
        d_first_term = cuda.device_array(shape=(self.data.shape[1], ), dtype=np.float32)
        d_second_term = cuda.device_array(shape=(self.data.shape[1], ), dtype=np.float32)
        d_third_term = cuda.device_array(shape=(self.data.shape[1], ), dtype=np.float32)
        d_gradients = cuda.device_array(shape=(self.data.shape[1],), dtype=np.float32)
        gradient_computation[self.blocks, self.threads](d_sim_matr, d_data, d_gamma, d_degrees, d_first_term, d_second_term, d_third_term, d_gradients)
        gradients = d_gradients.copy_to_host()
        gradients = np.nan_to_num(gradients, nan=0)
        return gradients

    def optimize_gamma(self, optimizer, num_iterations=100):
        '''
        Function to call optimization function 
        '''
        print("Beggining Optimization: ", optimizer)
        if optimizer == 'adam':
            opt_obj = AdamOptimizer(self.similarity_matrix, self.gamma, self.generate_edge_weights, self.objective_function, self.gradient_function, num_iterations)
        elif optimizer == 'simulated_annealing':
            print("before optimizer: ", self.similarity_matrix)
            opt_obj = SimulatedAnnealingOptimizer(self.similarity_matrix, self.gamma, self.generate_edge_weights, self.objective_function, num_iterations, cooling_rate=.95)
        elif optimizer == 'particle_swarm':
            opt_obj = ParticleSwarmOptimizer(self.similarity_matrix, self.gamma, self.objective_function,  self.generate_edge_weights, 3, len(self.gamma), num_iterations)
        self.gamma = opt_obj.optimize()
        print("Optimized Gamma: ", self.gamma)

    def generate_optimal_edge_weights(self, num_iterations):
        '''
        Function to optimize gamma and set the resulting similarity matrix
        '''
        print("Generating Optimal Edge Weights")
        #self.similarity_matrix = self.generate_edge_weights(self.gamma)
        self.optimize_gamma('simulated_annealing', num_iterations)
        self.similarity_matrix = self.generate_edge_weights(self.gamma)
        #self.optimize_gamma('adam', num_iterations)
        #self.optimize_gamma('particle_swarm')

    def generate_edge_weights(self, gamma):
        '''
        Parallelization of edge weight computation using sparse matrix operations
        '''
        print("Generating Edge Weights")
        #print("Data Size: ", self.data.shape)
        #print("Graph Size: ", self.similarity_matrix.shape)
        #print("CURRENT SIM  MATR : ", self.similarity_matrix)
        #print("Current Gamma: ", gamma)
        #print("current DATA: ", self.data)
        #curr_sim_matr = sp.lil_matrix(self.similarity_matrix.shape)
        curr_sim_matr = np.zeros_like(self.similarity_matrix)
        d_adj_matr = cuda.to_device(self.similarity_matrix)
        d_data = cuda.to_device(self.data)
        d_gamma = cuda.to_device(gamma)
        d_pt_degrees = cuda.to_device(np.zeros_like(self.similarity_matrix[0]))
        d_sim_matr = cuda.to_device(curr_sim_matr)
        cuda.synchronize()
        edge_weight_computation[self.blocks, self.threads](d_adj_matr, d_data, d_gamma, d_pt_degrees, d_sim_matr)
            
        #print("RETURNED FROM EWC")
        
        cuda.synchronize()
        curr_sim_matr = d_sim_matr.copy_to_host()
        #print("EWC MATRIX: ", curr_sim_matr)
        curr_sim_matr = self.subtract_identity(curr_sim_matr)
        curr_sim_matr = np.nan_to_num(curr_sim_matr, nan=0, posinf=1e10, neginf=-1e10)
        print("FINISHED GENERATING EDGE WEIGHTS")
        return curr_sim_matr
        
    def subtract_identity(self, adj_matrix):
        '''
        Subtract matrix by identity for normalized symmetric laplacian
        '''
        np.fill_diagonal(adj_matrix, -1)
        return adj_matrix

    def unit_normalization(self, matrix):
        '''
        Normalize matrix to unit length
        '''
        norms = np.sqrt(np.sum(matrix.multiply(matrix), axis=1))
        norms = norms.A.flatten()  
        matrix = matrix.multiply(1 / norms[:, None])  
        return matrix

    def get_eigenvectors(self, num_components, min_variance):
        '''
        Cast similarity matrix to lower dimensional representation using PCA
        '''
        print("Computing Eigenvectors")
        pca = PCA()
        if num_components == 'lowest_var':
            pca.fit(np.asarray(self.similarity_matrix)) 
            expl_var = pca.explained_variance_ratio_
            cum_variance = expl_var.cumsum()
            num_components = (cum_variance <= min_variance).sum() + 1
        pca = PCA(n_components=num_components)
        pca_result = pca.fit_transform(self.similarity_matrix) #.toarray())
        pca_normalized = self.unit_normalization(sp.csr_matrix(pca_result))
        #pca_normalized = self.unit_normalization(pca_result)
        self.eigenvectors = pd.DataFrame(pca_normalized.toarray())
        print("Eigenvector Computation Complete")

@cuda.jit
def objective_computation(adj_matrix, data, gamma, degree_idx, approx_error, xi_reconstruction, out):
    row_len = len(adj_matrix[0])
    x, y = cuda.grid(2)
    total = 0.0
    for i in range(row_len):
        total += adj_matrix[x, i]
    degree_idx[x] = total 
    x, y = cuda.grid(2)
    total = 0.0
    for i in range(data.shape[1]):
        if abs(degree_idx[x]) > 1e-10:
            total += sqrt(abs(data[x, i] - ((adj_matrix[x, y] * data[y, i]) / degree_idx[x])**2))
        else:
            total += 0
    out[x] = total

@cuda.jit
def edge_weight_computation(curr_sim_mtrx, data, gamma, pt_degrees, out):
    
    row_len = len(gamma)
    x, y = cuda.grid(2)        
    total = 0.0

    for i in range(len(curr_sim_mtrx[0])):
        total += curr_sim_mtrx[x, i]

    pt_degrees[x] = total 

    x, y = cuda.grid(2)
    total = 0.0
    for i in range(row_len):
        if abs(gamma[i]) > 1e-5:
            total += (exp(-(((data[x, i] - data[y, i])**2) / gamma[i]**2)))
    
    norm_term = sqrt(abs(pt_degrees[x] * pt_degrees[y]))
    if abs(norm_term) > 1e-10:
        total /= -norm_term   
    else:
        total = 0

    out[x, y] = total

@cuda.jit
def gradient_computation(adj_matrix, data, gamma, pt_degrees, first_terms, second_terms, third_terms, out):    
    row_len = len(gamma)
    x, y = cuda.grid(2)
    total = 0.0
    gamma_cubed = 0.0
    for i in range(row_len):
        total += adj_matrix[i, y]
        if abs(gamma[i]) > 1e-7:
            gamma_cubed = gamma[i] ** (-3)
    pt_degrees[x] = total
    gamma[x] = gamma_cubed
    
    x, y = cuda.grid(2)
    first_term = 0.0
    second_term = 0.0
    third_term = 0.0
    for i in range(row_len):
        if i != y and abs(pt_degrees[x]) > 1e-10:
            normed_recons = (data[x, i] - (adj_matrix[x, y] * data[y, i])) / pt_degrees[x]
            first_term += normed_recons
            third_term += (2 * adj_matrix[x, i]) * (((data[x, i] - data[y, i])**2) * gamma[i]) * normed_recons
        if i != y:
            second_term += (2 * adj_matrix[x, i]) * (((data[x, i] - data[y, i])**2) * gamma[i]) * data[y, i]
      
    first_terms[x] = first_term
    second_terms[x] = second_term
    third_terms[x] = third_term
    
    for i in range(row_len):
        out[i] = first_terms[i] * (second_terms[i] - third_terms[i])
