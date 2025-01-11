import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from math import isclose
from sklearn.decomposition import PCA
import scipy.sparse as sp
import warnings
from math import sqrt, exp

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
        self.similarity_matrix = self.correct_similarity_matrix_diag(similarity_matrix)
        self.threads = (16, 16)
        self.blocks = (len(self.similarity_matrix[0]) + self.threads[0] - 1) // self.threads[0], \
                      (len(self.similarity_matrix[0]) + self.threads[1] - 1) // self.threads[1]

    #@vectorize
    def correct_similarity_matrix_diag(self, similarity_matrix):
        '''
        Correct diagonals of precomputed similarity matrix, if necessary
        '''
        if not sp.issparse(similarity_matrix):
            similarity_matrix = sp.csr_matrix(similarity_matrix)
        identity = sp.lil_matrix((self.data.shape[0], self.data.shape[0]))
        identity.setdiag(np.ones(self.data.shape[0]))
        similarity_matrix = similarity_matrix.tolil()  # Convert to LIL format for easy manipulation
        #similarity_matrix.setdiag(np.ones(self.data.shape[0]))
        #print("Sim matrix after correction: ", similarity_matrix.toarray())
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

        #print("adj_matr type: ", self.data.shape[0])
        #print("Adjacency Matrix: ", self.data.to_numpy())
        matr_d = cuda.to_device(adj_matr)
        data_d = cuda.to_device(self.data.to_numpy())
        gamma_d = cuda.to_device(gamma)
        degree_d = cuda.device_array(shape=(self.data.shape[0],), dtype=np.float32)
        error_arr_d = cuda.device_array(shape=(self.data.shape[0], self.data.shape[0]), dtype=np.float32)
        xi_reconstruction = cuda.device_array(shape=(self.data.shape[0],), dtype=np.float32)
        out_d = cuda.device_array(shape=(self.data.shape[0],), dtype=np.float32)
        
        #print("BEFORE CUDA KERNEL CALL")
        #print("NUMBER OF BLOCKS: ", self.blocks)
        #print("NUMBER OF THREADS: ", self.threads) 

        #print(f"Free memory before operation: {cuda.current_context().get_memory_info()[0] / 1e9} GB")

        cuda.synchronize()
        objective_computation[self.blocks, self.threads](matr_d, data_d, gamma_d, degree_d, error_arr_d, xi_reconstruction, out_d)
        #print("AFTER CUDA KERNEL CALL")
        
        #print(f"Free memory after synchronization: {cuda.current_context().get_memory_info()[0] / 1e9} GB")
        
        cuda.synchronize()
        error = out_d.copy_to_host()
        error = np.nan_to_num(error, nan=0, posinf=1e10, neginf=-1e10)
    
        print("Error Array: ", error)
        #print("Final Error: ", np.sum(error))

        return np.sum(error)

    #@vectorize
    def gradient_computation(self, section, similarity_matrix, gamma):
        '''
        Compute gradient for a section, adjusted for sparse matrix usage
        '''
        gradient = np.zeros(len(gamma))
        for idx in section:
            dii = np.sum(similarity_matrix[idx, :])
            xi_reconstruction = np.sum([similarity_matrix[idx, y] * np.asarray(self.data.loc[[y]])[0] for y in range(len(similarity_matrix[idx])) if idx != y], axis=0)
            if dii != 0 and not isclose(dii, 0, abs_tol=1e-100):
                xi_reconstruction /= dii
                first_term = (np.asarray(self.data.loc[[idx]])[0] - xi_reconstruction) / dii
            else:
                first_term = np.zeros_like(xi_reconstruction)
            cubed_gamma = np.where(np.abs(gamma) > 1e-7, gamma**(-3), 0)
            dw_dgamma = np.sum([(2 * similarity_matrix[idx, y] * (((np.asarray(self.data.loc[[idx]])[0] - np.asarray(self.data.loc[[y]])[0])**2) * cubed_gamma) * np.asarray(self.data.loc[[y]])[0]) for y in range(self.data.shape[0]) if idx != y])
            dD_dgamma = np.sum([(2 * similarity_matrix[idx, y] * (((np.asarray(self.data.loc[[idx]])[0] - np.asarray(self.data.loc[[y]])[0])**2) * cubed_gamma) * xi_reconstruction) for y in range(self.data.shape[0]) if idx != y])
            gradient += first_term * (dw_dgamma - dD_dgamma)
            gradient = np.nan_to_num(gradient, nan=0)
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
        return np.sum(gradients, axis=0)

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
        self.similarity_matrix = self.generate_edge_weights(self.gamma)
        self.optimize_gamma('simulated_annealing', num_iterations)

    def generate_edge_weights(self, gamma):
        '''
        Parallelization of edge weight computation using sparse matrix operations
        '''
        print("Generating Edge Weights")
        print("Data Size: ", self.data.shape)
        print("Graph Size: ", self.similarity_matrix.shape)
        #curr_sim_matr = sp.lil_matrix(self.similarity_matrix.shape)
        curr_sim_matr = np.zeros_like(self.similarity_matrix)
        print("New Sim Matr Size: ", curr_sim_matr.shape)
        #mm_file = './mmap_file'
        #curr_sim_matr = np.memmap(mm_file + 'curr_sim_matr', dtype='float32', mode='w+', shape=curr_sim_matr.shape)
        
        d_adj_matr = cuda.to_device(self.similarity_matrix)
        d_data = cuda.to_device(self.data)
        d_gamma = cuda.to_device(gamma)
        d_pt_degrees = cuda.to_device(np.zeros_like(gamma))
        d_sim_matr = cuda.to_device(curr_sim_matr)

        edge_weight_computation[self.blocks, self.threads](d_adj_matr, d_data, d_gamma, d_pt_degrees, d_sim_matr)
        
        curr_sim_matr = d_sim_matr.copy_to_host()

        curr_sim_matr = self.subtract_identity(curr_sim_matr)

        return curr_sim_matr
        
    def subtract_identity(self, adj_matrix):
        '''
        Subtract matrix by identity for normalized symmetric laplacian
        '''
        #identity = sp.lil_matrix(adj_matrix.shape)
        #identity.setdiag(2)  # Set diagonal elements to 2
        #adj_matrix = identity - adj_matrix
        np.fill_diagonal(adj_matrix, -1)
        return adj_matrix

    def unit_normalization(self, matrix):
        '''
        Normalize matrix to unit length
        '''
        norms = np.sqrt(np.sum(matrix.multiply(matrix), axis=1))
        norms = norms.A.flatten()  # Convert from matrix to 1D array
        matrix = matrix.multiply(1 / norms[:, None])  # Normalize each row
        return matrix

    def get_eigenvectors(self, num_components, min_variance):
        '''
        Cast similarity matrix to lower dimensional representation using PCA
        '''
        print("Computing Eigenvectors")
        pca = PCA()
        if num_components == 'lowest_var':
            #pca.fit(self.similarity_matrix.toarray())  # Convert sparse to dense for PCA fitting
            pca.fit(np.asarray(self.similarity_matrix)) 
            expl_var = pca.explained_variance_ratio_
            cum_variance = expl_var.cumsum()
            num_components = (cum_variance <= min_variance).sum() + 1
        pca = PCA(n_components=num_components)
        pca_result = pca.fit_transform(self.similarity_matrix.toarray())
        #pca_result = pca.fit_transform(np.asarray(self.similarity_matrix))
        pca_normalized = self.unit_normalization(sp.csr_matrix(pca_result))
        #print(type(pca_normalized))
        self.eigenvectors = pd.DataFrame(pca_normalized.toarray())
        #print(self.eigenvectors)
        #self.eigenvectors = pca_normalized.toarray()
        print("Eigenvector Computation Complete")

#(matr_d, data_d, gamma_d, degree_d, xi_reconstruction, out_d)

@cuda.jit
def objective_computation(adj_matrix, data, gamma, degree_idx, approx_error, xi_reconstruction, out):
    '''
    Compute reconstruction error for a section, adjusted for sparse matrix usage
    '''
    row_len = len(adj_matrix[0])

    #### Accumulate point degree
    #idx = cuda.grid(1)
    #stride = cuda.gridsize(1)
   
    x, y = cuda.grid(2)

    total = 0.0
    for i in range(row_len):
        total += adj_matrix[x, i]
        
    degree_idx[x] = total 
    
    #cuda.syncthreads()
    
    #### Approx error no nXn to hold the error for each combination of points
    #### Accumulate errors here
    
    x, y = cuda.grid(2)

    total = 0.0

    for i in range(data.shape[1]):
        total += adj_matrix[x, y] * data[y, i]

    approx_error[x, y] = total

    #### SUM ERRROS FOR EACH POINT
    x, y = cuda.grid(2)

    total = 0.0
    
    for i in range(row_len):
        total += approx_error[i, y]
    
    if abs(degree_idx[x]) > 1e-20:
        xi_reconstruction[x] = total / degree_idx[x]

    
    x, y = cuda.grid(2)

    total = 0.0

    for i in range(row_len):
        total += data[] xi_reconstruction[x]   

    

    #cuda.syncthreads()

@cuda.jit
def edge_weight_computation(curr_sim_mtrx, data, gamma, pt_degrees, out):
    
    row_len = len(gamma)
    
    #### Need Similarity between all points
    x, y = cuda.grid(2)        
    
    #### COLLECT  ALL DEGREES
  
    total = 0.0
    
    for i in range(row_len):
        total += curr_sim_mtrx[i, y]
    
    pt_degrees[x] = total 
    
    #### CALC SIMILARITY MEASURE FOR ALL PAIRS OF POINTS
    
    x, y = cuda.grid(2)
    
    total = 0.0
    
    for i in range(row_len):
        norm_term = sqrt(abs(pt_degrees[x] * pt_degrees[y]))
        if abs(gamma[i]) > 1e-5 and abs(norm_term) > 1e-5:
            total += (exp(-(((data[x, i] - data[y, i])**2) / gamma[i]))) / norm_term 
            
    out[x, y] = total
    
    '''
    #@vectorize
    def similarity_function(self, pt1_idx, pt2_idx, gamma):
        
        Compute similarity between two points using sparse matrix operations
        
        point1 = np.asarray(self.data.loc[[pt1_idx]])[0]
        point2 = np.asarray(self.data.loc[[pt2_idx]])[0]
        deg_pt1 = np.sum(self.similarity_matrix[pt1_idx, :])
        deg_pt2 = np.sum(self.similarity_matrix[pt2_idx, :])
        
        # Compute squared difference and apply gamma
        similarity_measure = np.sum(np.where(np.abs(gamma) > 1e-5, (((point1 - point2)**2) / (gamma**2)), 0))
        similarity_measure = np.exp(-similarity_measure)
        
        degree_normalization_term = np.sqrt(np.abs(deg_pt1 * deg_pt2))
        
        if degree_normalization_term != 0 and not isclose(degree_normalization_term, 0, abs_tol=1e-100):
            return similarity_measure / degree_normalization_term
        else:
            return 0

    def edge_weight_computation(self, section, gamma):
    
        #Compute edge weights for a section using sparse matrix operations
        
        res = []
        for idx in section:
            for vertex in range(self.data.shape[0]):
                if vertex != idx:
                    res.append((idx, vertex, self.similarity_function(idx, vertex, gamma)))
        return res
    '''    

