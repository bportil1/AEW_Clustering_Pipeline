import numpy as np
import pandas as pd
from multiprocessing import Pool
from multiprocessing import cpu_count
from math import isclose
from sklearn.decomposition import PCA
import scipy.sparse as sp
import warnings

from optimizers import *

warnings.filterwarnings("ignore")

class aew():
    def __init__(self, similarity_matrix, data, labels, gamma_init=None):
        '''
        Initialize class attributes
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
        # Ensure similarity_matrix is sparse (CSR format)
        similarity_matrix = sp.csr_matrix(similarity_matrix)
        
        # Set diagonal to 1 (or any other value you prefer)
        similarity_matrix.setdiag(1)
        
        return similarity_matrix

    def gamma_initializer(self, gamma_init=None):
        '''
        Initialize gamma by user chosen option
        '''
        if gamma_init is None:
            gamma = np.ones(self.data.loc[[0]].shape[1])
        elif gamma_init == 'var':
            gamma = np.var(self.data, axis=0).values
        elif gamma_init == 'random_int':
            gamma = np.random.randint(0, 1000, (1, 41))
        elif gamma_init == 'random_float':
            rng = np.random.default_rng()
            gamma = rng.random(size=(1, 41)) 
        return gamma

    def similarity_function(self, pt1_idx, pt2_idx, gamma):         
        '''
        Compute similarity between two points (adjusted for sparse matrix)
        '''
        point1 = np.asarray(self.data.loc[[pt1_idx]])[0]
        point2 = np.asarray(self.data.loc[[pt2_idx]])[0]
        
        deg_pt1 = np.sum(self.similarity_matrix[pt1_idx].toarray())
        deg_pt2 = np.sum(self.similarity_matrix[pt2_idx].toarray())
        
        similarity_measure = np.sum(np.where(np.abs(gamma) > 1e-5, (((point1 - point2)**2)/(gamma**2)), 0))
        similarity_measure = np.exp(-similarity_measure, dtype=np.longdouble)
        
        degree_normalization_term = np.sqrt(np.abs(deg_pt1 * deg_pt2))
        if degree_normalization_term != 0 and not isclose(degree_normalization_term, 0, abs_tol=1e-100):
            return similarity_measure / degree_normalization_term
        else:
            return 0

    def objective_computation(self, section, adj_matrix, gamma):
        '''
        Compute reconstruction error (adjusted for sparse matrix)
        '''        
        approx_error = 0
        for idx in section:
            degree_idx = np.sum(adj_matrix[idx].toarray())
            xi_reconstruction = np.sum([adj_matrix[idx, y]*np.asarray(self.data.loc[[y]])[0] 
                                        for y in range(len(adj_matrix[idx].toarray()[0])) if idx != y], 0)            
            if degree_idx != 0 and not isclose(degree_idx, 0, abs_tol=1e-100):
                xi_reconstruction /= degree_idx
                xi_reconstruction = xi_reconstruction[0]
            else:
                xi_reconstruction = np.zeros(len(self.gamma))
        return np.sum((np.asarray(self.data.loc[[idx]])[0] - xi_reconstruction)**2)

    def objective_function(self, adj_matr, gamma):
        '''
        Parallelization of error computation (adjusted for sparse matrix)
        '''
        split_data = self.split(range(self.data.shape[0]), cpu_count())
        with Pool(processes=cpu_count()) as pool:
            errors = [pool.apply_async(self.objective_computation, (section, adj_matr, gamma)) \
                                                                 for section in split_data]
            error = [error.get() for error in errors]
        return np.sum(error)

    def gradient_computation(self, section, similarity_matrix, gamma):
        '''
        Compute gradient (adjusted for sparse matrix)
        '''        
        gradient = np.zeros(len(gamma))
        for idx in section:
            dii = np.sum(similarity_matrix[idx].toarray())
            xi_reconstruction = np.sum([similarity_matrix[idx, y]*np.asarray(self.data.loc[[y]])[0] 
                                        for y in range(len(similarity_matrix[idx].toarray()[0])) if idx != y], 0)
            if dii != 0 and not isclose(dii, 0, abs_tol=1e-100):
                xi_reconstruction = np.divide(xi_reconstruction, dii, casting='unsafe', dtype=np.longdouble)
                first_term = np.divide((np.asarray(self.data.loc[[idx]])[0] - xi_reconstruction), dii, casting='unsafe', dtype=np.longdouble)
            else:
                first_term  = np.zeros_like(xi_reconstruction)
                xi_reconstruction  = np.zeros_like(xi_reconstruction)
            
            cubed_gamma = np.where( np.abs(gamma) > 1e-7 ,  gamma**(-3), 0)
            dw_dgamma = np.sum([(2*similarity_matrix[idx, y]* (((np.asarray(self.data.loc[[idx]])[0] - np.asarray(self.data.loc[[y]])[0])**2)*cubed_gamma)*np.asarray(self.data.loc[[y]])[0]) for y in range(self.data.shape[0]) if idx != y])
            dD_dgamma = np.sum([(2*similarity_matrix[idx, y]* (((np.asarray(self.data.loc[[idx]])[0] - np.asarray(self.data.loc[[y]])[0])**2)*cubed_gamma)*xi_reconstruction) for y in range(self.data.shape[0]) if idx != y])
            gradient = gradient + (first_term * (dw_dgamma - dD_dgamma))
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
        Parallelization of gradient computation (adjusted for sparse matrix)
        '''        
        gradient = []
        split_data = self.split(range(self.data.shape[0]), cpu_count())
        with Pool(processes=cpu_count()) as pool:
            gradients = [pool.apply_async(self.gradient_computation, (section, similarity_matrix, gamma)) \
                                                                 for section in split_data]
            gradients = [gradient.get() for gradient in gradients]
        return np.sum(gradients, axis=0)

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
        Compute edge weights (adjusted for sparse matrix)
        '''
        res = []
        for idx in section:
            for vertex in range(self.data.shape[0]):
                if vertex != idx:
                    res.append((idx, vertex, self.similarity_function(idx, vertex, gamma)))
        return res

    def generate_edge_weights(self, gamma):
        '''
        Parallelization of edge weight computation (adjusted for sparse matrix)
        '''
        print("Generating Edge Weights")
        curr_sim_matr = sp.csr_matrix(self.correct_similarity_matrix_diag(np.zeros_like(self.similarity_matrix)))
        split_data = self.split(range(self.data.shape[0]), cpu_count())
        with Pool(processes=cpu_count()) as pool:
            edge_weight_res = [pool.apply_async(self.edge_weight_computation

