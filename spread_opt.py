import numpy as np
import pandas as pd
from multiprocessing import Pool
from multiprocessing import cpu_count
from math import isclose
from math import ceil

from copy import deepcopy

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

import warnings

warnings.filterwarnings("ignore")

class aew():
    def __init__(self, similarity_matrix, data, labels, gamma_init=None):

        #### DATA HOLDING OBJECTS
        self.data = data
        self.labels = labels
        self.eigenvectors = None
        self.gamma = gamma_initializer(gamma_init)
        self.similarity_matrix = correct_similarity_matrix_diag(similarity_matrix)

    def correct_similarity_matrix_diag(self, similarity_matrix):
        identity = np.zeros((self.data.shape[0], self.data.shape[0]))
        identity_diag = np.diag(identity)
        identity_diag_res = identity_diag + 1
        np.fill_diagonal(similarity_matrix, identity_diag_res)

        return similarity_matrix

    def gamma_initializer(self, gamma_init=None):
        if gamma_init == None:
            self.gamma = np.ones(self.data.loc[[0]].shape[1])
        elif gamma_init == 'var':
            self.gamma = np.var(self.data, axis=0).values
        elif gamma_init == 'random_int':
            self.gamma = np.random.randint(0, 1000, (1, 41))
        elif gamma_init == 'random_float':
            rng = np.random.default_rng()
            self.gamma = rng.random(size=(1, 41)) 

    def similarity_function(self, pt1_idx, pt2_idx): # -> Computation accuracy verified
        point1 = np.asarray(self.data.loc[[pt1_idx]])[0]
        point2 = np.asarray(self.data.loc[[pt2_idx]])[0]

        temp_res = 0

        deg_pt1 = np.sum(self.similarity_matrix[pt1_idx])
        deg_pt2 = np.sum(self.similarity_matrix[pt2_idx])
             
        similarity_measure = np.sum(np.where(np.abs(self.gamma) > .1e-5, (((point1 - point2)**2)/(self.gamma)), 0))
        similarity_measure = np.exp(-similarity_measure, dtype=np.longdouble)

        degree_normalization_term = np.sqrt(np.abs(deg_pt1 * deg_pt2))

        ##May need to relax this bound
        if degree_normalization_term != 0 and not isclose(degree_normalization_term, 0, abs_tol=1e-100):
            return similarity_measure / degree_normalization_term
        else:
            return 0

    def objective_computation(self, section):
        approx_error = 0
        for idx in section:
            degree_idx = np.sum(self.similarity_matrix[idx])
            xi_reconstruction = np.sum([self.similarity_matrix[idx][y]*np.asarray(self.data.loc[[y]])[0] for y in range(len(self.similarity_matrix[idx])) if idx != y], 0)            

            if degree_idx != 0 and not isclose(degree_idx, 0, abs_tol=1e-100):
                xi_reconstruction /= degree_idx
                xi_reconstruction = xi_reconstruction[0]
            else:
                xi_reconstruction = np.zeros(len(self.gamma))

        return np.sum((np.asarray(self.data.loc[[idx]])[0] - xi_reconstruction)**2)

    def objective_function(self):
        split_data = self.split(range(self.data.shape[0]), cpu_count())
        with Pool(processes=cpu_count()) as pool:
            errors = [pool.apply_async(self.objective_computation, (section, )) \
                                                                 for section in split_data]

            error = [error.get() for error in errors]
        return np.sum(error)

    def gradient_computation(self, section):
        gradient = np.zeros(len(self.gamma))
        for idx in section:
            dii = np.sum(self.similarity_matrix[idx])
            xi_reconstruction = np.sum([self.similarity_matrix[idx][y]*np.asarray(self.data.loc[[y]])[0] for y in range(len(self.similarity_matrix[idx])) if idx != y], 0)
            if dii != 0 and not isclose(dii, 0, abs_tol=1e-100):
                xi_reconstruction = np.divide(xi_reconstruction, dii, casting='unsafe', dtype=np.longdouble)
                first_term = np.divide((np.asarray(self.data.loc[[idx]])[0] - xi_reconstruction), dii, casting='unsafe', dtype=np.longdouble)
            else:
                first_term  = np.zeros_like(xi_reconstruction)
                xi_reconstruction  = np.zeros_like(xi_reconstruction)
            cubed_gamma = np.where( np.abs(self.gamma) > .1e-7 ,  self.gamma**(-3), 0)
            dw_dgamma = np.sum([(2*self.similarity_matrix[idx][y]* (((np.asarray(self.data.loc[[idx]])[0] - np.asarray(self.data.loc[[y]])[0])**2)*cubed_gamma)*np.asarray(self.data.loc[[y]])[0]) for y in range(self.data.shape[0]) if idx != y])
            dD_dgamma = np.sum([(2*self.similarity_matrix[idx][y]* (((np.asarray(self.data.loc[[idx]])[0] - np.asarray(self.data.loc[[y]])[0])**2)*cubed_gamma)*xi_reconstruction) for y in range(self.data.shape[0]) if idx != y])
            gradient = gradient + (first_term * (dw_dgamma - dD_dgamma))
        return gradient

    def split(self, a, n):
        k, m = divmod(len(a), n)
        return [a[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]

    def gradient_function(self):
        gradient = []
    
        split_data = self.split(range(self.data.shape[0]), cpu_count())
        
        with Pool(processes=cpu_count()) as pool:
            gradients = [pool.apply_async(self.gradient_computation, (section, )) \
                                                                 for section in split_data]

            gradients = [gradient.get() for gradient in gradients]

        #gradient = np.zeros(self.data.loc[[0]].shape[1])

        #for grad in gradients:
            #gradient = gradient + grad
        return np.sum(gradients)

    def gradient_descent(self, learning_rate, num_iterations, tol):
        print("Beggining Gradient Descent")
        last_error = -9999999
        min_error = float('inf')
        min_gamma = []
        min_sim_matrix = []
        last_change = 0
        # Perform the gradient descent iterations
        for i in range(num_iterations):
            print("Current Iteration: ", str(i+1))
            print("Computing Gradient")
            gradient = self.gradient_function()
            print("Current Gradient: ", gradient)
            print("Computing Error")
            curr_error = self.objective_function()
            print("Current Error: ", curr_error)
                    

            gradient = np.where(gradient > 0, gradient * -1, gradient)

            print(gradient)

            if curr_error < tol:
                break
                
            elif last_error - curr_error < -100:   #last_error < curr_error: 
                if learning_rate > .000000001:
                    learning_rate -= .0000001
                else:
                    learning_rate /= (.000001)

            elif last_error - curr_error < 100:
                learning_rate += .000001
                #learning_rate *= (1.00002)
           
            elif last_error - curr_error > 100 and i < 5:
                learning_rate *= (1.02)

            elif last_error > 500:
                learning_rate *= (1.03)

            last_error = curr_error
            
            last_change += 1 

            if curr_error <= min_error and i != 0:
                min_error = curr_error
                min_gamma = self.gamma
                min_sim_matrix = deepcopy(self.similarity_matrix) 
                last_change = 0

            print("Gamma: ", self.gamma)
            self.gamma = (self.gamma + (gradient * learning_rate))
            print("Updated Gamma: ", self.gamma)
            self.generate_edge_weights()
            print("Updated Learning Rate: ", learning_rate)

        self.gamma = min_gamma
        print("Updated Final Error: ", min_error)
        print("Updated Final Gamma: ", self.gamma)

        self.generate_edge_weights
        self.similarity_matrix = min_sim_matrix
        print("Completed Gradient Descent")

    def generate_optimal_edge_weights(self, num_iterations):
        print("Generating Optimal Edge Weights")

        self.gradient_descent(.000002, num_iterations, .01)

        self.generate_edge_weights()
    

    #### still working with graph change to similarity matrix
    def edge_weight_computation(self, section):

        res = []

        for idx in section:
            point = slice(self.data_graph.indptr[idx], self.data_graph.indptr[idx+1])
            for vertex in self.data_graph.indices[point]:
                res.append((idx, vertex, self.similarity_function(idx, vertex)))

        return res

    def generate_edge_weights(self):
        print("Generating Edge Weights")

        split_data = self.split(range(self.data.shape[0]), cpu_count())

        with Pool(processes=cpu_count()) as pool:
            edge_weight_res = [pool.apply_async(self.edge_weight_computation, (section, )) for section in split_data]

            edge_weights = [edge_weight.get() for edge_weight in edge_weight_res]

        for section in edge_weights:
            for weight in section:
                if weight[0] != weight[1]:
                    self.similarity_matrix[weight[0]][weight[1]] = weight[2]
                    self.similarity_matrix[weight[1]][weight[0]] = weight[2]

        self.subtract_identity()

        #self.remove_disconnections()

        self.eigenvectors = self.get_eigenvectors()

        print("Edge Weight Generation Complete")


    def scale_matrix(self):
        scaler = MinMaxScaler()

        self.similarity_matrix = scaler.fit_transform(self.similarity_matrix)

    def subtract_identity(self):
        identity = np.zeros((len(self.similarity_matrix[0]), len(self.similarity_matrix[0]))) 
        identity_diag = np.diag(identity)
        identity_diag_res = identity_diag + 2 
        np.fill_diagonal(identity, identity_diag_res) 
        self.similarity_matrix = identity - self.similarity_matrix

    def remove_disconnections(self):
        empty_rows = np.all(self.similarity_matrix == 0, axis = 1)
        empty_cols = np.all(self.similarity_matrix == 0, axis = 0)
        self.similarity_matrix = self.similarity_matrix[~empty_rows, :][:, ~empty_cols]
        self.labels = self.labels.loc[~empty_rows]

    def unit_normalization(self, matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix/norms

    def get_eigenvectors(self, num_components='lowest_var', min_variance):
        pca = PCA()

        if num_components == 'lowest_var':

            pca.fit(self.similarity_matrix)

            expl_var = pca.explained_variance_ratio_
    
            cum_variance = expl_var.cumsum()

            num_components = ( cum_variance <= min_variance).sum() + 1

        pca = PCA(n_components=num_components)

        pca = pca.fit_transform(self.similarity_matrix)

        pca = self.unit_normalization(pca.real)

        return pca

