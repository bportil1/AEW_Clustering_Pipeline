import numpy as np
import pandas as pd
from multiprocessing import Pool
from multiprocessing import cpu_count
from math import isclose
from math import ceil

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

import warnings

warnings.filterwarnings("ignore")

class aew():
    def __init__(self, data_graph, data, labels, precomputed_gamma=np.empty((0,0))):

        #self.similarity_matrix = np.zeros((data.shape[0], data.shape[0]))

        #print(data.shape[0])
        #print(len(self.similarity_matrix[0]))
        #identity = np.zeros((len(self.similarity_matrix[0]), len(self.similarity_matrix[0])))
        #identity_diag = np.diag(identity)
        #identity_diag_res = identity_diag + 1
        #np.fill_diagonal(self.similarity_matrix, identity_diag_res)

        #print(self.similarity_matrix[:5][:5])

        self.gamma = precomputed_gamma
        self.data_graph = data_graph

        #print(self.data_graph.toarray()[:5][:5])
        
        self.similarity_matrix = self.data_graph.toarray()
        identity = np.zeros((data.shape[0], data.shape[0]))
        identity_diag = np.diag(identity)
        identity_diag_res = identity_diag + 1
        np.fill_diagonal(self.similarity_matrix, identity_diag_res)

        #print(self.similarity_matrix[:5][:5])

        self.data = data
        self.labels = labels
        self.min_error = float('inf')
        self.eigenvectors = None

        if self.gamma.shape == (0,0):
            self.gamma = np.ones(self.data.loc[[0]].shape[1])

    def similarity_function(self, pt1_idx, pt2_idx): # -> Computation accuracy verified
        point1 = np.asarray(self.data.loc[[pt1_idx]])[0]
        point2 = np.asarray(self.data.loc[[pt2_idx]])[0]

        temp_res = 0

        deg_pt1 = np.sum(self.similarity_matrix[pt1_idx])
        deg_pt2 = np.sum(self.similarity_matrix[pt2_idx])
             
        #quared_gamma = np.where( np.abs(self.gamma) > .1e-5 ,  self.gamma**2, 0)

        #similarity_measure = np.sum(((point1 - point2)**2)/(squared_gamma))
        
        similarity_measure = np.sum(np.where(np.abs(self.gamma) > .1e-5, (((point1 - point2)**2)/(self.gamma)), 0))
        similarity_measure = np.exp(-similarity_measure, dtype=np.longdouble)

        #print("Sim meas: ", similarity_measure)

        degree_normalization_term = np.sqrt(np.abs(deg_pt1 * deg_pt2))

        #print("deg norm _term: ", degree_normalization_term)

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
            #print("Degree of idx: ", dii)
            #print("Sim matr row of idx: ", self.similarity_matrix[]
            #print("vals: ", self.similarity_matrix[idx][0], " ", np.asarray(self.data.loc[[0]])[0], len(self.similarity_matrix[idx]), np.asarray(self.data.loc[[idx]]) )
            xi_reconstruction = np.sum([self.similarity_matrix[idx][y]*np.asarray(self.data.loc[[y]])[0] for y in range(len(self.similarity_matrix[idx])) if idx != y], 0)
            #print("First rec: ", xi_reconstruction)
            if dii != 0 and not isclose(dii, 0, abs_tol=1e-100):
                xi_reconstruction = np.divide(xi_reconstruction, dii, casting='unsafe', dtype=np.longdouble)
                #print("Rec in if cond: ", xi_reconstruction)
                #print("Arr: ", np.asarray(self.data.loc[[idx]])[:1]) 
                first_term = np.divide((np.asarray(self.data.loc[[idx]])[0] - xi_reconstruction), dii, casting='unsafe', dtype=np.longdouble)
                #print("First term in if: ", first_term)
            else:
                first_term  = np.zeros_like(xi_reconstruction)
                #print("First term in else: ", first_term)

                xi_reconstruction  = np.zeros_like(xi_reconstruction)
                #print("Rec in else: ", xi_reconstruction)

            

            cubed_gamma = np.where( np.abs(self.gamma) > .1e-7 ,  self.gamma**(-3), 0)

            #print("gamma: ", self.gamma)

            #print("cubed_gamma: ", cubed_gamma)

            dw_dgamma = np.sum([(2*self.similarity_matrix[idx][y]* (((np.asarray(self.data.loc[[idx]])[0] - np.asarray(self.data.loc[[y]])[0])**2)*cubed_gamma)*np.asarray(self.data.loc[[y]])[0]) for y in range(self.data.shape[0]) if idx != y])
            #print("W der term: ", dw_dgamma)
            #print("Arr: ", np.asarray(self.data.loc[[idx]])[:5])
            dD_dgamma = np.sum([(2*self.similarity_matrix[idx][y]* (((np.asarray(self.data.loc[[idx]])[0] - np.asarray(self.data.loc[[y]])[0])**2)*cubed_gamma)*xi_reconstruction) for y in range(self.data.shape[0]) if idx != y])
            #print("D der term: ", dD_dgamma)
            gradient = gradient + (first_term * (dw_dgamma - dD_dgamma))
            #print("fin gradient: ", gradient)
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

        gradient = np.zeros(self.data.loc[[0]].shape[1])

        for grad in gradients:
            #print(grad)
            #print(gradient)
            gradient = gradient + grad
            #print(gradient)

        return gradient

    def gradient_descent(self, learning_rate, num_iterations, tol):
        print("Beggining Gradient Descent")
        last_error = -9999999
        min_error = float('inf')
        min_gamma = []
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
                    learning_rate -= .00001
                else:
                    learning_rate /= (.00001)

            elif last_error - curr_error < 100:
                learning_rate += .000001
                #learning_rate *= (1.00002)
           
            elif last_error - curr_error > 100 and i < 5:
                learning_rate *= (1.02)

            elif last_error > 500:
                learning_rate *= (1.03)



            last_error = curr_error
            
            if curr_error <= min_error and i != 0:
                min_error = curr_error
                min_gamma = self.gamma


            print("Gamma: ", self.gamma)
            self.gamma = (self.gamma + (gradient * learning_rate))
            print("Updated Gamma: ", self.gamma)
            self.generate_edge_weights()
            print("Updated Learning Rate: ", learning_rate)

        self.gamma = min_gamma
        print("Updated Final Error: ", min_error)
        print("Updated Final Gamma: ", self.gamma)

        self.generate_edge_weights()
        print("Adj Matr: ", self.similarity_matrix[300][:10])

        print("Adj Matr Max: ", np.amax(self.similarity_matrix))

        print("Adj Matr Min: ", np.amin(self.similarity_matrix))

        print("Completed Gradient Descent")

    def generate_optimal_edge_weights(self, num_iterations):
        print("Generating Optimal Edge Weights")

        self.gradient_descent(.000002, num_iterations, .01)

        #self.gradient_descent(50, num_iterations, .01)

        self.generate_edge_weights()
    
    def edge_weight_computation(self, section):

        res = []

        for idx in section:
            point = slice(self.data_graph.indptr[idx], self.data_graph.indptr[idx+1])
            for vertex in self.data_graph.indices[point]:
                res.append((idx, vertex, self.similarity_function(idx, vertex)))

        return res

    def generate_edge_weights(self):
        print("Generating Edge Weights")

        #print(self.similarity_matrix)

        split_data = self.split(range(self.data.shape[0]), cpu_count())

        #print(split_data)

        with Pool(processes=cpu_count()) as pool:
            edge_weight_res = [pool.apply_async(self.edge_weight_computation, (section, )) for section in split_data]

            edge_weights = [edge_weight.get() for edge_weight in edge_weight_res]

        for section in edge_weights:
            for weight in section:
                #print(weight)
                if weight[0] != weight[1]: #and weight[2] > 1*(10*-20):
                    self.similarity_matrix[weight[0]][weight[1]] = weight[2]
                    self.similarity_matrix[weight[1]][weight[0]] = weight[2]

        #print(len(self.similarity_matrix))
        #print(len(self.similarity_matrix[0]))

        #print(self.similarity_matrix)

        #print("Diagonal 1: ", np.diagonal(self.similarity_matrix))


        #self.laplacian_normalization()

        #print(self.similarity_matrix)

        #print(np.identity(len(self.similarity_matrix[0])))

        #print("Diagonal 2: ", np.diagonal(self.similarity_matrix))

        self.subtract_identity()

        #print(self.similarity_matrix[1])

        #print(self.similarity_matrix[1][1])

        #print("Diagonal 3: ", np.diagonal(self.similarity_matrix))

        #self.rewrite_edges()

        #self.remove_disconnections()

        #self.scale_matrix()

        #print(np.any(np.sum(self.similarity_matrix, axis=1) < 0))

        #print(np.any(np.sum(self.similarity_matrix, axis=0) < 0))
        
        #print(np.any(np.sum(self.similarity_matrix, axis=1) == 0))

        #print(np.any(np.sum(self.similarity_matrix, axis=0) == 0))

        self.eigenvectors = self.get_eigenvectors()

        #self.data_graph = self.data_graph.toarray()

        print("Edge Weight Generation Complete")


    def scale_matrix(self):
        scaler = MinMaxScaler()

        self.similarity_matrix = scaler.fit_transform(self.similarity_matrix)

    def rewrite_edges(self):

        rows, cols = self.data_graph.nonzero()

        for idx in range(len(rows)):
            row = rows[idx]
            col = cols[idx]
            self.data_graph[row, col] = self.similarity_matrix[row, col]

    def subtract_identity(self):
        identity = np.zeros((len(self.similarity_matrix[0]), len(self.similarity_matrix[0]))) 
        identity_diag = np.diag(identity)
        identity_diag_res = identity_diag + 2 
        np.fill_diagonal(identity, identity_diag_res) 
        self.similarity_matrix = identity - self.similarity_matrix

    def remove_disconnections(self):
        empty_rows = np.all(self.similarity_matrix == 0, axis = 1)
        empty_cols = np.all(self.similarity_matrix == 0, axis = 0)
        #print(self.similarity_matrix[:5])
        self.similarity_matrix = self.similarity_matrix[~empty_rows, :][:, ~empty_cols]
        #print(self.similarity_matrix[:5])
        self.labels = self.labels.loc[~empty_rows]

    def unit_normalization(self, matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix/norms

    def get_eigenvectors(self):
        pca = PCA()

        pca.fit(self.similarity_matrix)

        expl_var = pca.explained_variance_ratio_
    
        cum_variance = expl_var.cumsum()

        desired_variance = 0.90

        num_components = ( cum_variance <= desired_variance).sum() + 1

        #selected_columns = self.similarity_matrix.columns[:num_components]

        #selected_columns = self.similarity_matrix[:,:num_components]

        #print(num_components)

        #print(selected_columns)

        #print(pca.fit(self.similarity_matrix).explained_variance_ratio_[:5])

        #print(pca.fit(self.similarity_matrix).singular_values_)

        pca = PCA(n_components=num_components)

        pca = pca.fit_transform(self.similarity_matrix)

        pca = self.unit_normalization(pca.real)

        '''
        pca2 = PCA(n_components=num_components)

        pca2 = pca2.fit(self.similarity_matrix)

        loadings = pca2.components_

        df = pd.DataFrame(self.similarity_matrix)

        # Create a DataFrame to hold the loadings
        loadings_df = pd.DataFrame(loadings.T, columns=[f"PC{i+1}" for i in range(num_components)], index=df.columns)

        # Get the absolute values of the loadings to determine the most important features
        abs_loadings_df = loadings_df.abs()

        # For each principal component, get the top contributing features
        top_features = [abs_loadings_df.nlargest(3, f"PC{i+1}") for i in range(num_components)]

        #Print the top features for each principal component
        for i, top_features in enumerate(top_features):
            print(f"Top features for PC{i+1}:")
            print(top_features)
        #idx = eigenvalues.argsort()[-num_cols:][::-1]
        '''
        #idx = eigenvalues.argsort()[::-1]

        #print(eigenvectors[:, idx])

        #print(eigenvectors[:, idx])

        #return eigenvectors[:, idx].real

        #print(pca)

        #print(pca.real)

        return pca

