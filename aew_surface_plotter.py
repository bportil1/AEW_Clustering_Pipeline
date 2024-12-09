import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import isclose
import math
import itertools
from multiprocessing import Pool
from multiprocessing import cpu_count
from preprocessing_utils import *
from aew_sm import *
import plotly.express as px
import plotly.graph_objects as go

# Define the class that contains the objective computation function
class OptimizationFunction:
    def __init__(self, data=None, similarity_matrix=None, gamma=None):
        self.data = data
        self.similarity_matrix = similarity_matrix
        self.gamma = gamma
        
    def objective_computation(self, section, adj_matrix, gamma):
        '''
        Compute reconstruction error for a section, adjusted for sparse matrix usage
        '''
        approx_error = 0
        for idx in section:
            #degree_idx = np.sum(adj_matrix[idx, :].toarray())
            degree_idx = np.sum(adj_matrix[idx, :])
            xi_reconstruction = np.sum([adj_matrix[idx, y] * np.asarray(self.data.loc[[y]])[0] for y in range(len(gamma)) if idx != y], axis=0)
            if degree_idx != 0 and not isclose(degree_idx, 0, abs_tol=1e-100):
                xi_reconstruction /= degree_idx
            else:
                xi_reconstruction = np.zeros(len(gamma))
            approx_error += np.sum((np.asarray(self.data.loc[[idx]])[0] - xi_reconstruction)**2)
        return approx_error

    def objective_function(self, adj_matr, gamma):
        '''
        Parallelization of error computation
        '''
        split_data = self.split(range(self.data.shape[0]), cpu_count())
        with Pool(processes=cpu_count()) as pool:
            errors = [pool.apply_async(self.objective_computation, (section, adj_matr, gamma)) for section in split_data]
            error = [error.get() for error in errors]
        return np.sum(error)

    def split(self, a, n):
        k, m = divmod(len(a), n)
        return [a[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def plot_error_surface(aew_obj):

    # Create an instance of the optimization function
    opt_function = OptimizationFunction(data = aew_obj.data, similarity_matrix = aew_obj.similarity_matrix)
    
    values = np.arange(-2, 2.1, .10)

    sim_gammas = np.asarray([list(pair) for pair in itertools.product(values, repeat=2)])

    objective_values = np.zeros(sim_gammas.shape[0])

    for idx, gamma in enumerate(sim_gammas):
        print("Gamma: ", gamma)
        curr_adj_matr = aew_obj.generate_edge_weights(gamma)

        #objective_values.append(opt_function.objective_function(curr_adj_matr))
        
        objective_values[idx] = opt_function.objective_function(curr_adj_matr, gamma)

    # Plotting the surface
    X = np.unique(sim_gammas[:,0]) #np.linspace(0, 10, 10)  # X-axis points (random for illustration)
    Y = np.unique(sim_gammas[:,1])  #np.linspace(0, 10, 10)  # Y-axis points (random for illustration)

    Z = np.zeros((len(X), len(Y)))  

    for gamma, obj_val in zip(sim_gammas, objective_values):
        x_idx = np.where(X == gamma[0])[0][0]
        y_idx = np.where(Y == gamma[1])[0][0]
        Z[x_idx, y_idx] = obj_val

    X, Y = np.meshgrid(X, Y)

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])

    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor='limegreen', project_z=True))

    fig.update_layout(
            title = 'Spread Parameter Error Surface',
            scene=dict(
            xaxis=dict(range=[X.min(), X.max()]),
            yaxis=dict(range=[Y.min(), Y.max()]),
            zaxis=dict(title="Error", range=[Z.min(), Z.max()])
            )
    )

    fig.write_html("error_surface_cm1.html")
    fig.show()

def surface_plotter_driver():
    input_file = 'sq_ds/cm1.csv'
    data_obj = data(input_file, 'whole')
    data_obj.encode_categorical('defects', 'labels')
    data_obj.scale_data('min_max')
    data_obj.generate_graphs(100)
    aew_obj = aew(data_obj.graph, data_obj.data, data_obj.data, data_obj.labels, gamma_init='var')
    aew_obj.get_eigenvectors(2, .90)
    aew_obj.data = aew_obj.eigenvectors 
    plot_error_surface(aew_obj)

if __name__ == '__main__':
    surface_plotter_driver()

