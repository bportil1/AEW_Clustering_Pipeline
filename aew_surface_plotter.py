import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import isclose

# Define the class that contains the objective computation function
class OptimizationFunction:
    def __init__(self, data, similarity_matrix, gamma):
        self.data = data
        self.similarity_matrix = similarity_matrix
        self.gamma = gamma
        
    def objective_computation(self, section):
        approx_error = 0
        for idx in section:
            degree_idx = np.sum(self.similarity_matrix[idx])
            xi_reconstruction = np.sum([self.similarity_matrix[idx][y] * np.asarray(self.data.loc[[y]])[0] 
                                        for y in range(len(self.similarity_matrix[idx])) if idx != y], 0)            

            if degree_idx != 0 and not isclose(degree_idx, 0, abs_tol=1e-100):
                xi_reconstruction /= degree_idx
                xi_reconstruction = xi_reconstruction[0]
            else:
                xi_reconstruction = np.zeros(len(self.gamma))

            # Calculate the error (difference between reconstruction and original data)
            approx_error += np.sum((np.asarray(self.data.loc[[idx]])[0] - xi_reconstruction)**2)
        
        return approx_error

import math

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

# Generate synthetic data and similarity matrix
np.random.seed(42)
data = pd.DataFrame(np.random.rand(10, 3), columns=['Feature1', 'Feature2', 'Feature3'])  # 10 data points, 3 features
similarity_matrix = np.random.rand(10, 10)  # Random similarity matrix (10x10)

# Normalize similarity matrix so that diagonal is 1 (self-similarity)
np.fill_diagonal(similarity_matrix, 1)

# Define a gamma vector (arbitrary)
gamma = np.array([1.0, 1.0, 1.0])

# Create an instance of the optimization function
opt_function = OptimizationFunction(data, similarity_matrix, gamma)

# Generate random sections (random subset of indices)
sections = [np.random.choice(data.index, size=5, replace=False) for _ in range(5)]  # 5 random sections

# Evaluate the objective function at random points (sections)
objective_values = np.array([opt_function.objective_computation(section) for section in sections])

# Plotting the surface
x = np.linspace(0, 100, 10)  # X-axis points (random for illustration)
y = np.linspace(0, 100, 10)  # Y-axis points (random for illustration)
X, Y = np.meshgrid(x, y)   # Create a meshgrid for 3D plotting
Z = np.random.rand(10, 10)  # Use random Z-values for the surface

# Create a surface plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot a surface
ax.plot_surface(X, Y, Z, cmap='viridis')

# Add labels and title
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Objective Value')
ax.set_title('Surface of the Objective Function')

plt.show()

