import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example function to generate a cross-correlation matrix
def generate_cross_correlation_matrix(data):
    # If the data is not a DataFrame, convert it to one
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    # Compute the cross-correlation matrix
    corr_matrix = data.corr()

    # Visualize the matrix with a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title("Cross-Correlation Matrix")
    plt.show()

    return corr_matrix

# Example usage
if __name__ == "__main__":
    # Sample data (each column represents a variable)
    data = {
        'Model_A': [64.57, 66.71, 66.03, 68.00, 72.37, 77.01, 82.57, 74.4, 73.49, 98.09, 97.71, 97.56, 96.04],
        'Model_B': [86.20, 74.91, 73.65, 75.00, 70.12, np.nan, np.nan, np.nan, np.nan, 61.54, 48.85, 48.85, 52.07],
        'Model_C': [78.65, 74.66, 75.02, 81.00, 78.96, 79.80, 81.92, np.nan, 74.23, 57.50, 50.00, 49.92, 52.77],
        'Model_D': [34.09, 27.68, 30.99, 33.00, 30.65, 30.46, 29.71, np.nan, 81.20, 97.25, 96.59, 96.51, 95.93],
        'Model_E': [70.00, 53.00, 63.00, 77.00, 80.00, 77.01, np.nan, np.nan, np.nan, 81.69, 49.34, 61.57, 78.77]
    }

    d_file = 'sq_ds/pc1.csv'

    # Convert the dictionary into a Pandas DataFrame
    df = pd.read_csv(d_file)

    # Generate and print the cross-correlation matrix
    corr_matrix = generate_cross_correlation_matrix(df)
    print(corr_matrix)

