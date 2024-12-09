import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_cross_correlation_matrix(data):
    '''
    Function to generate a cross correlation matrix for the given data
    '''
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

if __name__ == "__main__":
    d_file = 'sq_ds/pc1.csv'

    df = pd.read_csv(d_file)
    corr_matrix = generate_cross_correlation_matrix(df)
    print(corr_matrix)

