

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from src.reduction import Reduction
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import mean_squared_error

from scipy.stats import kurtosis

def compute_kurtosis(X_transformed):
    return kurtosis(X_transformed, axis=0)


def calculate_reconstruction_error(X, rp):
    X_projected = rp.fit_transform(X)
    projection_matrix = rp.components_
    X_reconstructed = X_projected @ np.linalg.pinv(projection_matrix)
    return mean_squared_error(X, X_reconstructed)

def plot_reduction(X, y, model, title):

    if model == "pca":
        # Perform PCA
        model = Reduction("pca")
        X_pca = model.fit(X)

        # Get eigenvalues (singular values squared divided by number of samples)
        eigenvalues = model.model.explained_variance_
        eigenvectors = model.model.components_

        # print("Eigenvalues:")
        # print(eigenvalues)
        # print("\nEigenvectors (Principal Components):")
        # print(eigenvectors)

        # Plot the eigenvalues
        fig,ax1 = plt.subplots(figsize=(3,2.5))
        ax2 = ax1.twinx()

        # plt.subplot(2, 1, 1)
        ax1.plot(eigenvalues, marker='o')
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('Eigenvalue')
        # plt.title(f'{title}')

        # Plot the explained variance
        explained_variance_ratio = model.model.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)

        # plt.subplot(2, 1, 2)
        ax2.plot(cumulative_explained_variance, marker='o', color='red')
        # ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')

        ax2.set_ylim(0, 1.1)
        plt.ylabel('Cumulative Explained Variance')
        plt.title(f'{title}',fontsize=10)

        plt.rcParams.update({
            'font.size': 10,          # General font size
            'axes.titlesize': 8,     # Font size for titles
            'axes.labelsize': 8,     # Font size for x and y labels
            # 'xtick.labelsize': 8,    # Font size for x-axis tick labels
            # 'ytick.labelsize': 8,    # Font size for y-axis tick labels
            'legend.fontsize': 8     # Font size for legend
        })

        plt.tight_layout() 
        plt.savefig(f'{title}.pdf')
        plt.show()



    elif model == "ica":
        components_range = range(1, X.shape[1] + 1)
        average_kurtosis = []

        for n_components in components_range:
            model = Reduction("ica",n_components=n_components)
            X_ica = model.fit(X)
            kurtosis_values = compute_kurtosis(X_ica)
            avg_kurtosis = np.mean(np.abs(kurtosis_values))
            average_kurtosis.append(avg_kurtosis)
        
        plt.figure(figsize=(3, 2.5))
        plt.plot(components_range, average_kurtosis, marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Average Absolute Kurtosis')
        plt.title(f'{title}',fontsize=10)
        # plt.grid(True)
        plt.show()

    elif model == "rp":
        components_range = range(1, X.shape[1] + 1)
        reconstruction_errors = []

        for n_components in components_range:
            model = Reduction("rp",n_components=n_components)
            X_ica = model.fit(X)
            reconstruction_error = calculate_reconstruction_error(X_ica, model.model)
            reconstruction_errors.append(reconstruction_error)
        
        plt.figure(figsize=(3, 2.5))
        plt.plot(components_range, reconstruction_errors, marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Reconstruction Error')
        plt.title(f'{title}',fontsize=10)


        plt.rcParams.update({
            'font.size': 10,          # General font size
            'axes.titlesize': 8,     # Font size for titles
            'axes.labelsize': 8,     # Font size for x and y labels
            # 'xtick.labelsize': 8,    # Font size for x-axis tick labels
            # 'ytick.labelsize': 8,    # Font size for y-axis tick labels
            'legend.fontsize': 8,     # Font size for legend
        })
        # plt.grid(True)
        # plt.ylim(0, 1000) 
        plt.show()