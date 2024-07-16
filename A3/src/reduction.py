
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
import numpy as np
from scipy.stats import kurtosis
from sklearn.metrics import mean_squared_error

def compute_kurtosis(X_transformed):
    return kurtosis(X_transformed, axis=0)

def calculate_reconstruction_error(X, rp):
    X_projected = rp.fit_transform(X)
    projection_matrix = rp.components_
    X_reconstructed = X_projected @ np.linalg.pinv(projection_matrix)
    return mean_squared_error(X, X_reconstructed)

class Reduction:
    def __init__(self, model=None, n_components=None, random_state=110):
        self.name=model
        if model == "pca":
            model = PCA
            
        elif model == "ica":
            model = FastICA
        elif model == "rp":
            model = GaussianRandomProjection
        
        self.model=model(n_components, random_state=random_state)

    def fit(self, X):

        return self.model.fit_transform(X)
    
    def evaluate(self,X):
        if self.name == "pca":
            eigenvalues = self.model.explained_variance_
            eigenvectors = self.model.components_
            explained_variance_ratio = self.model.explained_variance_ratio_
            cumulative_explained_variance = np.cumsum(explained_variance_ratio)
            return eigenvalues,cumulative_explained_variance
        elif self.name == "ica":
            components_range = range(1, X.shape[1] + 1)
            average_kurtosis = []
            for n_components in components_range:
                model = Reduction("ica",n_components=n_components)
                X_ica = model.fit(X)
                kurtosis_values = compute_kurtosis(X_ica)
                avg_kurtosis = np.mean(np.abs(kurtosis_values))
                average_kurtosis.append(avg_kurtosis)
            
            return average_kurtosis
        
        elif self.name == "rp":
            components_range = range(1, X.shape[1] + 1)
            reconstruction_errors = []

            for n_components in components_range:
                model = Reduction("rp",n_components=n_components)
                X_ica = model.fit(X)
                reconstruction_error = calculate_reconstruction_error(X_ica, model.model)
                reconstruction_errors.append(reconstruction_error)
            return reconstruction_errors