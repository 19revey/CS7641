from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from src.reduction import Reduction
from sklearn.metrics import adjusted_rand_score, silhouette_score


class Cluster:
    def __init__(self, model=None, n_clusters=2, reduction=None,n_components=None, random_state=19):
        if model == "em":
            self.cluster_model = GaussianMixture
        elif model == "km":
            self.cluster_model = KMeans

        self.model=self.cluster_model(n_clusters, random_state=random_state)
        self.n_clusters=n_clusters
        self.random_state=random_state
        self.X_reduced=None
        self.reduction=reduction
        self.pca=None


        if self.reduction == "pca":
            # Perform PCA
            self.pca = Reduction("pca",n_components=n_components)
            # X_reduction = self.reduction.fit(X)
        elif reduction == "ica":    
            self.pca = Reduction("ica",n_components=n_components)
        elif reduction == "rp":
            self.pca = Reduction("rp",n_components=n_components)  

    def fit(self, X):
        if self.pca:
            self.X_reduced=self.pca.fit(X)
        else:
            self.X_reduced=X
        self.model.fit(self.X_reduced)
    
        return self.model.predict(self.X_reduced)
    
    def evaluate(self, X, y,n_range):
        aris=[]
        sils=[]
        for i in n_range:
            self.model=self.cluster_model(i, random_state=19)
            self.model.fit(self.X_reduced)
            labels = self.model.predict(self.X_reduced)
            ari = adjusted_rand_score(y, labels)
            sil = silhouette_score(X, labels)
            aris.append(ari)
            sils.append(sil)

        return aris, sils    