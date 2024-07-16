

from sklearn.metrics import adjusted_rand_score, silhouette_score
from matplotlib import pyplot as plt
from src.cluster import Cluster
import numpy as np

# from sklearn.metrics import normalized_mutual_info_score

def plot_clustering(X, y, model, title, reduction=None, n_components=None):

    aris=[]
    sils=[]
    n=range(2, 7)
    for i in n:
        cls = Cluster(model= model,n_clusters=i,reduction=reduction, n_components=n_components,random_state=19)
        cls.fit(X)
        labels = cls.predict()
        ari = adjusted_rand_score(y, labels)
        sil = silhouette_score(X, labels)
        aris.append(ari)
        sils.append(sil)    

    # plot the iterative learning curve (loss)
    fig = plt.figure(figsize=(3,2.5))
    ax1=fig.add_subplot(1,1,1)



    # plt.plot(n,aris, label='ARI',marker='o')
    plt.plot(n,sils, label='Silhouette',marker='o')
    ax1.set_xlabel("Numbero of Components")
    ax1.set_ylabel("Score")
    ax1.set_title(f"{title}")
    # plt.grid(visible=True)
    plt.legend(frameon=False)

    ax1.set_ylim(0,1)

    plt.rcParams.update({
        'font.size': 10,          # General font size
        'axes.titlesize': 8,     # Font size for titles
        'axes.labelsize': 8,     # Font size for x and y labels
        # 'xtick.labelsize': 8,    # Font size for x-axis tick labels
        # 'ytick.labelsize': 8,    # Font size for y-axis tick labels
        'legend.fontsize': 8     # Font size for legend
    })

    plt.tight_layout() 
    plt.savefig(f'{title}.png',dpi=300)
    plt.show()


if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(42)
    X = np.vstack([np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100),
                   np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], 100),
                   np.random.multivariate_normal([10, 0], [[1, 0], [0, 1]], 100)])

    y = np.array([0]*100 + [1]*100 + [2]*100)

    plot_clustering(X, y, model='kmeans', title='kmeans')
    plot_clustering(X, y, model='gm', title='gm')