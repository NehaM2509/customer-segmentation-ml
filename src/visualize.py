import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def plot_clusters(df):
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x=df['Annual Income (k$)'],
        y=df['Spending Score (1-100)'],
        hue=df['Cluster'],
        palette='viridis'
    )
    plt.title("Customer Segmentation")
    plt.show()

def plot_pca(X_scaled, labels):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x=components[:,0],
        y=components[:,1],
        hue=labels,
        palette='viridis'
    )
    plt.title("PCA Cluster Visualization")
    plt.show()
