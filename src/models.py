from sklearn.cluster import KMeans, AgglomerativeClustering

def elbow_method(X_scaled):
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    wcss = []
    for i in range(1, 11):
        model = KMeans(n_clusters=i, random_state=42)
        model.fit(X_scaled)
        wcss.append(model.inertia_)

    plt.figure()
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.show()

def run_kmeans(X_scaled, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_scaled)
    return model, labels

def run_hierarchical(X_scaled, n_clusters=5):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X_scaled)
    return model, labels
