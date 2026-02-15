from sklearn.metrics import silhouette_score

def evaluate_model(X_scaled, labels):
    return silhouette_score(X_scaled, labels)
