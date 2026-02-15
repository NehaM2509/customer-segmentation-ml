from src.preprocess import load_and_preprocess
from src.models import run_kmeans, run_hierarchical, elbow_method
from src.evaluate import evaluate_model
from src.visualize import plot_clusters, plot_pca
import pickle

# -------------------------
# Load & Preprocess
# -------------------------
df, X_scaled, scaler = load_and_preprocess("data/Mall_Customers.csv")

# -------------------------
# Elbow Method
# -------------------------
elbow_method(X_scaled)

# -------------------------
# Run Models
# -------------------------
kmeans_model, kmeans_labels = run_kmeans(X_scaled, n_clusters=5)
hier_model, hier_labels = run_hierarchical(X_scaled, n_clusters=5)

# -------------------------
# Evaluate Models
# -------------------------
kmeans_score = evaluate_model(X_scaled, kmeans_labels)
hier_score = evaluate_model(X_scaled, hier_labels)

print("\nModel Comparison:")
print(f"KMeans Silhouette Score: {kmeans_score}")
print(f"Hierarchical Silhouette Score: {hier_score}")

# -------------------------
# Select Best Model
# -------------------------
if kmeans_score > hier_score:
    print("KMeans selected as best model.")
    df['Cluster'] = kmeans_labels
    best_model = kmeans_model
    best_labels = kmeans_labels
else:
    print("Hierarchical selected as best model.")
    df['Cluster'] = hier_labels
    best_model = hier_model
    best_labels = hier_labels

# -------------------------
# Visualizations
# -------------------------
plot_clusters(df)
plot_pca(X_scaled, best_labels)

# -------------------------
# Business Insights
# -------------------------
print("\nCluster Business Insights:")
print(df.groupby("Cluster")[['Annual Income (k$)', 'Spending Score (1-100)']].mean())

# -------------------------
# Save Best Model
# -------------------------
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nBest model and scaler saved successfully!")
