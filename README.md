# Customer Segmentation using Unsupervised Machine Learning

## Overview

This project implements a modular customer segmentation system using unsupervised learning techniques. The objective is to group retail customers based on purchasing behavior and extract meaningful business insights.

The system compares multiple clustering algorithms and automatically selects the best-performing model based on evaluation metrics.

## Problem Statement

Retail businesses need to understand customer behavior to:

* Improve targeted marketing
* Identify high-value customers
* Optimize promotional strategies
* Increase overall revenue

This project segments customers using Annual Income and Spending Score.

## Project Structure

Customer-Segmentation-ML/
│
├── data/
├── src/
│   ├── preprocess.py
│   ├── models.py
│   ├── evaluate.py
│   └── visualize.py
│
├── main.py
├── requirements.txt
└── README.md

The project follows a modular structure separating preprocessing, modeling, evaluation, and visualization components.

## Features Implemented

* Data preprocessing with feature scaling
* Elbow Method for optimal cluster analysis
* KMeans clustering
* Hierarchical clustering
* Model comparison using Silhouette Score
* PCA for dimensionality reduction and visualization
* Automatic best model selection
* Cluster-wise business insight extraction
* Model saving using pickle

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## Model Evaluation

Both KMeans and Hierarchical clustering were evaluated using Silhouette Score.
The best-performing model is selected automatically based on metric comparison

## Business Insights

The clustering reveals distinct customer groups such as:

* High income, high spending – Premium segment
* High income, low spending – Upsell opportunity
* Low income, high spending – Discount-driven customers
* Low income, low spending – Low engagement group

These insights can be used to design targeted marketing strategies.

## How to Run the Project

1. Clone the repository:git clone https://github.com/NehaM2509/customer-segmentation-ml.git

2. Navigate to project folder:cd customer-segmentation-ml

3. Install dependencies:pip install -r requirements.txt

4. Run the project:python main.py

## Future Improvements

* Add DBSCAN clustering
* Deploy interactive Streamlit dashboard
* Build REST API using FastAPI
* Integrate with marketing recommendation engine

This project demonstrates structured ML pipeline design, evaluation-driven model selection, and business-focused data interpretation.



