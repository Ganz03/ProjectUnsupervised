import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Title of the App
st.title("Telco Customer Churn Analysis and Clustering")

# Upload CSV File
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    # Load Dataset
    df = pd.read_csv(uploaded_file)
    st.write("## Data Preview")
    st.dataframe(df.head())

    # Data Cleaning
    df['Total Charges'] = df['Total Charges'].replace(" ", "0").astype(float)

    # Sidebar for Features Selection
    st.sidebar.title("Feature Selection")
    numerical_features = st.sidebar.multiselect(
        "Select Numerical Features",
        ['Monthly Charges', 'Total Charges', 'CLTV', 'Tenure Months'],
        default=['Monthly Charges', 'Total Charges', 'CLTV', 'Tenure Months']
    )

    categorical_features = st.sidebar.multiselect(
    "Select Categorical Features",
    ['Contract', 'Payment Method', 'Paperless Billing'], 
    default=['Contract', 'Payment Method', 'Paperless Billing'] 
)

    # Encode Categorical Features
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        label_encoders[feature] = le

    # Clustering
    st.write("## Customer Clustering")
    df_clustering = df[numerical_features + categorical_features].dropna()
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_clustering)

    # Train Final KMeans Model
    optimal_clusters = 4  # Based on the Elbow Method and Silhouette Score
    kmeans_final = KMeans(n_clusters=optimal_clusters, random_state=42).fit(df_scaled)
    cluster_labels = kmeans_final.labels_

    # Save Models
    with open('kmeans_model.pkl', 'wb') as model_file:
        pickle.dump(kmeans_final, model_file)

    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    with open('label_encoders.pkl', 'wb') as encoder_file:
        pickle.dump(label_encoders, encoder_file)

    # Visualizations
    st.write("## Data Visualizations")

    # 1. Distribusi Churn Berdasarkan Metode Pembayaran
    st.write("### Distribusi Churn Berdasarkan Metode Pembayaran")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Payment Method', hue='Churn Label', palette='Set1')
    plt.title("Distribusi Churn Berdasarkan Metode Pembayaran")
    plt.xlabel("Metode Pembayaran")
    plt.ylabel("Jumlah")
    plt.xticks(rotation=45)
    plt.legend(title='Churn', loc='upper right')
    st.pyplot(plt)

    # 2. Pengaruh Contract terhadap Churn
    st.write("### Pengaruh Contract terhadap Churn")
    sns.countplot(data=df, x='Contract', hue='Churn Label')
    plt.title("Pengaruh Contract terhadap Churn")
    plt.xlabel("Contract")
    plt.ylabel("Jumlah")
    plt.legend(title='Churn')
    st.pyplot(plt)

    # 3. Matriks Korelasi Antar Variabel Numerik
    st.write("### Matriks Korelasi Antar Variabel Numerik")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numeric_columns].corr()

    # Plotting Heatmap Korelasi
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Matriks Korelasi Antar Variabel Numerik")
    st.pyplot(plt)

    # 4. Distribusi Fitur Kategori
    for feature in categorical_features:
        st.write(f"### Distribusi {feature}")
        plt.figure(figsize=(8, 6))  # Set ukuran figure untuk setiap plot
        sns.countplot(x=df[feature], palette='viridis')
        plt.title(f"Distribusi {feature}")
        plt.xticks(rotation=45)  # Rotasi label x untuk keterbacaan yang lebih baik
        st.pyplot(plt)

    # 5. Clustering Visualization using PCA (2D)
    st.write("### Clustering Visualization (PCA 2D Projection)")
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df_scaled)
    df_pca = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
    df_pca['Cluster'] = cluster_labels

    fig, ax = plt.subplots()
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='Set3', data=df_pca, ax=ax, legend="full")
    st.pyplot(fig)

    # 6. Elbow and Silhouette Score Visualizations
    st.write("### Elbow Method and Silhouette Scores")

    # Calculate Inertia and Silhouette Scores for different numbers of clusters
    inertia = []
    silhouette_scores = []
    cluster_range = range(2, 11)  # Start from 2 clusters

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(df_scaled)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df_scaled, cluster_labels))

    # Plot Elbow and Silhouette Scores
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Elbow Method Plot
    axes[0].plot(cluster_range, inertia, marker='o')
    axes[0].set_title('Elbow Method for Optimal Clusters')
    axes[0].set_xlabel('Number of Clusters')
    axes[0].set_ylabel('Inertia')
    axes[0].grid(True)

    # Silhouette Score Plot
    axes[1].plot(cluster_range, silhouette_scores, marker='o', color='orange')
    axes[1].set_title('Silhouette Scores for Clustering')
    axes[1].set_xlabel('Number of Clusters')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].grid(True)

    st.pyplot(fig)

    # 7. Silhouette Score for Clustering
    st.write("### Silhouette Score for Clustering")
    silhouette_avg = silhouette_score(df_scaled, cluster_labels)
    st.write(f"The average silhouette score for the clustering is: **{silhouette_avg:.2f}**")

    # Prediction for New Data
    st.write("## Predict Cluster for New Data")
    with st.form("prediction_form"):
        st.write("Enter the values for the following features:")
        
        # Input for numerical features
        inputs = {}
        for feature in numerical_features:
            inputs[feature] = st.number_input(f"{feature}", min_value=0.0, step=0.1)

        # Input for categorical features
        for feature in categorical_features:
            unique_values = label_encoders[feature].classes_
            selected_value = st.selectbox(f"{feature}", unique_values)
            inputs[feature] = label_encoders[feature].transform([selected_value])[0]

        # Submit Button
        submitted = st.form_submit_button("Predict Cluster")

        if submitted:
            new_data = pd.DataFrame([inputs])
            new_data_scaled = scaler.transform(new_data)
            cluster_prediction = kmeans_final.predict(new_data_scaled)
            st.write(f"The predicted cluster for the input data is: **Cluster {cluster_prediction[0]}**")
