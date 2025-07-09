# full_script.py

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from google.oauth2 import service_account
from scipy.stats import shapiro
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from sklearn.decomposition import PCA
import streamlit as st

# === Configuration projet (à saisir en dur) ===
PROJECT_ID = "sandbox-jndong"
DATASET_ID = "dev"
TABLE_ID = "ventes"

# Chemins
FOLDER_RESULT = 'folder_result'
os.makedirs(FOLDER_RESULT, exist_ok=True)
LOG_FILE = 'log_statistiques.log'

# Logger
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Chargement des données BigQuery
def load_data_from_bigquery():
    try:
        credentials = service_account.Credentials.from_service_account_file('dbt_sa_bigquery.json')
        client = bigquery.Client(project=PROJECT_ID, credentials=credentials)
        query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`"
        df = client.query(query).to_dataframe()
        logging.info("Données chargées depuis BigQuery.")
        return df
    except Exception as e:
        logging.error(f"Erreur lors du chargement des données : {e}")
        print(f"❌ Erreur : {e}")
        return None

# 1. Statistiques descriptives
def descriptive_stats(df):
    desc = df.describe(include='all')
    path = os.path.join(FOLDER_RESULT, 'descriptive_stats.csv')
    desc.to_csv(path)
    logging.info("Statistiques descriptives calculées.")

# 2. Corrélation entre deux colonnes numériques
def compute_correlation(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[0], numeric_cols[1]
        corr, p = pearsonr(df[col1], df[col2])
        print(f"📈 Corrélation entre {col1} et {col2} : {corr:.2f}, p-value: {p:.4f}")
        plt.figure()
        sns.scatterplot(x=df[col1], y=df[col2])
        plt.title(f"{col1} vs {col2}")
        plt.savefig(os.path.join(FOLDER_RESULT, f"correlation_{col1}_{col2}.png"))
        plt.close()
        logging.info(f"Corrélation entre {col1} et {col2} calculée.")
    else:
        logging.warning("Pas assez de colonnes numériques pour la corrélation.")

# 3. Test de normalité
def test_normality(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        try:
            stat, p = shapiro(df[col].dropna())
            print(f"🧮 Test Shapiro-Wilk pour {col}: p-value = {p:.4f}")
            logging.info(f"Test Shapiro-Wilk pour {col}: p-value = {p:.4f}")
        except:
            continue

# 4. Régression linéaire
def linear_regression(df):
    numeric_df = df.select_dtypes(include=np.number)
    if len(numeric_df.columns) < 2:
        logging.warning("Régression linéaire impossible : pas assez de variables numériques.")
        return
    X_col = numeric_df.columns[0]
    y_col = numeric_df.columns[-1]
    X = df[[X_col]]
    y = df[y_col]
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    print(f"📉 Régression linéaire : R² = {r2:.2f}")
    plt.figure()
    plt.scatter(X, y, color="blue")
    plt.plot(X, model.predict(X), color="red")
    plt.title(f"{y_col} = f({X_col})")
    plt.savefig(os.path.join(FOLDER_RESULT, f"regression_linear_{X_col}_{y_col}.png"))
    plt.close()
    logging.info(f"Régression linéaire effectuée entre {X_col} et {y_col}")

# 5. Régression logistique
def logistic_regression(df):
    numeric_df = df.select_dtypes(include=np.number)
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if len(cat_cols) == 0 or len(numeric_df.columns) == 0:
        logging.warning("Régression logistique impossible : pas de colonnes catégorielles ou numériques.")
        return
    X_col = numeric_df.columns[0]
    y_col = cat_cols[0]
    if len(df[y_col].unique()) > 10:
        logging.warning(f"La variable cible '{y_col}' semble numérique.")
        return
    X = df[[X_col]]
    y = df[y_col].astype(str).astype('category').cat.codes
    model = LogisticRegression(max_iter=1000).fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    print(f"🎯 Régression logistique : Précision = {acc:.2f}")
    logging.info(f"Régression logistique effectuée entre {X_col} et {y_col}")

# 6. Clustering K-Means
def kmeans_clustering(df):
    numeric_df = df.select_dtypes(include=np.number)
    if len(numeric_df.columns) < 2:
        logging.warning("Clustering K-Means impossible.")
        return
    cols = numeric_df.columns[:2]  # Prendre les 2 premières colonnes
    X = df[cols]
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    df['cluster'] = kmeans.labels_
    path = os.path.join(FOLDER_RESULT, 'kmeans_results.csv')
    df.to_csv(path, index=False)
    print(f"📌 Clustering K-Means effectué sur {cols}. Sauvegardé dans {path}")
    logging.info(f"Clustering K-Means effectué sur {cols}")

# 7. Analyse PCA
def pca_analysis(df):
    numeric_df = df.select_dtypes(include=np.number)
    if len(numeric_df.columns) < 2:
        logging.warning("Analyse PCA impossible.")
        return
    cols = numeric_df.columns.tolist()[:2]
    X = df[cols]
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    plt.figure()
    plt.scatter(components[:, 0], components[:, 1])
    plt.title("PCA")
    plt.savefig(os.path.join(FOLDER_RESULT, 'pca_plot.png'))
    plt.close()
    print(f"🔍 Analyse PCA effectuée sur {cols}. Sauvegardée dans folder_result/")
    logging.info(f"Analyse PCA effectuée sur {cols}")

# 8. Modèle ARIMA (si applicable)
def arima_forecast(df):
    time_cols = df.select_dtypes(include=np.number).columns
    if len(time_cols) == 0:
        logging.warning("Aucune colonne numérique pour ARIMA.")
        return
    col = time_cols[0]
    if len(df[col]) < 10:
        logging.warning(f"Pas assez de données pour ARIMA sur {col}.")
        return
    try:
        model = ARIMA(df[col], order=(1,1,1))
        results = model.fit()
        forecast = results.forecast(steps=5)
        print(f"⏳ Prévision ARIMA pour les 5 prochaines valeurs de '{col}' : {forecast.values}")
        logging.info(f"Modèle ARIMA appliqué à {col}")
    except:
        logging.warning(f"Échec de l'application ARIMA sur {col}.")

# Interface Streamlit
def run_streamlit():
    st.set_page_config(page_title="Dashboard", layout="wide")
    st.title("📊 Dashboard Automatique")

    files = [f for f in os.listdir(FOLDER_RESULT) if os.path.isfile(os.path.join(FOLDER_RESULT, f))]

    if not files:
        st.warning("Aucun résultat trouvé. Exécutez d'abord le script principal.")
    else:
        st.success("Résultats trouvés !")

    for file in files:
        path = os.path.join(FOLDER_RESULT, file)
        if file.endswith(".csv"):
            df = pd.read_csv(path)
            st.subheader(file)
            st.dataframe(df)
        elif file.endswith(".png"):
            st.subheader(file)
            st.image(path)
        elif file.endswith(".txt"):
            st.subheader(file)
            with open(path, 'r') as f:
                st.text(f.read())

    with open(LOG_FILE, "r") as log_file:
        logs = log_file.read()

    st.sidebar.header("Logs")
    st.sidebar.text_area("", value=logs, height=300)

# Point d'entrée principal
if __name__ == "__main__":
    import sys
    if "--streamlit" in sys.argv:
        run_streamlit()
    else:
        print("🔄 Chargement des données...")
        df = load_data_from_bigquery()
        if df is not None and not df.empty:
            print("✅ Données chargées avec succès.")
            print("📊 Colonnes disponibles :", df.columns.tolist())

            # Exécution automatique des algorithmes
            descriptive_stats(df)
            compute_correlation(df)
            test_normality(df)
            linear_regression(df)
            logistic_regression(df)
            kmeans_clustering(df)
            pca_analysis(df)
            arima_forecast(df)

            print("✅ Analyse terminée. Consultez les fichiers dans folder_result/ et le log.")
        else:
            print("❌ Aucune donnée disponible.")