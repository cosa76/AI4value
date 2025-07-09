import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import dash
from dash import dcc, html
from google.cloud import bigquery
from flask import send_from_directory

# === Configuration en dur ===
PROJECT_ID = "sandbox-jndong"
DATASET_ID = "DataOriginal2"
FOLDER_PATH = os.path.join(os.path.dirname(__file__), "assets")  # Dossier d'exports visuels
LOG_FILE = "./log_insights.log"
CREDENTIALS_JSON = "Load_file_in_bigquery/dbt_sa_bigquery.json"  # Chemin vers votre fichier JSON Google Cloud
MAX_TABLES = 5  # Limiter le nombre de tables traitées pour test

# === Création des dossiers nécessaires ===
os.makedirs(FOLDER_PATH, exist_ok=True)

# === Configuration du logging ===
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === Connexion à BigQuery ===
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_JSON
client = bigquery.Client(project=PROJECT_ID)

# === Fonction : Charger les noms des tables ===
def get_tables_from_dataset():
    dataset_ref = client.dataset(DATASET_ID, project=PROJECT_ID)
    tables = list(client.list_tables(dataset_ref))
    return [table.table_id for table in tables[:MAX_TABLES]]

# === Fonction : Charger les données en DataFrame ===
def load_table_data(table_id):
    full_table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_id}"
    query = f"SELECT * FROM `{full_table_id}` LIMIT 1000"
    df = client.query(query).to_dataframe()
    return df

# === Fonction : Générer des insights visuels ===
def generate_insights(df, table_name):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    table_folder = os.path.join(FOLDER_PATH, table_name)
    os.makedirs(table_folder, exist_ok=True)

    # Histogrammes pour colonnes numériques
    for col in numeric_cols:
        try:
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution de {col}")
            plt.savefig(os.path.join(table_folder, f"histogram_{col}.png"))
            plt.close()
        except Exception as e:
            logging.error(f"[{table_name}] Échec histogramme {col} : {str(e)}")

    # Countplots pour colonnes catégorielles
    for col in categorical_cols:
        try:
            plt.figure(figsize=(8, 4))
            sns.countplot(data=df, y=col)
            plt.title(f"Répartition de {col}")
            plt.tight_layout()
            plt.savefig(os.path.join(table_folder, f"countplot_{col}.png"))
            plt.close()
        except Exception as e:
            logging.error(f"[{table_name}] Échec countplot {col} : {str(e)}")

    # Matrice de corrélation
    if len(numeric_cols) >= 2:
        try:
            corr = df[numeric_cols].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            plt.title("Matrice de corrélation")
            plt.savefig(os.path.join(table_folder, "correlation_matrix.png"))
            plt.close()
        except Exception as e:
            logging.error(f"[{table_name}] Échec corrélation : {str(e)}")

    # Graphique interactif Plotly
    if len(numeric_cols) >= 2:
        try:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                             title=f"Relation entre {numeric_cols[0]} et {numeric_cols[1]}")
            fig.write_html(os.path.join(table_folder, "scatter_plotly.html"))
        except Exception as e:
            logging.error(f"[{table_name}] Échec scatter Plotly : {str(e)}")

    logging.info(f"[{table_name}] Insights générés.")

# === Fonction : Lancer l'app Dash ===
def run_dash_app(tables_with_insights):
    app = dash.Dash(__name__, assets_folder=FOLDER_PATH)
    app.layout = html.Div([
        html.H1("🔍 Dashboard Automatique d'Insights Dataset BigQuery"),
        html.P(f"Dataset : {DATASET_ID} | Project : {PROJECT_ID}"),
        html.Div(id='insights-container')
    ])

    for table_name in tables_with_insights:
        table_folder = os.path.join(FOLDER_PATH, table_name)
        images = [img for img in os.listdir(table_folder) if img.endswith('.png') or img.endswith('.html')]
        elements = []

        for img in images:
            path = os.path.join(table_name, img)
            if img.endswith(".html"):
                # Utiliser un iframe pour afficher le contenu HTML au lieu de dangerouslySetInnerHTML
                iframe_path = f"/assets/{path}"
                elements.append(html.H4(img))
                elements.append(
                    html.Iframe(
                        src=iframe_path,
                        style={'width': '100%', 'height': '600px', 'border': 'none'}
                    )
                )
            else:
                elements.append(html.H4(img))
                elements.append(html.Img(src=f"/assets/{path}", style={'width': '80%', 'margin-bottom': '20px'}))

        app.layout.children.append(html.Div([
            html.H2(f"📊 Insights - Table : {table_name}"),
            *elements
        ]))

    # Ne pas définir de route personnalisée, Dash gère automatiquement les assets
    # lorsque assets_folder est correctement configuré

    app.run_server(host="127.0.0.1", port=8050, debug=False)  # Spécifier l'adresse locale et le port

# === Main pipeline ===
def main():
    logging.info("Début du traitement")
    tables = get_tables_from_dataset()
    logging.info(f"Tables trouvées : {tables}")

    tables_with_insights = []

    for table_id in tables:
        logging.info(f"Traitement de la table : {table_id}")
        try:
            df = load_table_data(table_id)
            if not df.empty:
                generate_insights(df, table_id)
                tables_with_insights.append(table_id)
            else:
                logging.warning(f"[{table_id}] DataFrame vide.")
        except Exception as e:
            logging.error(f"Échec traitement table {table_id} : {str(e)}")

    run_dash_app(tables_with_insights)

# === Point d'entrée ===
if __name__ == "__main__":
    main()