import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import panel as pn
from google.cloud import bigquery

# Initialiser Panel
pn.extension('plotly')

# === Configuration en dur ===
PROJECT_ID = "sandbox-jndong"
DATASET_ID = "DataOriginal2"
FOLDER_PATH = os.path.join(os.path.dirname(__file__), "assets_panel")  # Dossier d'exports visuels
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

# === Fonction : Générer des insights visuels avec Panel ===
def generate_panel_insights(df, table_name):
    table_folder = os.path.join(FOLDER_PATH, table_name)
    os.makedirs(table_folder, exist_ok=True)
    
    insights = []
    
    # Créer un titre pour cette table
    insights.append(pn.pane.Markdown(f"# 📊 Insights pour la table: {table_name}"))
    
    # Afficher les informations de base sur le DataFrame
    insights.append(pn.pane.Markdown("## Aperçu des données"))
    insights.append(pn.pane.DataFrame(df.head(), width=800))
    
    insights.append(pn.pane.Markdown("## Statistiques descriptives"))
    try:
        insights.append(pn.pane.DataFrame(df.describe().T, width=800))
    except Exception as e:
        logging.error(f"[{table_name}] Échec des statistiques descriptives : {str(e)}")
    
    # Identifier les types de colonnes
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
    # Visualisations pour colonnes numériques
    if numeric_cols:
        insights.append(pn.pane.Markdown("## Distributions des variables numériques"))
        
        for col in numeric_cols[:5]:  # Limiter à 5 colonnes pour éviter une surcharge
            try:
                # Histogramme avec Plotly pour interactivité
                fig = px.histogram(df, x=col, marginal="box", title=f"Distribution de {col}")
                insights.append(pn.pane.Plotly(fig, height=400))
                
                # Sauvegarder aussi en fichier statique pour référence
                plt.figure(figsize=(8, 4))
                sns.histplot(df[col], kde=True)
                plt.title(f"Distribution de {col}")
                plt.savefig(os.path.join(table_folder, f"histogram_{col}.png"))
                plt.close()
            except Exception as e:
                logging.error(f"[{table_name}] Échec histogramme {col} : {str(e)}")
    
    # Visualisations pour colonnes catégorielles
    if categorical_cols:
        insights.append(pn.pane.Markdown("## Distributions des variables catégorielles"))
        
        for col in categorical_cols[:5]:  # Limiter à 5 colonnes
            try:
                # Graphique à barres avec Plotly
                value_counts = df[col].value_counts().reset_index()
                value_counts.columns = [col, 'count']
                # Limiter à 15 catégories maximum pour la lisibilité
                if len(value_counts) > 15:
                    value_counts = pd.concat([
                        value_counts.head(10),
                        pd.DataFrame({col: ['Autres'], 'count': [value_counts.iloc[10:]['count'].sum()]})
                    ])
                
                fig = px.bar(value_counts, x=col, y='count', title=f"Répartition de {col}")
                insights.append(pn.pane.Plotly(fig, height=400))
                
                # Sauvegarder aussi en fichier statique
                plt.figure(figsize=(8, 6))
                sns.countplot(data=df, y=col, order=df[col].value_counts().iloc[:15].index)
                plt.title(f"Répartition de {col}")
                plt.tight_layout()
                plt.savefig(os.path.join(table_folder, f"countplot_{col}.png"))
                plt.close()
            except Exception as e:
                logging.error(f"[{table_name}] Échec countplot {col} : {str(e)}")
    
    # Matrice de corrélation
    if len(numeric_cols) >= 2:
        insights.append(pn.pane.Markdown("## Corrélations entre variables numériques"))
        try:
            corr = df[numeric_cols].corr()
            
            # Heatmap interactive avec Plotly
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Matrice de corrélation")
            insights.append(pn.pane.Plotly(fig, height=600))
            
            # Sauvegarder aussi en fichier statique
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            plt.title("Matrice de corrélation")
            plt.savefig(os.path.join(table_folder, "correlation_matrix.png"))
            plt.close()
        except Exception as e:
            logging.error(f"[{table_name}] Échec corrélation : {str(e)}")
    
    # Diagramme de dispersion pour paires de variables numériques
    if len(numeric_cols) >= 2:
        insights.append(pn.pane.Markdown("## Relations entre variables"))
        try:
            # Limiter à 2 premières colonnes numériques pour la démonstration
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                            title=f"Relation entre {numeric_cols[0]} et {numeric_cols[1]}")
            insights.append(pn.pane.Plotly(fig, height=400))
            
            # Sauvegarder aussi en HTML
            fig.write_html(os.path.join(table_folder, "scatter_plotly.html"))
        except Exception as e:
            logging.error(f"[{table_name}] Échec scatter Plotly : {str(e)}")
    
    logging.info(f"[{table_name}] Insights Panel générés.")
    return insights

# === Fonction principale pour créer le tableau de bord Panel ===
def create_panel_dashboard(tables_with_insights, tables_data):
    # Créer un panneau d'onglets pour chaque table
    tabs = pn.Tabs()
    
    # Page d'accueil
    current_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    welcome = pn.Column(
        pn.pane.Markdown(f"""
        # 🔍 Dashboard Automatique d'Insights Dataset BigQuery
        
        ## Projet : {PROJECT_ID}
        ## Dataset : {DATASET_ID}
        
        Ce tableau de bord présente une analyse automatique de vos tables BigQuery.
        Sélectionnez un onglet pour explorer les insights de chaque table.
        
        *Généré le : {current_date}*
        """),
        pn.pane.Markdown("### Tables analysées :"),
        pn.pane.Markdown("\n".join([f"- {table}" for table in tables_with_insights]))
    )
    
    tabs.append(("Accueil", welcome))
    
    # Créer un onglet pour chaque table
    for table_name in tables_with_insights:
        df = tables_data[table_name]
        insights = generate_panel_insights(df, table_name)
        tab_content = pn.Column(*insights, width=900)
        tabs.append((table_name, tab_content))
    
    # Utilisez MaterialTemplate ou simplement un layout de base
    dashboard = pn.template.MaterialTemplate(
        title="Insights BigQuery Dashboard",
        sidebar=["### Navigation", tabs],
        main=[pn.Column(
            pn.Row(
                pn.pane.Markdown(f"# 🔍 Dashboard Insights BigQuery"),
                pn.pane.Markdown(f"*Projet: {PROJECT_ID} | Dataset: {DATASET_ID}*")
            ),
            tabs[1:] if len(tabs) > 1 else pn.pane.Markdown("### Aucune donnée disponible")
        )]
    )
    
    return dashboard

# === Main pipeline ===
def main():
    logging.info("Début du traitement")
    tables = get_tables_from_dataset()
    logging.info(f"Tables trouvées : {tables}")

    tables_with_insights = []
    tables_data = {}

    for table_id in tables:
        logging.info(f"Traitement de la table : {table_id}")
        try:
            df = load_table_data(table_id)
            if not df.empty:
                tables_data[table_id] = df
                tables_with_insights.append(table_id)
            else:
                logging.warning(f"[{table_id}] DataFrame vide.")
        except Exception as e:
            logging.error(f"Échec traitement table {table_id} : {str(e)}")

    if tables_with_insights:
        dashboard = create_panel_dashboard(tables_with_insights, tables_data)
        # Lancer le serveur Panel sur localhost avec port spécifié
        # Définir les origines autorisées pour les connexions WebSocket
        pn.config.sizing_mode = "stretch_width"
        pn.config.sizing_mode = "stretch_width"
        pn.serve(
            dashboard,
            port=5006, 
            address="127.0.0.1", 
            show=True,
            websocket_origin=["127.0.0.1:5006", "localhost:5006"]
        )
    else:
        logging.error("Aucune table avec des insights à afficher.")

# === Point d'entrée ===
if __name__ == "__main__":
    main()