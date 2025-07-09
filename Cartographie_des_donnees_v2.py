# -*- coding: utf-8 -*-
"""
Cartographie interactive + Export vers BigQuery + Logging
"""
import os
from datetime import datetime
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

# Configuration
CREDENTIAL = "Load_file_in_bigquery/dbt_sa_bigquery.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIAL
PROJECT_ID =  "sandbox-jndong"
DATASET_ID = "DataOriginal"
DATASET_CARTOGRAPHIE_ID = "Data_dictionary"  # 👈 Nouveau dataset pour les résultats
FOLDER_PATH = os.path.join(os.path.dirname(__file__), "cartographie_output")  # Dossier principal de sortie
LOG_FILE = os.path.join(FOLDER_PATH, "log_cartographie.log")

# Création du dossier si inexistant
os.makedirs(FOLDER_PATH, exist_ok=True)

# --- Initialisation du logging ---
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")

log("🚀 Démarrage de l'application de cartographie.")

# --- Initialisation client BigQuery ---
client = bigquery.Client(project=PROJECT_ID)

# --- Création du dataset cartographie si inexistant ---
def create_cartography_dataset():
    try:
        client.get_dataset(f"{PROJECT_ID}.{DATASET_CARTOGRAPHIE_ID}")
        log(f"✅ Dataset '{DATASET_CARTOGRAPHIE_ID}' déjà existant.")
    except NotFound:
        dataset_ref = client.dataset(DATASET_CARTOGRAPHIE_ID)
        dataset = bigquery.Dataset(dataset_ref)
        dataset.description = "Dataset généré automatiquement par l'outil de cartographie"
        dataset.location = "EU"  # ou "US"
        dataset = client.create_dataset(dataset)
        log(f"🆕 Dataset '{DATASET_CARTOGRAPHIE_ID}' créé.")

create_cartography_dataset()

# --- Fonctions utilitaires ---

def get_tables():
    tables = list(client.list_tables(DATASET_ID))
    return {t.table_id: f"{PROJECT_ID}.{DATASET_ID}.{t.table_id}" for t in tables}

def get_table_metadata(table_id):
    table_ref = client.dataset(DATASET_ID).table(table_id)
    table_obj = client.get_table(table_ref)
    schema = [{"name": f.name, "type": f.field_type, "description": f.description or ''} for f in table_obj.schema]
    return {
        "nom_complet": f"{PROJECT_ID}.{DATASET_ID}.{table_id}",
        "lignes": table_obj.num_rows,
        "schema": pd.DataFrame(schema)
    }

def load_sample_data(table_id, limit=10000):
    query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{table_id}` LIMIT {limit}"
    df = client.query(query).to_dataframe()
    log(f"📥 Données chargées pour la table {table_id} (limite : {limit})")
    return df

def analyser_qualite(table_id):
    df = load_sample_data(table_id, limit=100000)
    qualite = []
    for col in df.columns:
        serie = df[col]
        q = {
            "table_id": table_id,
            "colonne": col,
            "type": str(serie.dtype),
            "valeurs_manquantes": serie.isna().sum(),
            "pourcentage_manquant": round(serie.isna().mean() * 100, 2),
        }
        if pd.api.types.is_numeric_dtype(serie):
            Q1 = serie.quantile(0.25)
            Q3 = serie.quantile(0.75)
            IQR = Q3 - Q1
            q["valeurs_aberrantes"] = ((serie < (Q1 - 1.5 * IQR)) | (serie > (Q3 + 1.5 * IQR))).sum()
        else:
            q["valeurs_aberrantes"] = None
        qualite.append(q)
    log(f"🔍 Analyse qualité terminée pour la table {table_id}.")
    return pd.DataFrame(qualite)

def analyser_stats(table_id):
    df = load_sample_data(table_id, limit=10000)
    stats = []
    for col in df.columns:
        serie = df[col]
        if pd.api.types.is_numeric_dtype(serie):
            stats.append({
                "table_id": table_id,
                "colonne": col,
                "min": serie.min(),
                "max": serie.max(),
                "moyenne": serie.mean(),
                "mediane": serie.median(),
                "ecart_type": serie.std()
            })
    log(f"📊 Statistiques calculées pour la table {table_id}.")
    return pd.DataFrame(stats)

def exporter_vers_bigquery(df, table_name):
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
    )
    table_ref = client.dataset(DATASET_CARTOGRAPHIE_ID).table(table_name)
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    log(f"💾 Données exportées dans la table {DATASET_CARTOGRAPHIE_ID}.{table_name}")

# --- Application Dash ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# --- Layout de l'application ---
app.layout = dbc.Container([
    html.H1("📊 Cartographie Interactive des Données", className="my-4 text-center"),

    dbc.Row([
        dbc.Col([
            html.Label("Sélectionnez une table :"),
            dcc.Dropdown(
                id='table-selector',
                options=[{'label': t, 'value': t} for t in get_tables().keys()],
                value=list(get_tables().keys())[0]
            )
        ], width=6)
    ]),

    dbc.Row([
        dbc.Col([
            html.Button("Exporter toutes les données dans BigQuery", id='export-button', n_clicks=0, className='btn btn-success mt-3')
        ])
    ]),

    html.Div(id='export-status', className="mt-2 text-success"),

    html.Div(id='metadata-output', className="mt-4"),
    html.Div(id='data-preview', className="mt-4"),
    html.Div(id='distribution-plot', className="mt-4"),
    html.Div(id='correlation-plot', className="mt-4"),

], fluid=True)

# --- Callbacks ---

@app.callback(
    Output('metadata-output', 'children'),
    Input('table-selector', 'value')
)
def update_metadata(table_id):
    meta = get_table_metadata(table_id)
    df_schema = meta["schema"]
    return dbc.Card([
        dbc.CardHeader("📚 Métadonnées"),
        dbc.CardBody([
            html.P(f"Table : {meta['nom_complet']}"),
            html.P(f"Nombre de lignes : {meta['lignes']}"),
            html.P(f"Nombre de colonnes : {len(df_schema)}"),
            dbc.Table.from_dataframe(df_schema, striped=True, bordered=True, hover=True)
        ])
    ])

@app.callback(
    Output('data-preview', 'children'),
    Input('table-selector', 'value')
)
def preview_data(table_id):
    df = load_sample_data(table_id, limit=100)
    if df.empty:
        return html.Div("⚠️ Aucune donnée disponible.")
    return dbc.Card([
        dbc.CardHeader("🧾 Aperçu des données"),
        dbc.CardBody([
            dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True)
        ])
    ])

@app.callback(
    Output('distribution-plot', 'children'),
    Input('table-selector', 'value')
)
def plot_distribution(table_id):
    df = load_sample_data(table_id, limit=10000)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return html.Div("🚫 Aucune colonne numérique pour afficher la distribution.")

    plots = []
    for col in numeric_cols[:3]:  # Limite à 3 colonnes pour performance
        fig = px.histogram(df, x=col, title=f"Distribution - {col}", nbins=50)
        plots.append(dcc.Graph(figure=fig))

    return dbc.Card([
        dbc.CardHeader("📈 Distribution des données numériques"),
        dbc.CardBody(plots)
    ])

@app.callback(
    Output('correlation-plot', 'children'),
    Input('table-selector', 'value')
)
def plot_correlation(table_id):
    df = load_sample_data(table_id, limit=10000)
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        return html.Div("🚫 Pas assez de colonnes numériques pour afficher les corrélations.")

    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Heatmap des corrélations")
    return dbc.Card([
        dbc.CardHeader("🔗 Corrélations entre variables"),
        dbc.CardBody(dcc.Graph(figure=fig))
    ])

@app.callback(
    Output('export-status', 'children'),
    Input('export-button', 'n_clicks')
)
def export_all_data(n_clicks):
    if n_clicks == 0:
        return ""

    tables = get_tables().keys()
    dfs_metadata = []
    dfs_stats = []
    dfs_qualite = []

    for table_id in tables:
        meta = get_table_metadata(table_id)
        meta["schema"]["table_id"] = table_id
        dfs_metadata.append(meta["schema"])

        stats = analyser_stats(table_id)
        dfs_stats.append(stats)

        qualite = analyser_qualite(table_id)
        dfs_qualite.append(qualite)

    df_metadata_final = pd.concat(dfs_metadata, ignore_index=True)
    df_stats_final = pd.concat(dfs_stats, ignore_index=True)
    df_qualite_final = pd.concat(dfs_qualite, ignore_index=True)

    exporter_vers_bigquery(df_metadata_final, "metadata_des_tables")
    exporter_vers_bigquery(df_stats_final, "statistiques_descriptives")
    exporter_vers_bigquery(df_qualite_final, "qualite_des_donnees")

    return html.Div("✅ Toutes les données ont été exportées dans le dataset cartographie.", className="alert alert-success")

# --- Lancement de l'application ---
if __name__ == '__main__':
    log("🏁 Démarrage de l'application Dash.")
    app.run(debug=True)