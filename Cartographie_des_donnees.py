import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import bigquery
from collections import defaultdict
import os

# Configuration
CREDENTIAL = "/Users/jndong/AI4value_good/Load_file_in_bigquery/dbt_sa_bigquery.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIAL

# --- Configuration ---
PROJECT_ID =  "sandbox-jndong"
DATASET_ID = "DataOriginal"

os.makedirs("rapport_dash", exist_ok=True)

# --- Initialisation client BigQuery ---
client = bigquery.Client(project=PROJECT_ID)

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
    return client.query(query).to_dataframe()

# --- Application Dash ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Pour déploiement Flask/Gunicorn

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
    table_info = [
        html.P(f"Table : {meta['nom_complet']}"),
        html.P(f"Nombre de lignes : {meta['lignes']}"),
        html.P(f"Nombre de colonnes : {len(df_schema)}")
    ]
    return dbc.Card([
        dbc.CardHeader("📚 Métadonnées"),
        dbc.CardBody([
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

# --- Lancement de l'application ---
if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run(debug=True)