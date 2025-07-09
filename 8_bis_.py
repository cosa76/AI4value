import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from dash import Dash, html, dcc, dash_table
import plotly.express as px
import mlflow
import mlflow.sklearn
import joblib
from jinja2 import Template
import webbrowser

# === Configuration du logging ===
logging.basicConfig(
    filename='log_ml_data.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === Configuration BigQuery ===
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "dbt_sa_bigquery.json"
client_bq = bigquery.Client(project="sandbox-jndong")
project_id = "sandbox-jndong"
dataset_id = "DataOriginal"
folder_path = "./results/"
model_folder = os.path.join(folder_path, "trained_models")

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# Activer MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("ML_Pipeline_Experiment")

# === Exploration des tables du dataset ===
def list_tables_in_dataset(dataset_id):
    tables = client_bq.list_tables(dataset_id)
    return [table.table_id for table in tables]

# === Lecture des données d'une table ===
def load_table_data(table_id):
    query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}` LIMIT 1000"
    df = client_bq.query(query).to_dataframe()
    return df

# === Nettoyage avancé des données ===
def clean_data(df):
    # Supprimer les colonnes avec trop de NaN (>90%)
    threshold_col = len(df) * 0.1
    df = df.dropna(thresh=threshold_col, axis=1)

    # Remplacer les NaN par la médiane pour les numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Encoder les colonnes catégorielles
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes

    logging.info("Données nettoyées avec succès.")
    return df

# === Analyse des colonnes ===
def analyze_columns(df):
    metadata = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            metadata[col] = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            metadata[col] = 'datetime'
        elif df[col].dtype == 'object':
            metadata[col] = 'categorical'
        else:
            metadata[col] = 'unknown'
    return metadata

# === Sélection de la meilleure variable cible ===
def select_best_target(df, metadata):
    numeric_cols = [k for k, v in metadata.items() if v == 'numeric']
    if len(numeric_cols) < 2:
        return None
    target_col = df[numeric_cols].iloc[:, -1].name
    return target_col

# === Création de modèles ML ===
def create_models(df, metadata):
    results = []

    numeric_cols = [k for k, v in metadata.items() if v == 'numeric']
    if len(numeric_cols) < 2:
        logging.warning("Pas assez de colonnes numériques pour les modèles.")
        return results

    target_col = select_best_target(df, metadata)
    if not target_col:
        logging.warning("Aucune variable cible valide trouvée.")
        return results

    X = df[numeric_cols].drop(columns=[target_col], errors='ignore')
    y = df[target_col]

    combined = pd.concat([X, y], axis=1).dropna()
    if len(combined) < 2:
        logging.warning("Pas assez de données après nettoyage.")
        return results

    X = combined[X.columns]
    y = combined[y.name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with mlflow.start_run():
        input_example = X.head(1).to_numpy().reshape(1, -1)

        # Régression linéaire
        try:
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            model_path = os.path.join(model_folder, "linear_regression.pkl")
            joblib.dump(lr, model_path)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(lr, "linear_regression", input_example=input_example)
            results.append({'model': 'LINEAR_REG', 'rmse': rmse, 'r2': r2})
            logging.info("Modèle LINEAR_REG créé avec succès.")
        except Exception as e:
            logging.error(f"LINEAR_REG échoué: {str(e)}")

        # XGBoost Regressor
        try:
            xgb = XGBRegressor(enable_categorical=True)
            xgb.fit(X_train, y_train)
            y_pred = xgb.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            model_path = os.path.join(model_folder, "xgboost_regressor.pkl")
            joblib.dump(xgb, model_path)
            mlflow.log_metric("xgb_rmse", rmse)
            mlflow.log_metric("xgb_r2", r2)
            mlflow.sklearn.log_model(xgb, "xgboost_regressor", input_example=X_train[:1])
            results.append({'model': 'XGBOOST_REG', 'rmse': rmse, 'r2': r2})
            logging.info("Modèle XGBOOST_REG créé avec succès.")
        except Exception as e:
            logging.error(f"XGBOOST_REG échoué: {str(e)}")

        # Neural Network simple
        try:
            model = Sequential([
                Input(shape=(X_train.shape[1],)),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=10, verbose=0)
            loss = model.evaluate(X_test, y_test, verbose=0)
            model_path = os.path.join(model_folder, "nn_regressor.h5")
            model.save(model_path)
            mlflow.tensorflow.log_model(model, "neural_network", input_example=X_train[:1])
            results.append({'model': 'NN_REG', 'loss': loss})
            logging.info("Modèle NN_REG créé avec succès.")
        except Exception as e:
            logging.error(f"NeuralNetwork échoué: {str(e)}")

    return results

# === Génération d’un rapport HTML ===
def generate_html_report(results_df, df_summary):
    template = """
    <html>
    <head><title>Rapport ML</title></head>
    <body>
        <h1>Résultats des modèles ML</h1>
        <h2>Données analysées</h2>
        {{ df }}
        <h2>Métriques</h2>
        {% if not results.empty %}
        <table border="1" class="dataframe">
          <thead>
            <tr style="text-align: right;">
              {% for col in results.columns %}
              <th>{{ col }}</th>
              {% endfor %}
            </tr>
          </thead>
          <tbody>
            {% for row in results.values %}
            <tr>
              {% for val in row %}
              <td>{{ val }}</td>
              {% endfor %}
            </tr>
            {% endfor %}
          </tbody>
        </table>
        {% endif %}
    </body>
    </html>
    """

    with open(os.path.join(folder_path, "rapport_ml.html"), "w") as f:
        f.write(Template(template).render(
            df=df_summary.to_html(index=False),
            results=results_df
        ))

    logging.info("Rapport HTML généré avec succès.")
    webbrowser.open_new_tab(os.path.join(folder_path, "rapport_ml.html"))

# === Visualisation Dash ===
def run_dash_app(results_df):
    if results_df.empty:
        logging.warning("Aucun modèle à afficher via Dash.")
        print("Aucun modèle à afficher.")
        return

    app = Dash(__name__)
    server = app.server

    app.layout = html.Div([
        html.H1("Performance des modèles ML"),
        html.Label("Filtrer par type de modèle"),
        dcc.Dropdown(
            id='model-filter',
            options=[{'label': model, 'value': model} for model in results_df['model'].unique()],
            value=results_df['model'].unique().tolist(),
            multi=True
        ),
        html.Button("Exporter en CSV", id="export-button"),
        dcc.Download(id="download-csv"),
        dcc.Graph(id='performance-graph'),
        dash_table.DataTable(
            data=results_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in results_df.columns],
            page_size=10
        )
    ])

    @app.callback(
        Output('performance-graph', 'figure'),
        Input('model-filter', 'value')
    )
    def update_graph(selected_models):
        filtered_df = results_df[results_df['model'].isin(selected_models)]
        metric_col = 'rmse' if 'rmse' in filtered_df.columns else 'loss'
        fig = px.bar(filtered_df, x='model', y=metric_col, title=f"Performance ({metric_col})")
        return fig

    @app.callback(
        Output("download-csv", "data"),
        Input("export-button", "n_clicks"),
        prevent_initial_call=True
    )
    def export_to_csv(_):
        csv_path = os.path.join(folder_path, "ml_results_export.csv")
        results_df.to_csv(csv_path, index=False)
        return dcc.send_file(csv_path)

    app.run(debug=True)

# === Boucle principale ===
def main():
    all_results = []
    tables = list_tables_in_dataset(dataset_id)
    for table in tables:
        logging.info(f"Traitement de la table {table}")
        df = load_table_data(table)
        df_clean = clean_data(df)
        metadata = analyze_columns(df_clean)
        logging.info(f"Analyse des colonnes: {metadata}")
        result = create_models(df_clean, metadata)
        all_results.extend(result)

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df.to_csv(os.path.join(folder_path, "ml_results.csv"), index=False)
        logging.info("Tous les modèles ont été créés et les résultats sauvegardés.")
    else:
        logging.warning("Aucun modèle n'a été généré.")

    # Générer un rapport HTML
    df_summary = pd.DataFrame({"Colonne": df.columns, "Type": metadata.values()})
    generate_html_report(results_df, df_summary)

    # Lancer Dash
    run_dash_app(results_df)

if __name__ == "__main__":
    main()