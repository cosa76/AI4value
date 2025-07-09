import os
import logging
from datetime import datetime
from google.cloud import bigquery
import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

# === Configuration ===
project_id = "sandbox-jndong"
dataset_id = "DataOriginal"
dataset_ml_id = "DataOriginal_ml" 
folder_path = "./reports"
SERVICE_ACCOUNT_KEY_JSON = "Load_file_in_bigquery/dbt_sa_bigquery.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY_JSON
os.makedirs(folder_path, exist_ok=True)

# Logger setup
logging.basicConfig(
    filename=os.path.join(folder_path, "log_ml_data.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

client = bigquery.Client(project=project_id)
dash_app = dash.Dash(__name__)

# === Fonctions utilitaires ===
def get_table_schemas():
    """Récupère toutes les tables du dataset et leurs schémas."""
    dataset_ref = client.dataset(dataset_id)
    tables = client.list_tables(dataset_ref)
    schemas = {}
    for table in tables:
        full_table_id = f"{project_id}.{dataset_id}.{table.table_id}"
        schema = client.get_table(full_table_id).schema
        schemas[table.table_id] = {field.name: field.field_type for field in schema}
    return schemas

def detect_column_types(schema):
    """Cartographie des colonnes par type."""
    numeric_cols = []
    string_cols = []
    date_cols = []
    bool_cols = []
    for col, dtype in schema.items():
        if dtype in ("FLOAT", "INTEGER"):
            numeric_cols.append(col)
        elif dtype == "STRING":
            string_cols.append(col)
        elif dtype in ("DATE", "DATETIME", "TIMESTAMP"):
            date_cols.append(col)
        elif dtype == "BOOLEAN":
            bool_cols.append(col)
    return {
        "numeric": numeric_cols,
        "string": string_cols,
        "date": date_cols,
        "bool": bool_cols
    }

# === Création des modèles BQML ===
def create_model(model_name, model_type, table_id, label_col=None, time_col=None):
    query = ""
    try:
        full_table = f"`{project_id}.{dataset_id}.{table_id}`"
        output_model = f"`{project_id}.{dataset_ml_id}.{model_name}`"

        if model_type == "LINEAR_REG":
            assert label_col and len(label_col) > 0, "Label column needed for linear regression."
            query = f"""
                CREATE OR REPLACE MODEL {output_model}
                OPTIONS(MODEL_TYPE='LINEAR_REG', LABELS=['{label_col}']) AS
                SELECT * FROM {full_table};
            """
        elif model_type == "LOGISTIC_REG":
            assert label_col and len(label_col) > 0, "Label column needed for logistic regression."
            query = f"""
                CREATE OR REPLACE MODEL {output_model}
                OPTIONS(MODEL_TYPE='LOGISTIC_REG', LABELS=['{label_col}']) AS
                SELECT * FROM {full_table};
            """
        elif model_type == "KMEANS":
            query = f"""
                CREATE OR REPLACE MODEL {output_model}
                OPTIONS(MODEL_TYPE='KMEANS') AS
                SELECT * EXCEPT({','.join(label_col)}) FROM {full_table};
            """
        elif model_type == "ARIMA_PLUS" and time_col:
            query = f"""
                CREATE OR REPLACE MODEL {output_model}
                OPTIONS(MODEL_TYPE='ARIMA_PLUS', TIME_SERIES_TIMESTAMP_COL='{time_col}', TIME_SERIES_DATA_COLUMN='sales');
            """
        elif model_type.startswith("BOOSTED_TREE_"):
            task = "REGRESSION" if "REGRESSOR" in model_type else "CLASSIFICATION"
            query = f"""
                CREATE OR REPLACE MODEL {output_model}
                OPTIONS(MODEL_TYPE='BOOSTED_TREE_{task}', LABELS=['{label_col}']) AS
                SELECT * FROM {full_table};
            """
        elif model_type.startswith("DNN_"):
            task = "REGRESSION" if "REGRESSOR" in model_type else "CLASSIFICATION"
            query = f"""
                CREATE OR REPLACE MODEL {output_model}
                OPTIONS(MODEL_TYPE='DNN_{task}', LABELS=['{label_col}']) AS
                SELECT * FROM {full_table};
            """
        elif model_type.startswith("AUTOML_"):
            task = "REGRESSION" if "REGRESSOR" in model_type else "CLASSIFICATION"
            query = f"""
                CREATE OR REPLACE MODEL {output_model}
                OPTIONS(MODEL_TYPE='AUTOML_{task}', LABELS=['{label_col}']) AS
                SELECT * FROM {full_table};
            """

        logging.info(f"[CREATE_MODEL] Exécution du modèle : {model_name}")
        job = client.query(query)
        job.result()
        return True, ""
    except Exception as e:
        error_msg = str(e)
        logging.error(f"[ERREUR] Modèle {model_name} non créé. Raison : {error_msg}")
        return False, error_msg

# === Analyse des données et création des modèles ===
def analyze_and_create_models():
    results = []
    schemas = get_table_schemas()

    for table_id, schema in schemas.items():
        col_info = detect_column_types(schema)
        numeric = col_info["numeric"]
        string = col_info["string"]
        date = col_info["date"]

        # Label possible (dernière colonne numérique ou booléenne)
        label_col = None
        for col in reversed(list(schema.keys())):
            if schema[col] in ("FLOAT", "INTEGER", "BOOLEAN"):
                label_col = col
                break

        time_col = date[0] if date else None

        print(f"\nAnalyse de la table : {table_id}")
        print(f"Colonnes numériques : {numeric}")
        print(f"Colonnes catégorielles : {string}")
        print(f"Colonnes temporelles : {date}")

        # Tentative de création des modèles
        if len(numeric) >= 2:
            success, msg = create_model("linear_reg_" + table_id, "LINEAR_REG", table_id, label_col=label_col)
            results.append({
                "table": table_id,
                "model": "LINEAR_REG",
                "success": success,
                "reason": msg or "Modèle linéaire appliqué sur colonnes numériques."
            })

        if len(numeric) >= 2:
            success, msg = create_model("logistic_reg_" + table_id, "LOGISTIC_REG", table_id, label_col=label_col)
            results.append({
                "table": table_id,
                "model": "LOGISTIC_REG",
                "success": success,
                "reason": msg or "Régression logistique appliquée pour classification binaire."
            })

        if len(numeric) >= 1:
            success, msg = create_model("kmeans_" + table_id, "KMEANS", table_id, label_col=[label_col])
            results.append({
                "table": table_id,
                "model": "KMEANS",
                "success": success,
                "reason": msg or "Clustering K-means appliqué sur colonnes numériques."
            })

        if time_col and len(numeric) >= 1:
            success, msg = create_model("arima_plus_" + table_id, "ARIMA_PLUS", table_id, time_col=time_col)
            results.append({
                "table": table_id,
                "model": "ARIMA_PLUS",
                "success": success,
                "reason": msg or "Modèle ARIMA+ appliqué sur série temporelle."
            })

        if len(numeric) >= 1:
            success, msg = create_model("xgboost_regressor_" + table_id, "BOOSTED_TREE_REGRESSOR", table_id, label_col=label_col)
            results.append({
                "table": table_id,
                "model": "BOOSTED_TREE_REGRESSOR",
                "success": success,
                "reason": msg or "XGBoost appliqué pour régression."
            })

        if len(numeric) >= 1 and label_col and schema[label_col] in ("FLOAT", "INTEGER"):
            success, msg = create_model("dnn_regressor_" + table_id, "DNN_REGRESSOR", table_id, label_col=label_col)
            results.append({
                "table": table_id,
                "model": "DNN_REGRESSOR",
                "success": success,
                "reason": msg or "Réseau de neurones (régression) appliqué."
            })

        if len(numeric) >= 1 and label_col and schema[label_col] == "BOOLEAN":
            success, msg = create_model("automl_classifier_" + table_id, "AUTOML_CLASSIFIER", table_id, label_col=label_col)
            results.append({
                "table": table_id,
                "model": "AUTOML_CLASSIFIER",
                "success": success,
                "reason": msg or "AutoML Classifier appliqué."
            })

    return pd.DataFrame(results)

# === Génération du rapport HTML ===
def generate_html_report(df_results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = os.path.join(folder_path, "ml_report.html")

    model_explanations = {
        "LINEAR_REG": "Régression linéaire : prédiction d'une variable continue via combinaison linéaire.",
        "LOGISTIC_REG": "Régression logistique : classification binaire basée sur probabilités.",
        "KMEANS": "K-Means clustering : groupement non supervisé basé sur similarités.",
        "ARIMA_PLUS": "Séries temporelles avec tendance/saisonnalité détectée automatiquement.",
        "BOOSTED_TREE_REGRESSOR": "Arbres boostés (XGBoost-like), efficace pour prédiction numérique.",
        "DNN_REGRESSOR": "Réseaux de neurones pour régression complexe.",
        "AUTOML_CLASSIFIER": "AutoML : optimisation automatisée pour classification.",
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("<html><head><title>Rapport ML</title></head><body>")
        f.write(f"<h1>Rapport de création des modèles ML - {now}</h1>")

        for table in df_results['table'].unique():
            f.write(f"<h2>Table : {table}</h2>")
            df_table = df_results[df_results['table'] == table]
            f.write("<ul>")
            for _, row in df_table.iterrows():
                status = "✅ Réussi" if row['success'] else "❌ Échoué"
                reason = row['reason']
                explanation = model_explanations.get(row['model'], "")
                f.write(f"<li><strong>{row['model']}</strong> : {status}<br>"
                        f"<em>{explanation}</em><br>"
                        f"Raison : {reason}</li><br>")
            f.write("</ul>")

        f.write("</body></html>")

    print(f"\n✅ Rapport généré dans : {report_path}")

# === Interface Dash pour visualisation ===
def run_dash_app(df_results):
    fig = px.bar(df_results, x='table', y='success', color='model', barmode='group',
                 title="Modèles ML créés avec succès par table")

    dash_app.layout = html.Div([
        html.H1("Visualisation des résultats ML"),
        dcc.Graph(
            id='results-graph',
            figure=fig
        )
    ])

    dash_app.run(debug=True)

# === Lancement du script ===
if __name__ == "__main__":
    df = analyze_and_create_models()
    print("\nRésumé des modèles ML créés :")
    print(df)
    generate_html_report(df)
    run_dash_app(df)