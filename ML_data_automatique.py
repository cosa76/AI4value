import dash, os
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import pearsonr

# Désactiver les warnings non critiques
import warnings
warnings.filterwarnings("ignore", message="BigQuery Storage module not found")

# Configuration
CREDENTIAL = "/Users/jndong/AI4value_good/Load_file_in_bigquery/dbt_sa_bigquery.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIAL
PROJECT_ID =  "sandbox-jndong"
DATASET_ID = "dev"
TABLE_ID = "bannk_data"

DATASET_ML = "DataOriginal_ml"
TABLE_RESULT_ML_ID = "table_result_ml"

# --- Chargement des données depuis BigQuery ---
def load_data_from_bigquery():
    client = bigquery.Client(project=PROJECT_ID)
    query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        LIMIT 10000
    """
    print("Chargement des données depuis BigQuery...")
    try:
        df = client.query(query).to_dataframe()
        print(f"Nombre de lignes chargées : {len(df)}")
        return df
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return pd.DataFrame()

# --- Analyse basique du jeu de données ---
def analyze_data(df):
    summary = {
        'total_rows': len(df),
        'columns': {}
    }

    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = df[col].isnull().sum() / len(df) * 100 if len(df) > 0 else 100.0
        unique_values = df[col].nunique(dropna=True)
        sample_values = []

        non_null = df[col].dropna()
        if not non_null.empty:
            sample_count = min(5, len(non_null))
            sample_values = non_null.sample(sample_count, random_state=42).tolist()
        else:
            sample_values = ["Colonne vide"]

        summary['columns'][col] = {
            'dtype': dtype,
            'missing_percent': round(missing, 2),
            'unique_count': unique_values,
            'sample_values': sample_values
        }

    return summary

# --- Classification des variables ---
def classify_variable_types(summary):
    numeric_cols = []
    categorical_cols = []
    text_cols = []
    datetime_cols = []

    total_rows = summary.get('total_rows', 0)

    for col, info in summary['columns'].items():
        if info['dtype'] in ['int64', 'float64']:
            numeric_cols.append(col)
        elif info['dtype'] == 'object':
            if info['unique_count'] < 30 and info['unique_count'] / total_rows < 0.05 if total_rows > 0 else False:
                categorical_cols.append(col)
            else:
                text_cols.append(col)
        elif 'datetime' in info['dtype']:
            datetime_cols.append(col)

    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'text': text_cols,
        'datetime': datetime_cols
    }

# --- Identifier une variable cible possible ---
def identify_target(df, var_types):
    potential_targets = []

    # Classification : si colonne catégorielle avec au moins 2 classes
    for col in var_types['categorical']:
        if df[col].nunique() >= 2 and df[col].nunique() <= 20:
            potential_targets.append((col, 'classification'))

    # Régression : si numérique et assez dispersé
    for col in var_types['numeric']:
        if df[col].nunique() > 10:
            potential_targets.append((col, 'regression'))

    # Si aucune target trouvée, on suppose clustering
    if not potential_targets:
        return None

    return potential_targets[0]

# --- Évaluation des relations ---
def evaluate_relations(df, target_col, var_types):
    if target_col is None or target_col not in df.columns:
        return {}

    results = {}
    features = [col for col in df.columns if col != target_col]
    y = df[target_col]

    for feat in features:
        if feat not in df.columns:
            continue

        # Si la feature est numérique, on utilise median()
        if feat in var_types['numeric']:
            x = df[feat].fillna(df[feat].median()).values.reshape(-1, 1)

        else:
            # Pour les colonnes catégorielles/textuelles
            mode_value = df[feat].mode()[0] if not df[feat].mode().empty else ""
            x = df[feat].fillna(mode_value).astype(str).astype('category').cat.codes.to_numpy().reshape(-1, 1)

        try:
            if y.dtype.name in ['int64', 'category', 'object']:
                mi = mutual_info_classif(x, y, discrete_features=True, random_state=42)[0]
                results[feat] = {'type': 'classification', 'mutual_info': mi}
            else:
                mi = mutual_info_regression(x, y, random_state=42)[0]
                corr, _ = pearsonr(x.ravel(), y)
                results[feat] = {'type': 'regression', 'mutual_info': mi, 'pearson_corr': corr}
        except Exception as e:
            print(f"Erreur d’évaluation pour {feat} : {e}")

    return results

# --- Suggestion de cas d'usage ---
def suggest_use_cases(var_types, target_info, feature_relations):
    use_cases = []

    if target_info and target_info[1] == 'classification':
        use_cases.append("Classification (ex: détection de fraude, segmentation client)")
    if target_info and target_info[1] == 'regression':
        use_cases.append("Régression (ex: prévision des ventes, estimation de prix)")
    if not target_info:
        use_cases.append("Clustering (ex: segmentation client, détection d'anomalies)")
    if len(var_types['datetime']) > 0:
        use_cases.append("Séries temporelles (ex: prévision de demande, analyse de tendances)")
    if len(var_types['text']) > 0:
        use_cases.append("Traitement du langage naturel (ex: analyse de sentiment, classification de documents)")

    return use_cases

# --- Exportation vers BigQuery ---
def export_to_bigquery(results_dict):
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = client.dataset(DATASET_ML).table(TABLE_RESULT_ML_ID)

    rows = []
    for key, value in results_dict.items():
        if isinstance(value, list):
            for v in value:
                rows.append({"key": key, "value": str(v)})
        elif isinstance(value, dict):
            for k_sub, v_sub in value.items():
                rows.append({"key": f"{key}.{k_sub}", "value": str(v_sub)})
        else:
            rows.append({"key": key, "value": str(value)})

    df_result = pd.DataFrame(rows)

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=[
            bigquery.SchemaField("key", "STRING"),
            bigquery.SchemaField("value", "STRING"),
        ],
    )

    try:
        job = client.load_table_from_dataframe(df_result, table_ref, job_config=job_config)
        job.result()
        print(f"Résultats sauvegardés dans {PROJECT_ID}.{DATASET_ML}.{TABLE_RESULT_ML_ID}")
    except Exception as e:
        print(f"Erreur lors de l'export vers BigQuery : {e}")

# --- Main ---
df = load_data_from_bigquery()
if not df.empty:
    summary = analyze_data(df)
    var_types = classify_variable_types(summary)
    target_info = identify_target(df, var_types)
    feature_relations = evaluate_relations(df, target_info[0], var_types) if target_info else {}
    use_cases = suggest_use_cases(var_types, target_info, feature_relations)

    # Préparer les résultats pour l'export
    results_dict = {
        "types_variables": var_types,
        "variable_cible": target_info[0] if target_info else "Aucune",
        "cas_usage": use_cases,
        "relations": feature_relations
    }

    export_to_bigquery(results_dict)

else:
    print("Le dataframe est vide. Impossible de continuer.")
    exit()

# --- Initialisation de l'app Dash ---
app = dash.Dash(__name__)
server = app.server  # Pour déploiement Heroku ou autre

# --- Layout statique et complet (pas d'append multiple)
app.layout = html.Div([
    html.H1("🔍 Identification Automatique des Cas d'Usage Machine Learning"),

    html.Ul([html.Li(uc) for uc in use_cases]),

    html.H3("📊 Visualisation des Variables Numériques"),
    dcc.Dropdown(
        id='numeric-column-dropdown',
        options=[{'label': col, 'value': col} for col in var_types.get('numeric', [])],
        value=var_types['numeric'][0] if var_types.get('numeric') else None
    ),
    dcc.Graph(id='numeric-histogram'),

    html.H3("🔗 Relations avec la Variable Cible"),
    dcc.Graph(id='feature-importance-graph'),
])

# --- Callbacks Dash ---
@app.callback(
    Output('numeric-histogram', 'figure'),
    Input('numeric-column-dropdown', 'value')
)
def update_histogram(selected_column):
    if selected_column and selected_column in df.columns:
        fig = px.histogram(df, x=selected_column, title=f"Histogramme - {selected_column}")
        return fig
    return px.histogram(title="Aucune colonne sélectionnée")

@app.callback(
    Output('feature-importance-graph', 'figure'),
    Input('numeric-column-dropdown', 'value')
)
def update_feature_importance(_):
    if feature_relations:
        feat_names = list(feature_relations.keys())
        mi_values = [feature_relations[f].get('mutual_info', 0) for f in feat_names]
        fig = px.bar(x=feat_names, y=mi_values, title="Importance des Features (Info Mutuelle)")
        return fig
    return px.bar(title="Aucune relation calculée")

# Lancer l'app
if __name__ == "__main__":
    app.run(debug=True)