# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import plotly.express as px
from google.cloud import bigquery
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import pearsonr
import gradio as gr

# --- Configuration ---
CREDENTIAL = "/Users/jndong/AI4value_good/Load_file_in_bigquery/dbt_sa_bigquery.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIAL
PROJECT_ID = "sandbox-jndong"
DATASET_ID = "dev"
TABLE_ID = "bannk_data"

# --- Chargement des données depuis BigQuery ---
def load_data_from_bigquery():
    client = bigquery.Client(project=PROJECT_ID)
    query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` LIMIT 10000"
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
    summary = {}
    for col in df.columns:
        # Gestion des types non hashables comme dict ou list
        if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            unique_count = "Non applicable (type complexe)"
            sample_values = df[col].dropna().head(5).astype(str).tolist()
        else:
            unique_count = df[col].nunique()
            sample_values = df[col].dropna().sample(min(5, len(df)), random_state=42).tolist() if len(df) > 0 else []

        summary[col] = {
            'dtype': str(df[col].dtype),
            'missing_percent': round(df[col].isnull().sum() / len(df) * 100, 2) if len(df) > 0 else 100,
            'unique_count': unique_count,
            'sample_values': sample_values
        }
    return summary

# --- Classification des variables ---
def classify_variable_types(summary):
    numeric_cols = []
    categorical_cols = []
    text_cols = []
    datetime_cols = []
    total_rows = len(df)

    for col, info in summary.items():
        dtype = info['dtype']
        unique_ratio = info['unique_count'] / total_rows if isinstance(info['unique_count'], int) else 0
        if dtype in ['int64', 'float64', 'Int64', 'Float64']:
            numeric_cols.append(col)
        elif dtype == 'object':
            if isinstance(info['unique_count'], int) and info['unique_count'] < 30 and unique_ratio < 0.05:
                categorical_cols.append(col)
            else:
                text_cols.append(col)
        elif 'datetime' in dtype:
            datetime_cols.append(col)
    return {'numeric': numeric_cols, 'categorical': categorical_cols, 'text': text_cols, 'datetime': datetime_cols}

# --- Identifier une variable cible possible ---
def identify_target(df, var_types):
    for col in var_types['categorical']:
        if df[col].nunique() >= 2 and df[col].nunique() <= 20:
            return (col, 'classification')
    for col in var_types['numeric']:
        if df[col].nunique() > 10:
            return (col, 'regression')
    return (None, None)

# --- Évaluation des relations ---
def evaluate_relations(df, target_col):
    if not target_col or target_col not in df.columns:
        return {}
    results = {}
    y = df[target_col]
    if y.dtype.name in ['object', 'category']:
        y = y.astype('category').cat.codes

    for feat in df.columns:
        if feat == target_col:
            continue
        x = df[feat]

        # Gestion des types différents
        if x.dtype.name in ['int64', 'float64', 'Int64', 'Float64']:
            x_filled = x.astype(float).fillna(x.astype(float).median()).values.reshape(-1, 1)
        elif 'datetime' in str(x.dtype):
            x_filled = x.fillna(pd.Timestamp('1900-01-01')).astype('category').cat.codes.to_numpy().reshape(-1, 1)
        else:
            x_filled = x.fillna('MISSING').astype('category').cat.codes.to_numpy().reshape(-1, 1)

        try:
            if y.dtype.name in ['int64', 'category']:
                mi = mutual_info_classif(x_filled, y, discrete_features=True, random_state=42)[0]
                results[feat] = {'type': 'classification', 'mutual_info': mi}
            else:
                mi = mutual_info_regression(x_filled, y, random_state=42)[0]
                corr, _ = pearsonr(x_filled.ravel(), y)
                results[feat] = {'type': 'regression', 'mutual_info': mi, 'pearson_corr': corr}
        except Exception as e:
            print(f"Erreur pour {feat} : {e}")
    return results

# --- Suggestion de cas d'usage ---
def suggest_use_cases(var_types, target_info):
    use_cases = []
    if target_info[1] == 'classification':
        use_cases.append("Classification (ex: détection de fraude, segmentation client)")
    elif target_info[1] == 'regression':
        use_cases.append("Régression (ex: prévision des ventes, estimation de prix)")
    if not target_info[0]:
        use_cases.append("Clustering (ex: segmentation client, détection d'anomalies)")
    if var_types['datetime']:
        use_cases.append("Séries temporelles (ex: prévision de demande, analyse de tendances)")
    if var_types['text']:
        use_cases.append("Traitement du langage naturel (ex: analyse de sentiment, classification de documents)")
    return use_cases

# --- Variables globales ---
df = pd.DataFrame()
feature_relations = {}

# --- Fonction principale pour Gradio ---
def run_analysis():
    global df, feature_relations
    df = load_data_from_bigquery()
    if df.empty:
        return "❌ Le dataframe est vide. Impossible de continuer.", [], "", px.bar(title="Erreur")

    summary = analyze_data(df)
    var_types = classify_variable_types(summary)
    target_info = identify_target(df, var_types)
    feature_relations = evaluate_relations(df, target_info[0]) if target_info[0] else {}
    use_cases = suggest_use_cases(var_types, target_info)

    # Retourner les résultats
    return "\n".join(use_cases), var_types.get('numeric', []), target_info[0], generate_feature_importance_plot(feature_relations)

def generate_histogram(selected_col):
    if selected_col in df.columns:
        return px.histogram(df, x=selected_col, title=f"Histogramme - {selected_col}")
    return px.histogram(title="Aucune colonne sélectionnée")

def generate_feature_importance_plot(relations):
    if not relations:
        return px.bar(title="Aucune donnée disponible")
    feat_names = list(relations.keys())
    mi_values = [relations[f].get('mutual_info', 0) for f in feat_names]
    return px.bar(x=feat_names, y=mi_values, title="Importance des Features (Info Mutuelle)")

# --- Interface Gradio ---
with gr.Blocks(title="🔍 Détecteur de Cas d'Usage ML") as demo:
    gr.Markdown("## 📊 Analyse Automatique des Données et Détection de Cas d'Usage Machine Learning")

    with gr.Row():
        btn_run = gr.Button("🔄 Lancer l'Analyse", variant="primary")

    gr.Markdown("### 🎯 Cas d'Usage Suggérés")
    output_use_cases = gr.Textbox(label="", lines=5)

    gr.Markdown("### 📈 Histogramme de Variable Numérique")
    with gr.Row():
        dropdown_col = gr.Dropdown(choices=[], label="Sélectionnez une colonne numérique")
        histo_output = gr.Plot()

    gr.Markdown("### 🔗 Importance des Features par Info Mutuelle")
    importance_plot = gr.Plot()

    # Callbacks
    def on_run_click():
        use_cases_str, numeric_cols, target_col, fig = run_analysis()
        return use_cases_str, gr.update(choices=numeric_cols), fig

    def update_histogram(selected_col):
        return generate_histogram(selected_col)

    btn_run.click(fn=on_run_click, outputs=[output_use_cases, dropdown_col, importance_plot])
    dropdown_col.change(fn=update_histogram, inputs=dropdown_col, outputs=histo_output)

# Lancer l'application Gradio
demo.launch(share=False)