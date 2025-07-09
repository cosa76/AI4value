import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
import plotly.express as px

# Définition de la clé d'authentification Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Load_file_in_bigquery/dbt_sa_bigquery.json"


def rename_duplicate_columns(df):
    """
    Renomme les colonnes en doublon dans un DataFrame.
    Exemple: si deux colonnes 'col1', alors deviennent 'col1' et 'col1_1'.
    """
    cols = df.columns.tolist()
    new_cols = []
    seen = {}

    for col in cols:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df


def authenticate_bigquery():
    """
    Authentification auprès de BigQuery.
    """
    try:
        client = bigquery.Client()
        print("Authentification réussie à BigQuery!")
        return client
    except Exception as e:
        print(f"Erreur d'authentification: {e}")
        return None


def list_views(client, project_id, dataset_id):
    """
    Liste toutes les vues disponibles dans le dataset spécifié.
    """
    dataset_ref = client.dataset(dataset_id, project=project_id)
    try:
        tables = list(client.list_tables(dataset_ref))
        views = [table.table_id for table in tables if table.table_type == 'VIEW']
        if not views:
            print(f"Aucune vue trouvée dans le dataset {dataset_id}")
        return views
    except Exception as e:
        print(f"Erreur lors de la récupération des vues: {e}")
        return []


def get_view_query(client, project_id, dataset_id, view_id):
    """
    Récupère la requête SQL source d'une vue existante.
    """
    view_ref = client.dataset(dataset_id, project=project_id).table(view_id)
    try:
        view = client.get_table(view_ref)
        return view.view_query
    except Exception as e:
        print(f"Impossible de récupérer la requête de la vue {view_id}: {e}")
        return None


def generate_clean_sql_from_query(sql_query, df):
    """
    Génère une requête SQL propre avec des colonnes renommées pour éviter les doublons.
    """
    if not sql_query or not df.columns.tolist():
        return sql_query

    original_columns = df.columns.tolist()
    cleaned_columns = []
    seen = {}

    for col in original_columns:
        if col in seen:
            seen[col] += 1
            cleaned_columns.append(f"`{col}` AS `{col}_{seen[col]}`")
        else:
            seen[col] = 0
            cleaned_columns.append(f"`{col}`")

    select_clause = ", ".join(cleaned_columns)
    return f"SELECT {select_clause} FROM ({sql_query})"


def recreate_view(client, project_id, dataset_id, view_id, clean_sql):
    """
    Recrée une vue dans BigQuery avec une requête SQL nettoyée (sans doublons).
    """
    view_ref = client.dataset(dataset_id, project=project_id).table(view_id)
    try:
        view = bigquery.Table(view_ref)
        view.view_query = clean_sql
        view = client.update_table(view, ["view_query"])
        print(f"✅ Vue '{view_id}' mise à jour sans doublons.")
    except Exception as e:
        print(f"❌ Erreur lors de la mise à jour de la vue '{view_id}': {e}")


def get_view_data(client, project_id, dataset_id, view_id, limit=1000):
    """
    Récupère les données d'une vue spécifique et corrige les doublons de colonnes.
    """
    query = f"""
    SELECT * 
    FROM `{project_id}.{dataset_id}.{view_id}`
    LIMIT {limit}
    """
    try:
        df = client.query(query).to_dataframe()
        df = rename_duplicate_columns(df)
        print(f"✅ Données récupérées avec succès: {len(df)} lignes")
        return df
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des données de la vue {view_id}: {e}")
        return None


def analyze_dataframe(df, view_name):
    """
    Analyse exploratoire des données et création de visualisations (simplifiée ici).
    """
    if df is None or df.empty:
        print("⚠️ Aucune donnée à analyser")
        return

    print(f"\n--- Analyse de la vue: {view_name} ---")
    print(f"Dimensions: {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print("\nTypes de données:")
    print(df.dtypes)

    viz_folder = f"visualisations_{view_name}"
    os.makedirs(viz_folder, exist_ok=True)

    # Histogramme simple
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        col = numeric_cols[0]
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution de {col}')
            plt.tight_layout()
            plt.savefig(f"{viz_folder}/hist_{col}.png")
            plt.close()
            print(f"📊 Histogramme généré pour {col}")
        except Exception as e:
            print(f"❌ Échec de génération de l'histogramme pour {col}: {e}")

    print(f"\nVisualisations sauvegardées dans le dossier '{viz_folder}'")


def main():
    # Configuration
    project_id = "sandbox-jndong"
    dataset_id = "DataOriginal_matching"

    # Authentification
    client = authenticate_bigquery()
    if not client:
        return

    # Récupération des vues
    views = list_views(client, project_id, dataset_id)
    if not views:
        return

    print("\nVues disponibles:")
    for i, view in enumerate(views, 1):
        print(f"{i}. {view}")

    # Sélection de la vue à traiter
    try:
        choice = int(input("\nEntrez le numéro de la vue à analyser (0 pour toutes): "))
        if choice == 0:
            selected_views = views
        elif 1 <= choice <= len(views):
            selected_views = [views[choice - 1]]
        else:
            print("❌ Choix invalide")
            return
    except ValueError:
        print("❌ Veuillez entrer un nombre valide")
        return

    # Traitement des vues sélectionnées
    all_viz_folders = []
    for view in selected_views:
        print(f"\n🔍 Traitement de la vue: {view}")

        # Récupération des données
        df = get_view_data(client, project_id, dataset_id, view)
        if df is None:
            continue

        # Analyse exploratoire
        viz_folder = analyze_dataframe(df, view)
        if viz_folder:
            all_viz_folders.append(viz_folder)

        # Récupération de la requête SQL originale
        sql_query = get_view_query(client, project_id, dataset_id, view)
        if not sql_query:
            continue

        # Nettoyage de la requête SQL
        clean_sql = generate_clean_sql_from_query(sql_query, df)
        if not clean_sql:
            print(f"❌ Impossible de générer une requête SQL propre pour {view}")
            continue

        # Mise à jour de la vue dans BigQuery
        recreate_view(client, project_id, dataset_id, view, clean_sql)

    if all_viz_folders:
        print("\n📊 Résumé des analyses:")
        for folder in all_viz_folders:
            print(f"- Visualisations générées dans : {folder}")


if __name__ == "__main__":
    main()