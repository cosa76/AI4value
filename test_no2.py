from google.cloud import bigquery
import pandas as pd
import os

# Définition de la clé d'authentification Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./dbt_sa_bigquery.json"

# Initialisation du client BigQuery
client = bigquery.Client()

# Identifiants du projet et des datasets
PROJECT_ID = "sandbox-jndong"
DATASET_ID = "DataOriginal"
DATASET_VIEW_ID = "DataOriginal_matching"  # Dataset où seront créées les vues


def delete_dataset(project_id: str, dataset_id: str, delete_contents: bool = True):
    dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
    try:
        client.delete_dataset(dataset_ref, delete_contents=delete_contents, not_found_ok=False)
        print(f"✅ Dataset supprimé : {project_id}.{dataset_id}")
    except Exception as e:
        print(f"❌ Erreur lors de la suppression du dataset : {e}")


def create_dataset_if_not_exists():
    dataset_id = f"{client.project}.{DATASET_VIEW_ID}"
    try:
        client.get_dataset(dataset_id)
        print(f"Dataset {dataset_id} already exists.")
    except Exception:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"
        client.create_dataset(dataset)
        print(f"Dataset {dataset_id} created.")


def get_tables_metadata():
    tables_metadata = []
    dataset_ref = client.dataset(DATASET_ID, project=PROJECT_ID)
    tables = list(client.list_tables(dataset_ref))

    for table in tables:
        table_ref = dataset_ref.table(table.table_id)
        table_info = client.get_table(table_ref)
        columns = {field.name: field.field_type for field in table_info.schema}
        tables_metadata.append({
            "table_name": table_info.table_id,
            "columns": columns  # Stocker les noms et types des colonnes
        })
    
    return pd.DataFrame(tables_metadata)


def compare_column_values(table1: str, table2: str, column1: str, column2: str, threshold: float = 0.8):
    """
    Compare les valeurs des colonnes entre deux tables et retourne True si elles sont similaires.
    :param table1: Nom de la première table.
    :param table2: Nom de la deuxième table.
    :param column1: Nom de la colonne dans la première table.
    :param column2: Nom de la colonne dans la deuxième table.
    :param threshold: Seuil de similarité (par défaut 0.8).
    :return: True si les valeurs sont similaires, sinon False.
    """
    # Récupérer les types des colonnes
    table1_columns = client.get_table(f"{PROJECT_ID}.{DATASET_ID}.{table1}").schema
    table2_columns = client.get_table(f"{PROJECT_ID}.{DATASET_ID}.{table2}").schema

    type1 = next((field.field_type for field in table1_columns if field.name == column1), None)
    type2 = next((field.field_type for field in table2_columns if field.name == column2), None)

    # Vérifier si les types sont compatibles
    if type1 != type2:
        print(f"Types incompatibles pour {column1} ({type1}) et {column2} ({type2}).")
        return False

    # Construire la requête SQL pour comparer les valeurs
    query = f"""
    WITH t1 AS (
        SELECT DISTINCT {column1} FROM `{PROJECT_ID}.{DATASET_ID}.{table1}`
    ),
    t2 AS (
        SELECT DISTINCT {column2} FROM `{PROJECT_ID}.{DATASET_ID}.{table2}`
    )
    SELECT 
        COUNT(*) AS matching_count,
        (SELECT COUNT(*) FROM t1) AS total_t1,
        (SELECT COUNT(*) FROM t2) AS total_t2
    FROM t1
    INNER JOIN t2
    ON t1.{column1} = t2.{column2}
    """
    try:
        query_job = client.query(query)
        result = query_job.result()
        row = next(result)

        matching_count = row["matching_count"]
        total_t1 = row["total_t1"]
        total_t2 = row["total_t2"]

        similarity_score = matching_count / max(total_t1, total_t2)
        return similarity_score >= threshold
    except Exception as e:
        print(f"Erreur lors de la comparaison des valeurs pour {column1} et {column2}: {e}")
        return False


def find_similar_columns(tables_metadata):
    similar_columns = []

    for i in range(len(tables_metadata)):
        for j in range(i + 1, len(tables_metadata)):
            table1 = tables_metadata.iloc[i]["table_name"]
            table2 = tables_metadata.iloc[j]["table_name"]
            cols1 = tables_metadata.iloc[i]["columns"]
            cols2 = tables_metadata.iloc[j]["columns"]

            # Comparaison par nom
            common_columns_by_name = set(cols1.keys()).intersection(cols2.keys())
            for col in common_columns_by_name:
                similar_columns.append({
                    "table1": table1,
                    "table2": table2,
                    "column1": col,
                    "column2": col,
                    "similarity_type": "by_name"
                })

            # Comparaison par valeurs
            for col1, type1 in cols1.items():
                for col2, type2 in cols2.items():
                    if col1 != col2 and type1 == type2:  # Vérifier la compatibilité des types
                        if compare_column_values(table1, table2, col1, col2):
                            similar_columns.append({
                                "table1": table1,
                                "table2": table2,
                                "column1": col1,
                                "column2": col2,
                                "similarity_type": "by_value"
                            })
    
    return pd.DataFrame(similar_columns)


def create_views_for_similar_columns(similar_columns_df, tables_metadata):
    for _, row in similar_columns_df.iterrows():
        table1 = row["table1"]
        table2 = row["table2"]
        column1 = row["column1"]
        column2 = row["column2"]

        view_name = f"view_{table1}_to_{table2}_on_{column1}_and_{column2}"
        view_id = f"{PROJECT_ID}.{DATASET_VIEW_ID}.{view_name}"

        cols1 = tables_metadata[tables_metadata["table_name"] == table1]["columns"].iloc[0]
        cols2 = tables_metadata[tables_metadata["table_name"] == table2]["columns"].iloc[0]

        renamed_cols1 = [f"t1.{col} AS t1_{col}" for col in cols1]
        renamed_cols2 = [f"t2.{col} AS t2_{col}" for col in cols2]

        query = f"""
        CREATE OR REPLACE VIEW `{view_id}` AS
        SELECT 
            {', '.join(renamed_cols1)},
            {', '.join(renamed_cols2)}
        FROM 
            `{PROJECT_ID}.{DATASET_ID}.{table1}` AS t1
        INNER JOIN 
            `{PROJECT_ID}.{DATASET_ID}.{table2}` AS t2
        ON 
            t1.{column1} = t2.{column2};
        """

        try:
            query_job = client.query(query)
            query_job.result()
            print(f"Vue '{view_name}' créée dans '{DATASET_VIEW_ID}'.")
        except Exception as e:
            print(f"Erreur lors de la création de la vue '{view_name}': {e}")


if __name__ == "__main__":
    # Supprimer le dataset après usage (à utiliser avec précaution)
    delete_dataset("sandbox-jndong", DATASET_VIEW_ID)

    create_dataset_if_not_exists()

    tables_metadata = get_tables_metadata()
    print("Métadonnées des tables :")
    print(tables_metadata)

    similar_columns_df = find_similar_columns(tables_metadata)
    print("\nColonnes similaires identifiées entre les tables :")
    print(similar_columns_df)

    if not similar_columns_df.empty:
        create_views_for_similar_columns(similar_columns_df, tables_metadata)
    else:
        print("Aucune colonne similaire trouvée entre les tables. Impossible de créer des vues.")