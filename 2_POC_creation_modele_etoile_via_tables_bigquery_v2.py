import os
from google.cloud import bigquery
import pandas as pd


# Définition de la clé d'authentification Google Cloud
CREDENTIAL = "/Users/jndong/AI4value_good/Load_file_in_bigquery/dbt_sa_bigquery.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIAL

# Initialisation du client BigQuery
client = bigquery.Client()

# Identifiants du projet et des datasets
PROJECT_ID = "sandbox-jndong"
DATASET_ID = "raw_data"
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
        columns = [field.name for field in table_info.schema]
        tables_metadata.append({
            "table_name": table_info.table_id,
            "columns": columns
        })
    
    return pd.DataFrame(tables_metadata)

def find_identical_columns(tables_metadata):
    identical_columns = []

    for i in range(len(tables_metadata)):
        for j in range(i + 1, len(tables_metadata)):
            table1 = tables_metadata.iloc[i]["table_name"]
            table2 = tables_metadata.iloc[j]["table_name"]
            cols1 = tables_metadata.iloc[i]["columns"]
            cols2 = tables_metadata.iloc[j]["columns"]
            common_columns = set(cols1).intersection(cols2)

            if common_columns:
                for col in common_columns:
                    identical_columns.append({
                        "table1": table1,
                        "table2": table2,
                        "identical_column": col
                    })
    
    return pd.DataFrame(identical_columns)

def create_views_for_identical_columns(identical_columns_df, tables_metadata):
    for _, row in identical_columns_df.iterrows():
        table1 = row["table1"]
        table2 = row["table2"]
        identical_column = row["identical_column"]

        view_name = f"view_{table1}_to_{table2}_on_{identical_column}"
        view_id = f"{PROJECT_ID}.{DATASET_VIEW_ID}.{view_name}"  # <-- Utilisation de DATASET_VIEW_ID ici

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
            t1.{identical_column} = t2.{identical_column};
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

    identical_columns_df = find_identical_columns(tables_metadata)
    print("\nColonnes identiques identifiées entre les tables :")
    print(identical_columns_df)

    if not identical_columns_df.empty:
        create_views_for_identical_columns(identical_columns_df, tables_metadata)
    else:
        print("Aucune colonne identique trouvée entre les tables. Impossible de créer des vues.")
