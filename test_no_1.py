import os
import re
from datetime import datetime, timezone
import unidecode
import pandas as pd
import chardet

from google.cloud import bigquery

# Configuration
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./dbt_sa_bigquery.json"
DATASET_NAME = "DataOriginal"
client = bigquery.Client()

def clean_column_names(df):
    """Nettoie les noms des colonnes en supprimant les accents, caractères spéciaux, etc."""
    cleaned_columns = []
    for col in df.columns:
        col = unidecode.unidecode(col)  # Supprimer accents
        col = re.sub(r"[^\w]+", "_", col)  # Remplace tout sauf lettres/chiffres par _
        col = col.strip("_")  # Enlève les _ au début/fin
        col = col.lower()  # Minuscule
        cleaned_columns.append(col)
    df.columns = cleaned_columns
    return df

def detect_csv_separator(file_path):
    """Détecte le séparateur utilisé dans un fichier CSV."""
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        separators = [',', ';', '\t', '|', ':']
        counts = {sep: first_line.count(sep) for sep in separators}
        return max(counts, key=counts.get)

def safe_read_text_file(file_path, read_type="csv"):
    """Lit un fichier CSV ou JSON en tenant compte des séparateurs et de l'en-tête."""
    with open(file_path, 'rb') as f:
        rawdata = f.read(10000)
        detected_encoding = chardet.detect(rawdata)['encoding']
    
    try:
        if read_type == "csv":
            separator = detect_csv_separator(file_path)
            df = pd.read_csv(file_path, encoding=detected_encoding, sep=separator, header='infer')
            # Si l'en-tête n'est pas détecté correctement, définir manuellement les colonnes
            if df.columns.str.contains("Unnamed").any():
                df = pd.read_csv(file_path, encoding=detected_encoding, sep=separator, header=None)
        elif read_type == "json":
            df = pd.read_json(file_path, encoding=detected_encoding)
        
        df = clean_column_names(df)
        return df
    except Exception as e:
        raise ValueError(f"Erreur lecture {file_path}: {e}")

def inventory_data(folder_path):
    """Liste tous les fichiers dans un dossier et collecte leurs informations."""
    files_info = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.startswith('.'):
                continue
            path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            size = os.path.getsize(path) / (1024 * 1024)
            mod_time = datetime.fromtimestamp(os.path.getmtime(path))
            files_info.append({
                "file_name": file,
                "file_path": path,
                "file_type": ext,
                "file_size_mb": round(size, 2),
                "last_modified": mod_time
            })
    return pd.DataFrame(files_info)

def evaluate_data_quality(file_path):
    """Évalue la qualité des données d'un fichier."""
    ext = os.path.splitext(file_path)[1].lower()
    report = {"file_path": file_path, "is_corrupted": False, "missing_values": None, "data_consistency": None}
    try:
        if ext == ".csv":
            df = safe_read_text_file(file_path, "csv")
        elif ext == ".json":
            df = safe_read_text_file(file_path, "json")
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
            df = clean_column_names(df)
        elif ext == ".avro":
            import fastavro
            with open(file_path, 'rb') as f:
                df = pd.DataFrame(list(fastavro.reader(f)))
                df = clean_column_names(df)
        else:
            return report
        report["missing_values"] = df.isnull().sum().sum()
        report["data_consistency"] = df.dtypes.apply(lambda x: str(x)).to_dict()
    except Exception as e:
        report["is_corrupted"] = True
        print(f"Erreur lecture {file_path}: {e}")
    return report

def delete_dataset(project_id: str, dataset_id: str, delete_contents: bool = True):
    """Supprime un dataset BigQuery."""
    dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
    try:
        client.delete_dataset(dataset_ref, delete_contents=delete_contents, not_found_ok=False)
        print(f"✅ Dataset supprimé : {project_id}.{dataset_id}")
    except Exception as e:
        print(f"❌ Erreur lors de la suppression du dataset : {e}")

def create_dataset_if_not_exists():
    """Crée un dataset BigQuery s'il n'existe pas déjà."""
    dataset_id = f"{client.project}.{DATASET_NAME}"
    try:
        client.get_dataset(dataset_id)
        print(f"Dataset {dataset_id} already exists.")
    except Exception:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"
        client.create_dataset(dataset)
        print(f"Dataset {dataset_id} created.")

def load_file_to_bigquery(file_path):
    """Charge un fichier dans BigQuery."""
    ext = os.path.splitext(file_path)[1].lower()
    name = os.path.splitext(os.path.basename(file_path))[0]
    table_id = f"{client.project}.{DATASET_NAME}.{name}"
    try:
        if ext == ".csv":
            df = safe_read_text_file(file_path, "csv")
            fmt = bigquery.SourceFormat.CSV
            job_config = bigquery.LoadJobConfig(autodetect=True, skip_leading_rows=1, source_format=fmt)
        elif ext == ".json":
            df = safe_read_text_file(file_path, "json")
            fmt = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
            job_config = bigquery.LoadJobConfig(autodetect=True, source_format=fmt)
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
            df = clean_column_names(df)
            job_config = bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.PARQUET)
        elif ext == ".avro":
            import fastavro
            with open(file_path, 'rb') as f:
                df = pd.DataFrame(list(fastavro.reader(f)))
                df = clean_column_names(df)
            job_config = bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.AVRO)
        else:
            return 0
        with open(file_path, "rb") as source_file:
            client.load_table_from_file(source_file, table_id, job_config=job_config).result()
        return len(df)
    except Exception as e:
        print(f"Erreur chargement {file_path}: {e}")
        return 0

def create_log_table():
    """Crée une table de logs dans BigQuery."""
    table_id = f"{client.project}.{DATASET_NAME}.log_inserted_data"
    schema = [
        bigquery.SchemaField("file_name", "STRING"),
        bigquery.SchemaField("file_path", "STRING"),
        bigquery.SchemaField("file_type", "STRING"),
        bigquery.SchemaField("file_size_mb", "FLOAT"),
        bigquery.SchemaField("rows_inserted", "INTEGER"),
        bigquery.SchemaField("insertion_time", "TIMESTAMP")
    ]
    try:
        client.get_table(table_id)
        print(f"Table {table_id} already exists.")
    except Exception:
        table = bigquery.Table(table_id, schema=schema)
        client.create_table(table)
        print(f"Table {table_id} created.")

def log_insert(file_name, file_path, file_type, size, rows):
    """Insère un log dans la table de logs."""
    table_id = f"{client.project}.{DATASET_NAME}.log_inserted_data"
    insertion_time = datetime.now(timezone.utc).isoformat()
    data = [{
        "file_name": file_name,
        "file_path": file_path,
        "file_type": file_type,
        "file_size_mb": size,
        "rows_inserted": rows,
        "insertion_time": insertion_time
    }]
    try:
        client.insert_rows_json(table_id, data)
        print(f"Log inserted successfully for {file_name}")
    except Exception as e:
        print(f"Error inserting log for {file_name}: {e}")

# Script principal
if __name__ == "__main__":
    folder = "/Users/jndong/files_data_type2"
    inventory = inventory_data(folder)
    print(inventory)

    quality_reports = [evaluate_data_quality(f) for f in inventory['file_path']]
    print(pd.DataFrame(quality_reports))

    # Supprimer le dataset après usage (à utiliser avec précaution)
    delete_dataset("sandbox-jndong", DATASET_NAME)

    create_dataset_if_not_exists()
    create_log_table()

    for _, row in inventory.iterrows():
        path, name, ftype, size = row['file_path'], row['file_name'], row['file_type'], row['file_size_mb']
        rows = load_file_to_bigquery(path)
        log_insert(name, path, ftype, size, rows)