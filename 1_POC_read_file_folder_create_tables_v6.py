import os
import re
import logging
from datetime import datetime, timezone
import chardet
import pandas as pd
import unidecode
from google.cloud import bigquery

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("load_to_bigquery.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Configuration globale
CREDENTIAL = "Load_file_in_bigquery/dbt_sa_bigquery.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIAL
DATASET_NAME = "fufu_republic"  # Dataset pour les données principales
DATASET_LOG = "Dataset_logs"   # Dataset pour les logs
LOG_TABLE_NAME = "logs_inserted_data"  # Nom de la table de log
client = bigquery.Client()


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les noms de colonnes pour être compatibles avec BigQuery."""
    cleaned_columns = [
        re.sub(r"[^\w]+", "_", unidecode.unidecode(col)).strip("_").lower()
        for col in df.columns
    ]
    df.columns = cleaned_columns
    return df


def delete_dataset_if_exists(project_id: str, dataset_id: str):
    """Supprime un dataset s'il existe."""
    dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
    try:
        client.delete_dataset(dataset_ref, delete_contents=True, not_found_ok=True)
        logger.info(f"✅ Dataset supprimé : {project_id}.{dataset_id}")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la suppression du dataset : {e}")


def create_dataset():
    """Crée le dataset principal s'il n'existe pas."""
    dataset_id = f"{client.project}.{DATASET_NAME}"
    try:
        client.get_dataset(dataset_id)
        logger.info(f"Dataset déjà existant : {dataset_id}")
    except Exception:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"
        client.create_dataset(dataset)
        logger.info(f"Dataset créé : {dataset_id}")


def create_log_table():
    """Crée la table de log dans le dataset LogsDataset si elle n'existe pas."""
    table_id = f"{client.project}.{DATASET_LOG}.{LOG_TABLE_NAME}"
    schema = [
        bigquery.SchemaField("file_name", "STRING"),
        bigquery.SchemaField("file_path", "STRING"),
        bigquery.SchemaField("file_type", "STRING"),
        bigquery.SchemaField("file_size_mb", "FLOAT"),
        bigquery.SchemaField("rows_inserted", "INTEGER"),
        bigquery.SchemaField("operation_status", "STRING"),  # Succès ou échec
        bigquery.SchemaField("error_message", "STRING"),     # Message d'erreur en cas d'échec
        bigquery.SchemaField("start_time", "TIMESTAMP"),     # Heure de début de l'opération
        bigquery.SchemaField("end_time", "TIMESTAMP"),       # Heure de fin de l'opération
        bigquery.SchemaField("user_or_service", "STRING")    # Utilisateur ou service exécutant le script
    ]
    try:
        # Vérifie si la table existe déjà
        client.get_table(table_id)
        logger.info(f"Table de log déjà existante : {table_id}")
    except Exception as e:
        # Si la table n'existe pas, la créer
        if "Not found" in str(e):
            table = bigquery.Table(table_id, schema=schema)
            client.create_table(table)
            logger.info(f"Table de log créée : {table_id}")
        else:
            logger.error(f"Erreur lors de la création de la table de log : {e}")


def safe_read_file(file_path: str) -> pd.DataFrame:
    """Lit un fichier CSV, JSON, Parquet ou Avro en gérant les erreurs."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".csv":
            with open(file_path, 'rb') as f:
                encoding = chardet.detect(f.read(10000))['encoding']
            df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='skip', encoding=encoding)
        elif ext == ".json":
            df = pd.read_json(file_path)
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
        elif ext == ".avro":
            import fastavro
            with open(file_path, 'rb') as f:
                df = pd.DataFrame(list(fastavro.reader(f)))
        else:
            raise ValueError(f"Format non supporté : {ext}")
        return clean_column_names(df)
    except Exception as e:
        logger.error(f"Erreur lecture fichier {file_path}: {e}")
        raise


def inventory_data(folder_path: str) -> pd.DataFrame:
    """Inventorie les fichiers dans un dossier."""
    infos = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.startswith('.'):
                continue
            path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            size = os.path.getsize(path) / (1024 * 1024)
            mod_time = datetime.fromtimestamp(os.path.getmtime(path))
            infos.append({
                "file_name": file,
                "file_path": path,
                "file_type": ext,
                "file_size_mb": round(size, 2),
                "last_modified": mod_time
            })
    return pd.DataFrame(infos)


def load_file_to_bigquery(file_path: str) -> int:
    """Charge un fichier dans BigQuery et retourne le nombre de lignes insérées."""
    ext = os.path.splitext(file_path)[1].lower()
    name = os.path.splitext(os.path.basename(file_path))[0]
    table_id = f"{client.project}.{DATASET_NAME}.{name}"

    try:
        df = safe_read_file(file_path)

        job_config = None
        if ext == ".csv":
            job_config = bigquery.LoadJobConfig(autodetect=True, skip_leading_rows=1, source_format=bigquery.SourceFormat.CSV)
        elif ext in (".json", ".parquet", ".avro"):
            source_format = {
                ".json": bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                ".parquet": bigquery.SourceFormat.PARQUET,
                ".avro": bigquery.SourceFormat.AVRO
            }[ext]
            job_config = bigquery.LoadJobConfig(source_format=source_format)

        if job_config:
            with open(file_path, "rb") as source_file:
                client.load_table_from_file(source_file, table_id, job_config=job_config).result()
            logger.info(f"✅ Fichier chargé : {file_path}")
            return len(df)
        else:
            logger.warning(f"Format non pris en charge : {file_path}")
            return 0
    except Exception as e:
        logger.error(f"Erreur chargement fichier {file_path}: {e}")
        return 0


def log_insert(file_name, file_path, file_type, size, rows, status, error_message=None):
    """Insère une entrée dans la table de log."""
    table_id = f"{client.project}.{DATASET_LOG}.{LOG_TABLE_NAME}"
    start_time = datetime.now(timezone.utc)
    end_time = datetime.now(timezone.utc)
    user_or_service = os.getenv("USER", "Unknown User")  # Récupère l'utilisateur ou le service

    data = [{
        "file_name": file_name,
        "file_path": file_path,
        "file_type": file_type,
        "file_size_mb": size,
        "rows_inserted": rows,
        "operation_status": status,
        "error_message": error_message,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "user_or_service": user_or_service
    }]
    try:
        # Vérifie que la table existe avant d'insérer des données
        client.get_table(table_id)
        client.insert_rows_json(table_id, data)
        logger.info(f"✅ Log inséré pour {file_name}")
    except Exception as e:
        if "Not found" in str(e):
            logger.error(f"❌ Table de log introuvable : {table_id}. Créez la table avant d'insérer des données.")
        else:
            logger.error(f"❌ Erreur insertion log pour {file_name}: {e}")


if __name__ == "__main__":
    folder = "/Users/jndong/CDE-dbt-assignment/Datasets"

    # Étape 1 : Supprimer le dataset principal s'il existe
    delete_dataset_if_exists(client.project, DATASET_NAME)

    # Étape 2 : Créer le dataset principal
    create_dataset()

    # Étape 3 : Créer la table de log dans le dataset LogsDataset
    create_log_table()

    # Étape 4 : Inventorier les fichiers
    inventory = inventory_data(folder)
    logger.info(f"🗂️ Inventaire trouvé : {len(inventory)} fichiers")

    # Étape 5 : Charger les fichiers dans BigQuery et remplir la table de log
    for _, row in inventory.iterrows():
        path, name, ftype, size = row['file_path'], row['file_name'], row['file_type'], row['file_size_mb']
        try:
            rows = load_file_to_bigquery(path)
            log_insert(name, path, ftype, size, rows, status="SUCCESS")
        except Exception as e:
            log_insert(name, path, ftype, size, 0, status="FAILURE", error_message=str(e))