import os
import logging
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

# Configuration du logging
logging.basicConfig(level=logging.INFO)

# Paramètres en dur
PROJECT_ID = "sandbox-jndong"
DATASET_ID = "DataOriginal"
DATASET_ML_ID = "DataOriginal_ml"  # Dataset cible pour les vues ML
SERVICE_ACCOUNT_KEY_JSON = "Load_file_in_bigquery/dbt_sa_bigquery.json"
MAX_NULL_PERCENTAGE = 0.5  # Seuil pour ignorer les colonnes avec trop de NaN
VIEW_PREFIX = "ml_"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY_JSON

class BQMLViewGenerator:
    def __init__(self, project_id, dataset_id, dataset_ml_id):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.dataset_ml_id = dataset_ml_id
        self.client = bigquery.Client(project=project_id)

    def list_tables(self):
        """Liste toutes les tables dans le dataset source"""
        dataset_ref = self.client.dataset(self.dataset_id)
        try:
            tables = list(self.client.list_tables(dataset_ref))
            return [table.table_id for table in tables]
        except NotFound as e:
            logging.error(f"Dataset {self.dataset_id} introuvable: {e}")
            return []

    def get_table_schema(self, table_id):
        """Récupère le schéma d'une table"""
        table_ref = self.client.dataset(self.dataset_id).table(table_id)
        try:
            table = self.client.get_table(table_ref)
            return table.schema
        except NotFound as e:
            logging.error(f"Table {table_id} introuvable: {e}")
            return []

    def analyze_columns(self, schema, sample_table_id):
        """Analyse les colonnes pour détecter celles utiles au ML"""
        query = f"SELECT * FROM `{self.project_id}.{self.dataset_id}.{sample_table_id}` LIMIT 1000"
        df = self.client.query(query).to_dataframe()

        ml_columns = []
        for field in schema:
            col_name = field.name
            col_type = field.field_type.lower()

            if col_type not in ["integer", "float", "string"]:
                continue  # Ignorer types complexes

            total_rows = len(df)
            if total_rows == 0:
                continue

            null_count = df[col_name].isnull().sum()
            null_ratio = null_count / total_rows

            if null_ratio > MAX_NULL_PERCENTAGE:
                logging.info(f"Ignorant '{col_name}' - Trop de valeurs manquantes ({null_ratio:.2%})")
                continue

            if any(kw in col_name.lower() for kw in ["id", "uuid", "guid"]):
                logging.info(f"Ignorant '{col_name}' - Probablement une clé primaire")
                continue

            ml_columns.append({
                'name': col_name,
                'type': col_type,
                'nullable': null_ratio > 0
            })

        return ml_columns

    def generate_sql_view(self, table_id, ml_columns):
        """Génère la requête SQL pour créer une vue ML-friendly dans le dataset ML"""
        selected_cols = []

        for col in ml_columns:
            col_name = col['name']
            col_type = col['type']

            if col_type == "string":
                selected_cols.append(f"CAST({col_name} AS STRING) AS {col_name}")
            elif col_type in ["integer", "float"]:
                if col['nullable']:
                    selected_cols.append(f"IFNULL({col_name}, 0) AS {col_name}")
                else:
                    selected_cols.append(col_name)

        cols_sql = ",\n    ".join(selected_cols)
        view_full_id = f"{self.project_id}.{self.dataset_ml_id}.{VIEW_PREFIX}{table_id}"
        source_table = f"{self.project_id}.{self.dataset_id}.{table_id}"

        sql = f"""
CREATE OR REPLACE VIEW `{view_full_id}` AS
SELECT
    {cols_sql}
FROM
    `{source_table}`
WHERE
    TRUE
        """
        return sql

    def create_ml_views(self):
        """Crée une vue ML par table dans le dataset ML"""
        tables = self.list_tables()
        if not tables:
            logging.warning("Aucune table trouvée dans le dataset source.")
            return

        for table_id in tables:
            logging.info(f"Traitement de la table : {table_id}")
            schema = self.get_table_schema(table_id)
            if not schema:
                continue

            ml_columns = self.analyze_columns(schema, table_id)
            if not ml_columns:
                logging.warning(f"Aucune colonne ML trouvée pour {table_id}")
                continue

            view_sql = self.generate_sql_view(table_id, ml_columns)
            try:
                job = self.client.query(view_sql)
                job.result()
                logging.info(f"Vue créée dans {self.dataset_ml_id} : {VIEW_PREFIX}{table_id}")
            except Exception as e:
                logging.error(f"Échec de création de la vue pour {table_id}: {e}")


if __name__ == "__main__":
    generator = BQMLViewGenerator(PROJECT_ID, DATASET_ID, DATASET_ML_ID)
    generator.create_ml_views()