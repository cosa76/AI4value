#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour générer automatiquement des vues Machine Learning dans BigQuery
à partir de tables existantes dans un dataset, avec paramètres définis en dur
"""

import os
import sys
import logging
import re

from google.cloud import bigquery

# Configuration de l'authentification
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Load_file_in_bigquery/dbt_sa_bigquery.json"

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class BigQueryMLViewsGenerator:
    """Classe pour générer des vues ML dans BigQuery"""

    def __init__(self, project_id, source_dataset_id, ml_dataset_id):
        self.project_id = project_id
        self.source_dataset_id = source_dataset_id
        self.ml_dataset_id = ml_dataset_id
        self.client = bigquery.Client(project=project_id)

        self.numeric_types = ['INTEGER', 'INT64', 'FLOAT', 'FLOAT64', 'NUMERIC', 'BIGNUMERIC']
        self.categorical_types = ['STRING', 'BOOL', 'BOOLEAN']
        self.temporal_types = ['DATE', 'DATETIME', 'TIME', 'TIMESTAMP']

    def create_ml_dataset_if_not_exists(self):
        dataset_ref = self.client.dataset(self.ml_dataset_id)
        try:
            self.client.get_dataset(dataset_ref)
            logger.info(f"Le dataset {self.ml_dataset_id} existe déjà")
        except Exception:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            self.client.create_dataset(dataset)
            logger.info(f"Dataset {self.ml_dataset_id} créé avec succès")

    def get_tables_list(self):
        tables = list(self.client.list_tables(f"{self.project_id}.{self.source_dataset_id}"))
        return [table.table_id for table in tables]

    def get_table_schema(self, table_id):
        table_ref = self.client.dataset(self.source_dataset_id).table(table_id)
        table = self.client.get_table(table_ref)
        return table.schema

    def analyze_column_values(self, table_id, column_name, data_type):
        query = f"""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT {column_name}) as unique_values,
            AVG(CASE WHEN {column_name} IS NULL THEN 1 ELSE 0 END) as null_ratio
        FROM 
            `{self.project_id}.{self.source_dataset_id}.{table_id}`
        """
        try:
            results = self.client.query(query).result().to_dataframe()
            if results.empty:
                return {"ml_type": "unknown", "metrics": {}}

            total_rows = results["total_rows"].iloc[0]
            unique_values = results["unique_values"].iloc[0]
            null_ratio = results["null_ratio"].iloc[0]
            unique_ratio = unique_values / total_rows if total_rows > 0 else 0

            ml_type = "unknown"

            if data_type in self.numeric_types:
                ml_type = "numerical" if unique_ratio >= 0.05 else "categorical"
                stats_query = f"""
                SELECT 
                    MIN({column_name}) as min_value,
                    MAX({column_name}) as max_value,
                    AVG({column_name}) as avg_value,
                    STDDEV({column_name}) as stddev_value
                FROM 
                    `{self.project_id}.{self.source_dataset_id}.{table_id}`
                WHERE
                    {column_name} IS NOT NULL
                """
                stats = self.client.query(stats_query).result().to_dataframe()
                if not stats.empty:
                    min_value = stats["min_value"].iloc[0]
                    max_value = stats["max_value"].iloc[0]
                    if min_value == 0 and max_value == 1:
                        ml_type = "binary_target"
                    return {
                        "ml_type": ml_type,
                        "metrics": {
                            "total_rows": total_rows,
                            "unique_values": unique_values,
                            "unique_ratio": unique_ratio,
                            "null_ratio": null_ratio,
                            "avg_value": stats["avg_value"].iloc[0],
                            "stddev_value": stats["stddev_value"].iloc[0]
                        }
                    }

            elif data_type in self.categorical_types:
                ml_type = "categorical" if unique_ratio < 0.5 else "text"
                col_lower = column_name.lower()
                if unique_ratio > 0.9 or re.search(r'id$|^id|_id$|^key$|_key$', col_lower):
                    ml_type = "id_or_key"
                elif re.search(r'name|label|category|type|status|state|country|region|city', col_lower):
                    ml_type = "categorical"

            elif data_type in self.temporal_types:
                ml_type = "timestamp" if re.search(r'date|time|timestamp|created|updated|ts$|_ts|_at$', column_name.lower()) else "temporal"

            return {
                "ml_type": ml_type,
                "metrics": {
                    "total_rows": total_rows,
                    "unique_values": unique_values,
                    "unique_ratio": unique_ratio,
                    "null_ratio": null_ratio
                }
            }

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de la colonne {column_name}: {str(e)}")
            return {"ml_type": "unknown", "metrics": {}}

    def generate_ml_view_for_table(self, table_id):
        schema = self.get_table_schema(table_id)
        columns_info = []

        for field in schema:
            analysis = self.analyze_column_values(table_id, field.name, field.field_type)
            columns_info.append({
                "name": field.name,
                "data_type": field.field_type,
                "ml_type": analysis["ml_type"],
                "metrics": analysis["metrics"]
            })

        select_parts = []
        potential_targets = [col for col in columns_info if col["ml_type"] == "binary_target"]
        categorical_features = [col for col in columns_info if col["ml_type"] == "categorical"]
        numerical_features = [col for col in columns_info if col["ml_type"] == "numerical"]
        temporal_features = [col for col in columns_info if col["ml_type"] in ["temporal", "timestamp"]]
        id_columns = [col for col in columns_info if col["ml_type"] == "id_or_key"]

        for col in columns_info:
            name = col["name"]
            ml_type = col["ml_type"]
            metrics = col.get("metrics", {})

            if ml_type == "id_or_key":
                select_parts.append(f"{name} as {name}_id")
            elif ml_type == "numerical":
                if metrics.get("stddev_value", 0) > 0:
                    select_parts.append(f"({name} - {metrics['avg_value']}) / {metrics['stddev_value']} as {name}_normalized")
                else:
                    select_parts.append(name)
            elif ml_type == "categorical":
                select_parts.append(name)
                if metrics.get("unique_values", 0) <= 10:
                    select_parts.append(f"-- Suggestion: Utiliser l'encodage one-hot pour la colonne {name}")
            elif ml_type == "timestamp":
                select_parts.extend([
                    f"EXTRACT(YEAR FROM {name}) AS {name}_year",
                    f"EXTRACT(MONTH FROM {name}) AS {name}_month",
                    f"EXTRACT(DAY FROM {name}) AS {name}_day",
                    f"EXTRACT(DAYOFWEEK FROM {name}) AS {name}_dayofweek"
                ])
            elif ml_type == "text":
                select_parts.append(name)
                select_parts.append(f"-- Suggestion: Considérer des techniques NLP pour la colonne {name}")
            else:
                select_parts.append(name)

        view_sql = f"""
        -- Vue ML générée automatiquement pour la table {table_id}
        -- Potentielles cibles: {', '.join([col["name"] for col in potential_targets]) if potential_targets else 'Aucune détectée'}
        SELECT
          {",\n  ".join(select_parts)}
        FROM
          `{self.project_id}.{self.source_dataset_id}.{table_id}`
        """

        view_name = f"{table_id}_ml"
        view_ref = self.client.dataset(self.ml_dataset_id).table(view_name)
        view = bigquery.Table(view_ref)
        view.view_query = view_sql

        try:
            self.client.delete_table(view_ref, not_found_ok=True)
            self.client.create_table(view)
            logger.info(f"Vue ML '{view_name}' créée avec succès")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la création de la vue ML '{view_name}': {str(e)}")
            return False

    def run(self):
        try:
            self.create_ml_dataset_if_not_exists()
            tables = self.get_tables_list()

            if not tables:
                logger.warning(f"Aucune table trouvée dans le dataset {self.source_dataset_id}")
                return

            logger.info(f"Tables trouvées: {tables}")
            success_count = 0

            for table_id in tables:
                logger.info(f"Génération de la vue ML pour la table {table_id}")
                if self.generate_ml_view_for_table(table_id):
                    success_count += 1

            logger.info(f"Génération terminée. {success_count}/{len(tables)} vues ML créées avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution: {str(e)}")


def main():
    """Point d'entrée principal avec paramètres définis en dur"""
    # ✅ Paramètres en dur à modifier si besoin
    project_id = "sandbox-jndong"
    source_dataset_id = "DataOriginal"
    ml_dataset_id = "DataOriginal_ml"

    print(f"\nCréation de vues ML pour '{project_id}.{source_dataset_id}' vers '{ml_dataset_id}'...")

    generator = BigQueryMLViewsGenerator(
        project_id=project_id,
        source_dataset_id=source_dataset_id,
        ml_dataset_id=ml_dataset_id
    )
    generator.run()


if __name__ == "__main__":
    main()