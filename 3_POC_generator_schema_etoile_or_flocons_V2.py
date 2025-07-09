# bigquery_schema_analyzer_with_logging.py

import logging
import os
from google.cloud import bigquery
from collections import defaultdict
import re
import datetime

# === Configuration du Logging ===
def setup_logger():
    logger = logging.getLogger("BigQuerySchemaAnalyzer")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Handler console
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


# === Envoi des Logs vers BigQuery ===
def log_to_bigquery(project_id, dataset_id, table_id, message, level="INFO"):
    client = bigquery.Client(project=project_id)
    table_ref = client.dataset(dataset_id).table(table_id)

    rows_to_insert = [
        (
            message,
            level,
            datetime.datetime.now().isoformat(),
            os.getenv("USER", "unknown"),
        )
    ]

    try:
        errors = client.insert_rows(table_ref, rows_to_insert)
        if errors:
            print(f"[Erreur] Impossible d'insérer les logs dans BigQuery : {errors}")
    except Exception as e:
        print(f"[Erreur] Échec de l'insertion des logs : {e}")


# === Classe principale d'analyse ===
class BigQuerySchemaAnalyzer:
    def __init__(self, project_id, source_dataset_id, target_dataset_id, log_dataset_id):
        self.project_id = project_id
        self.source_dataset_id = source_dataset_id
        self.target_dataset_id = target_dataset_id
        self.log_dataset_id = log_dataset_id
        self.client = bigquery.Client(project=project_id)
        self.tables_info = {}
        self.field_to_tables = defaultdict(list)
        self.tables_schema = {}
        self.possible_joins = []
        self.fact_tables = []
        self.dimension_tables = []

        # Setup logger
        self.logger = setup_logger()

    def ensure_log_table_exists(self):
        """Crée le dataset et la table de logs si nécessaire"""
        client = bigquery.Client(project=self.project_id)
        dataset_ref = client.dataset(self.log_dataset_id)

        try:
            client.get_dataset(dataset_ref)
        except:
            self.log(f"Le dataset {self.log_dataset_id} n'existe pas. Création en cours...", level="WARNING")
            dataset = bigquery.Dataset(dataset_ref)
            client.create_dataset(dataset)

        table_ref = dataset_ref.table("analysis_logs")
        try:
            client.get_table(table_ref)
            self.log("La table de logs existe déjà.")
        except:
            self.log("Création de la table de logs analysis_logs...", level="WARNING")
            schema = [
                bigquery.SchemaField("message", "STRING"),
                bigquery.SchemaField("level", "STRING"),
                bigquery.SchemaField("timestamp", "TIMESTAMP"),
                bigquery.SchemaField("user", "STRING")
            ]
            table = bigquery.Table(table_ref, schema=schema)
            client.create_table(table)
            self.log("Table de logs créée avec succès.")

    def log(self, message, level="INFO"):
        """Envoie le log à la fois dans la console ET dans BigQuery"""
        getattr(self.logger, level.lower())(message)
        log_to_bigquery(self.project_id, self.log_dataset_id, "analysis_logs", message, level)

    def list_tables(self):
        """Liste uniquement les tables (pas les vues) dans le dataset source"""
        self.log(f"Recherche des tables dans le dataset '{self.source_dataset_id}'")
        dataset_ref = self.client.dataset(self.source_dataset_id)
        all_tables = list(self.client.list_tables(dataset_ref))
        tables_only = []

        for table in all_tables:
            table_ref = self.client.dataset(self.source_dataset_id).table(table.table_id)
            table_obj = self.client.get_table(table_ref)
            if not table_obj.view_query:  # Vérifie si ce n'est pas une vue
                table_id = table.table_id
                self.log(f"- Table trouvée : {table_id}")
                self.tables_info[table_id] = {'rows': self.get_row_count(table_id)}
                tables_only.append(table_id)
        return tables_only

    def get_row_count(self, table_id):
        query = f"""
        SELECT COUNT(*) as count 
        FROM `{self.project_id}.{self.source_dataset_id}.{table_id}`
        """
        result = self.client.query(query).result()
        count = list(result)[0].count
        return count

    def get_schema(self, table_id):
        table_ref = self.client.dataset(self.source_dataset_id).table(table_id)
        table = self.client.get_table(table_ref)
        return table.schema

    def analyze_schemas(self):
        self.log("Analyse des schémas des tables...")
        for table_id in self.tables_info:
            schema = self.get_schema(table_id)
            fields = []
            for field in schema:
                field_name = field.name.lower()
                field_type = field.field_type
                normalized = self.normalize_field_name(field_name)
                fields.append({
                    'name': field_name,
                    'type': field_type,
                    'normalized_name': normalized
                })
                self.field_to_tables[normalized].append({
                    'table_id': table_id,
                    'field_name': field_name,
                    'field_type': field_type
                })
            self.tables_schema[table_id] = fields
        self.identify_potential_joins()

    def normalize_field_name(self, field_name):
        normalized = re.sub(r'^(id_|pk_|fk_)', '', field_name)
        normalized = re.sub(r'(_id|_key|_pk|_fk)$', '', normalized)
        normalized = re.sub(r'[^a-z0-9]', '', normalized)
        return normalized

    def identify_potential_joins(self):
        for normalized_field, tables in self.field_to_tables.items():
            if len(tables) > 1:
                for i, table1_info in enumerate(tables):
                    for table2_info in tables[i+1:]:
                        if table1_info['field_type'] == table2_info['field_type']:
                            self.possible_joins.append({
                                'table1': table1_info['table_id'],
                                'field1': table1_info['field_name'],
                                'table2': table2_info['table_id'],
                                'field2': table2_info['field_name'],
                                'similarity': 'exact' if table1_info['field_name'] == table2_info['field_name'] else 'normalized',
                                'normalized_field': normalized_field
                            })
        self.log(f"Trouvé {len(self.possible_joins)} jointures potentielles.")

    def identify_fact_dimension_tables(self):
        inbound_refs = defaultdict(int)
        for join in self.possible_joins:
            inbound_refs[join['table2']] += 1
        sorted_refs = sorted(inbound_refs.items(), key=lambda x: x[1], reverse=True)
        for table_id, ref_count in sorted_refs:
            if ref_count >= 2:
                self.dimension_tables.append(table_id)
        for table_id, info in sorted(self.tables_info.items(), key=lambda x: x[1]['rows'], reverse=True):
            if table_id not in self.dimension_tables:
                self.fact_tables.append(table_id)
                if len(self.fact_tables) >= 3:
                    break
        self.log(f"Tables de faits identifiées : {self.fact_tables}")
        self.log(f"Tables de dimensions identifiées : {self.dimension_tables}")

    def generate_star_schema_views(self):
        created_views = []
        for fact_table in self.fact_tables:
            joins = []
            for join in self.possible_joins:
                if join['table1'] == fact_table and join['table2'] in self.dimension_tables:
                    joins.append({
                        'fact_table': fact_table,
                        'dim_table': join['table2'],
                        'fact_field': join['field1'],
                        'dim_field': join['field2']
                    })
            if joins:
                view_name = f"star_{fact_table}"
                sql = self.generate_star_view_sql(fact_table, joins)
                self.create_view(view_name, sql)
                created_views.append(view_name)
        self.log(f"{len(created_views)} vues en schéma étoile créées.")
        return created_views

    def generate_snowflake_schema_views(self):
        created_views = []
        for fact_table in self.fact_tables:
            first_level_dims = []
            for join in self.possible_joins:
                if join['table1'] == fact_table and join['table2'] in self.dimension_tables:
                    first_level_dims.append({
                        'dim_table': join['table2'],
                        'fact_field': join['field1'],
                        'dim_field': join['field2']
                    })
            second_level_joins = []
            for first_dim in first_level_dims:
                for join in self.possible_joins:
                    if join['table1'] == first_dim['dim_table'] and join['table2'] in self.dimension_tables:
                        second_level_joins.append({
                            'parent_dim': first_dim['dim_table'],
                            'child_dim': join['table2'],
                            'parent_field': join['field1'],
                            'child_field': join['field2']
                        })
            if first_level_dims:
                view_name = f"snowflake_{fact_table}"
                sql = self.generate_snowflake_view_sql(fact_table, first_level_dims, second_level_joins)
                self.create_view(view_name, sql)
                created_views.append(view_name)
        self.log(f"{len(created_views)} vues en schéma flocon créées.")
        return created_views

    def generate_star_view_sql(self, fact_table, joins):
        select_clause = ["f.*"]
        join_clauses = []
        for i, join in enumerate(joins):
            dim_alias = f"d{i}"
            dim_table_ref = self.client.dataset(self.source_dataset_id).table(join['dim_table'])
            dim_schema = self.client.get_table(dim_table_ref).schema
            for field in dim_schema:
                if field.name != join['dim_field']:
                    select_clause.append(f"{dim_alias}.{field.name} AS {join['dim_table']}_{field.name}")
            join_clauses.append(f"""
                LEFT JOIN `{self.project_id}.{self.source_dataset_id}.{join['dim_table']}` AS {dim_alias}
                ON f.{join['fact_field']} = {dim_alias}.{join['dim_field']}
            """)
        return f"""
        SELECT
            {', '.join(select_clause)}
        FROM 
            `{self.project_id}.{self.source_dataset_id}.{fact_table}` AS f
            {' '.join(join_clauses)}
        """

    def generate_snowflake_view_sql(self, fact_table, first_level_dims, second_level_joins):
        select_clause = ["f.*"]
        join_clauses = []
        for i, dim in enumerate(first_level_dims):
            dim_alias = f"d{i}"
            dim_table_ref = self.client.dataset(self.source_dataset_id).table(dim['dim_table'])
            dim_schema = self.client.get_table(dim_table_ref).schema
            for field in dim_schema:
                if field.name != dim['dim_field']:
                    select_clause.append(f"{dim_alias}.{field.name} AS {dim['dim_table']}_{field.name}")
            join_clauses.append(f"""
                LEFT JOIN `{self.project_id}.{self.source_dataset_id}.{dim['dim_table']}` AS {dim_alias}
                ON f.{dim['fact_field']} = {dim_alias}.{dim['dim_field']}
            """)
            for j, second_join in enumerate(second_level_joins):
                if second_join['parent_dim'] == dim['dim_table']:
                    second_dim_alias = f"d{i}_{j}"
                    second_dim_table_ref = self.client.dataset(self.source_dataset_id).table(second_join['child_dim'])
                    second_dim_schema = self.client.get_table(second_dim_table_ref).schema
                    for field in second_dim_schema:
                        if field.name != second_join['child_field']:
                            select_clause.append(f"{second_dim_alias}.{field.name} AS {second_join['child_dim']}_{field.name}")
                    join_clauses.append(f"""
                        LEFT JOIN `{self.project_id}.{self.source_dataset_id}.{second_join['child_dim']}` AS {second_dim_alias}
                        ON {dim_alias}.{second_join['parent_field']} = {second_dim_alias}.{second_join['child_field']}
                    """)
        return f"""
        SELECT
            {', '.join(select_clause)}
        FROM 
            `{self.project_id}.{self.source_dataset_id}.{fact_table}` AS f
            {' '.join(join_clauses)}
        """

    def create_view(self, view_name, sql):
        view_id = f"{self.project_id}.{self.target_dataset_id}.{view_name}"
        view = bigquery.Table(view_id)
        view.view_query = sql
        try:
            try:
                self.client.delete_table(view_id)
            except:
                pass
            self.client.create_table(view)
            self.log(f"Vue créée : {view_id}")
            return True
        except Exception as e:
            self.log(f"Échec de création de la vue '{view_name}' : {e}", level="ERROR")
            return False

    def analyze_and_create_views(self):
        self.ensure_log_table_exists()
        self.log(f"Début de l'analyse du dataset '{self.source_dataset_id}'")
        self.list_tables()
        self.analyze_schemas()
        self.identify_fact_dimension_tables()
        star_views = self.generate_star_schema_views()
        snowflake_views = self.generate_snowflake_schema_views()
        self.create_custom_join_views()
        return {
            "star_views": star_views,
            "snowflake_views": snowflake_views
        }

    def create_custom_join_views(self):
        table_pairs = defaultdict(list)
        for join in self.possible_joins:
            pair_key = tuple(sorted([join['table1'], join['table2']]))
            table_pairs[pair_key].append(join)
        for pair, joins in table_pairs.items():
            table1, table2 = pair
            if (table1 in self.fact_tables and table2 in self.dimension_tables) or \
               (table2 in self.fact_tables and table1 in self.dimension_tables):
                continue
            if len(joins) > 0:
                join = joins[0]
                view_name = f"join_{table1}_{table2}"
                left_table, right_table = join['table1'], join['table2']
                left_field, right_field = join['field1'], join['field2']
                sql = f"""
                SELECT t1.*, t2.* 
                FROM `{self.project_id}.{self.source_dataset_id}.{left_table}` AS t1
                LEFT JOIN `{self.project_id}.{self.source_dataset_id}.{right_table}` AS t2
                ON t1.{left_field} = t2.{right_field}
                """
                self.create_view(view_name, sql)


# === Point d'entrée principal ===
if __name__ == "__main__":
    # Configuration GCP
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/jndong/AI4value_good/Load_file_in_bigquery/dbt_sa_bigquery.json"
    os.environ["USER"] = "automated_pipeline"

    # Paramètres projet
    PROJECT_ID = "sandbox-jndong"
    SOURCE_DATASET_ID = "raw_data"
    TARGET_DATASET_ID = "DataOriginal_matching"
    LOG_DATASET_ID = "Dataset_logs"

    # Exécution
    analyzer = BigQuerySchemaAnalyzer(
        project_id=PROJECT_ID,
        source_dataset_id=SOURCE_DATASET_ID,
        target_dataset_id=TARGET_DATASET_ID,
        log_dataset_id=LOG_DATASET_ID
    )

    results = analyzer.analyze_and_create_views()

    print("\n✅ Résumé final")
    print(f"Vues en schéma étoile : {len(results['star_views'])}")
    print(f"Vues en schéma flocon : {len(results['snowflake_views'])}")