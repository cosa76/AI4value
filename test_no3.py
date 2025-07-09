from google.cloud import bigquery
import pandas as pd
import re
import os
from collections import defaultdict
from fuzzywuzzy import fuzz

# Configuration Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./dbt_sa_bigquery.json"

class BigQuerySchemaAnalyzer:
    def __init__(self, project_id, source_dataset_id, target_dataset_id):
        self.project_id = project_id
        self.source_dataset_id = source_dataset_id
        self.target_dataset_id = target_dataset_id
        self.client = bigquery.Client(project=project_id)
        self.tables_info = {}
        self.field_to_tables = defaultdict(list)
        self.tables_schema = {}
        self.possible_joins = []
        self.fact_tables = []
        self.dimension_tables = []
        self.table_pk_candidates = {}
        self.table_fk_candidates = {}

    def list_tables(self):
        dataset_ref = self.client.dataset(self.source_dataset_id)
        all_tables = list(self.client.list_tables(dataset_ref))
        print(f"Tables trouvées dans {self.source_dataset_id}:")
        tables_only = []
        for table in all_tables:
            table_ref = self.client.dataset(self.source_dataset_id).table(table.table_id)
            table_obj = self.client.get_table(table_ref)
            if not table_obj.view_query:
                table_id = table.table_id
                print(f"- {table_id}")
                self.tables_info[table_id] = {'rows': self.get_row_count(table_id)}
                tables_only.append(table_id)
        return tables_only

    def get_row_count(self, table_id):
        query = f"""
        SELECT COUNT(*) AS count 
        FROM `{self.project_id}.{self.source_dataset_id}.{table_id}`
        """
        result = self.client.query(query).result()
        return list(result)[0].count

    def get_schema(self, table_id):
        table_ref = self.client.dataset(self.source_dataset_id).table(table_id)
        table = self.client.get_table(table_ref)
        return table.schema

    def analyze_schemas(self):
        measure_pattern = re.compile(r'(sum|avg|count|total|amount|quantity|price)', re.IGNORECASE)

        for table_id in self.tables_info:
            schema = self.get_schema(table_id)
            fields = []
            pk_candidates = []
            fk_candidates = []

            for field in schema:
                field_name = field.name
                field_type = field.field_type
                fields.append({'name': field_name, 'type': field_type})

                # Évaluation de la cardinalité
                card = self.get_cardinality(table_id, field_name)
                row_count = self.tables_info[table_id]['rows']

                if card == row_count:
                    pk_candidates.append(field_name)
                elif 1 < card < row_count:
                    fk_candidates.append(field_name)

                # Enregistrer le champ pour fuzzy matching
                normalized = self.normalize_field_name(field_name)
                self.field_to_tables[normalized].append({
                    'table_id': table_id,
                    'field_name': field_name,
                    'field_type': field_type
                })

            self.tables_schema[table_id] = fields
            self.table_pk_candidates[table_id] = pk_candidates
            self.table_fk_candidates[table_id] = fk_candidates

        # Identifier les jointures possibles par fuzzy match
        self.identify_potential_joins_with_fuzzymatch()

        # Afficher les candidats clés
        self.print_key_candidates()

    def normalize_field_name(self, name):
        return re.sub(r'[^a-z0-9]', '', name.lower())

    def identify_potential_joins_with_fuzzymatch(self):
        """Utilise fuzzy matching pour identifier des champs potentiellement liés"""
        field_names_by_table = defaultdict(dict)

        # Collecter tous les champs par table
        for table_id in self.tables_info:
            schema = self.tables_schema[table_id]
            field_names_by_table[table_id] = {self.normalize_field_name(f['name']): f for f in schema}

        # Comparer chaque paire de tables
        tables = list(self.tables_info.keys())
        for i, t1 in enumerate(tables):
            for t2 in tables[i+1:]:
                self.find_similar_fields(t1, t2, field_names_by_table)

    def find_similar_fields(self, table1, table2, field_map):
        """Trouve des champs similaires entre deux tables"""
        for norm1, field1 in field_map[table1].items():
            for norm2, field2 in field_map[table2].items():
                if field1['type'] == field2['type']:
                    similarity = fuzz.token_sort_ratio(norm1, norm2)
                    if similarity > 70:  # Seuil ajustable
                        self.possible_joins.append({
                            'table1': table1,
                            'field1': field1['name'],
                            'table2': table2,
                            'field2': field2['name'],
                            'similarity_score': similarity
                        })

    def identify_fact_dimension_tables(self):
        inbound_refs = defaultdict(int)
        measure_pattern = re.compile(r'(sum|avg|count|total|amount|quantity|price)', re.IGNORECASE)

        for join in self.possible_joins:
            inbound_refs[join['table2']] += 1

        sorted_refs = sorted(inbound_refs.items(), key=lambda x: x[1], reverse=True)

        # Prioriser les tables souvent référencées comme dimensions
        for table_id, ref_count in sorted_refs:
            if ref_count >= 2:
                self.dimension_tables.append(table_id)

        # Ajouter les tables avec PK mais sans mesure
        for table_id in self.tables_info:
            schema = self.tables_schema[table_id]
            measure_fields = [f for f in schema if re.search(measure_pattern, f['name'])]

            has_pk = len(self.table_pk_candidates.get(table_id, [])) > 0
            has_measures = len(measure_fields) > 0

            if has_pk and not has_measures and table_id not in self.dimension_tables:
                self.dimension_tables.append(table_id)

        # Tables de faits : grandes tables avec des mesures
        for table_id, info in sorted(self.tables_info.items(),
                                     key=lambda x: x[1]['rows'], reverse=True):
            schema = self.tables_schema[table_id]
            measure_fields = [f for f in schema if re.search(measure_pattern, f['name'])]
            if len(measure_fields) >= 2 or (info['rows'] > 10000 and table_id not in self.dimension_tables):
                if table_id not in self.fact_tables:
                    self.fact_tables.append(table_id)
                    if len(self.fact_tables) >= 5:
                        break

        print("✅ Tables de faits :", self.fact_tables)
        print("✅ Tables de dimensions :", self.dimension_tables)

    def generate_star_schema_views(self):
        created_views = []
        for fact_table in self.fact_tables:
            joins = []
            for join in self.possible_joins:
                if join['table1'] == fact_table and join['table2'] in self.dimension_tables:
                    joins.append({
                        'dim_table': join['table2'],
                        'fact_field': join['field1'],
                        'dim_field': join['field2']
                    })
            if joins:
                view_name = f"star_{fact_table}"
                sql = self.generate_star_view_sql(fact_table, joins)
                self.create_view(view_name, sql)
                created_views.append(view_name)
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
            for dim in first_level_dims:
                for join in self.possible_joins:
                    if join['table1'] == dim['dim_table'] and join['table2'] in self.dimension_tables:
                        second_level_joins.append({
                            'parent_dim': dim['dim_table'],
                            'child_dim': join['table2'],
                            'parent_field': join['field1'],
                            'child_field': join['field2']
                        })

            if first_level_dims:
                view_name = f"snowflake_{fact_table}"
                sql = self.generate_snowflake_view_sql(fact_table, first_level_dims, second_level_joins)
                self.create_view(view_name, sql)
                created_views.append(view_name)
        return created_views

    def generate_star_view_sql(self, fact_table, joins):
        select_clause = ["f.*"]
        join_clauses = []
        for i, join in enumerate(joins):
            alias = f"d{i}"
            select_clause.append(f"{alias}.* EXCEPT ({join['dim_field']})")
            join_clauses.append(f"""
            LEFT JOIN `{self.project_id}.{self.source_dataset_id}.{join['dim_table']}` AS {alias}
            ON f.{join['fact_field']} = {alias}.{join['dim_field']}
            """)
        return f"""
        SELECT {', '.join(select_clause)}
        FROM `{self.project_id}.{self.source_dataset_id}.{fact_table}` AS f
        {' '.join(join_clauses)}
        """

    def generate_snowflake_view_sql(self, fact_table, first_level_dims, second_level_joins):
        select_clause = ["f.*"]
        join_clauses = []
        for i, dim in enumerate(first_level_dims):
            alias = f"d{i}"
            select_clause.append(f"{alias}.* EXCEPT ({dim['dim_field']})")
            join_clauses.append(f"""
            LEFT JOIN `{self.project_id}.{self.source_dataset_id}.{dim['dim_table']}` AS {alias}
            ON f.{dim['fact_field']} = {alias}.{dim['dim_field']}
            """)
            for j, sj in enumerate(second_level_joins):
                if sj['parent_dim'] == dim['dim_table']:
                    salias = f"d{i}_{j}"
                    select_clause.append(f"{salias}.* EXCEPT ({sj['child_field']})")
                    join_clauses.append(f"""
                    LEFT JOIN `{self.project_id}.{self.source_dataset_id}.{sj['child_dim']}` AS {salias}
                    ON {alias}.{sj['parent_field']} = {salias}.{sj['child_field']}
                    """)
        return f"""
        SELECT {', '.join(select_clause)}
        FROM `{self.project_id}.{self.source_dataset_id}.{fact_table}` AS f
        {' '.join(join_clauses)}
        """

    def create_view(self, view_name, sql):
        view_id = f"{self.project_id}.{self.target_dataset_id}.{view_name}"
        try:
            self.client.delete_table(view_id)
        except Exception:
            pass
        view = bigquery.Table(view_id)
        view.view_query = sql
        try:
            self.client.create_table(view)
            print(f"Vue créée : {view_id}")
            return True
        except Exception as e:
            print(f"❌ Erreur lors de la création de {view_name} : {e}")
            return False

    def get_cardinality(self, table_id, field_name):
        query = f"""
        SELECT COUNT(DISTINCT `{field_name}`) AS distinct_count 
        FROM `{self.project_id}.{self.source_dataset_id}.{table_id}`
        WHERE `{field_name}` IS NOT NULL
        """
        try:
            result = self.client.query(query).result()
            return list(result)[0].distinct_count
        except Exception as e:
            print(f"Erreur pour {table_id}.{field_name} : {e}")
            return 0

    def print_key_candidates(self):
        print("\n🔑 Candidats clés détectés :")
        for table_id in self.tables_info:
            pks = self.table_pk_candidates.get(table_id, [])
            fks = self.table_fk_candidates.get(table_id, [])
            if pks or fks:
                print(f"  🔹 {table_id}:")
                if pks:
                    print(f"     PKs : {', '.join(pks)}")
                if fks:
                    print(f"     FKs : {', '.join(fks)}")

    def analyze_and_create_views(self):
        print(f"\n🔍 Analyse du dataset '{self.source_dataset_id}' dans le projet '{self.project_id}'\n")
        self.list_tables()
        self.analyze_schemas()
        self.identify_fact_dimension_tables()
        star_views = self.generate_star_schema_views()
        snowflake_views = self.generate_snowflake_schema_views()
        return {
            "star_views": star_views,
            "snowflake_views": snowflake_views
        }

def main():
    analyzer = BigQuerySchemaAnalyzer(
        project_id="sandbox-jndong",
        source_dataset_id="DataOriginal",
        target_dataset_id="DataOriginal_matching"
    )
    results = analyzer.analyze_and_create_views()
    print("\n📊 Résumé des vues générées :")
    print(f"  Star Schema : {len(results['star_views'])}")
    print(f"  Snowflake Schema : {len(results['snowflake_views'])}")

if __name__ == "__main__":
    main()