from google.cloud import bigquery
import pandas as pd
import re,os
import itertools
from collections import defaultdict


# Définition de la clé d'authentification Google Cloud
CREDENTIAL = "Load_file_in_bigquery/dbt_sa_bigquery.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIAL

class BigQuerySchemaAnalyzer:
    def __init__(self, project_id, source_dataset_id, target_dataset_id):
        """
        Initialise l'analyseur de schéma BigQuery.
        
        Args:
            project_id (str): ID du projet Google Cloud
            source_dataset_id (str): ID du dataset source à analyser
            target_dataset_id (str): ID du dataset cible où les vues seront créées
        """
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
    
    def list_tables(self):
        """Liste uniquement les tables (pas les vues) dans le dataset source"""
        dataset_ref = self.client.dataset(self.source_dataset_id)
        all_tables = list(self.client.list_tables(dataset_ref))
    
        print(f"Tables trouvées dans {self.source_dataset_id}:")
        tables_only = []
    
        for table in all_tables:
        # Obtenir l'objet table complet pour vérifier s'il s'agit d'une vue
            table_ref = self.client.dataset(self.source_dataset_id).table(table.table_id)
            table_obj = self.client.get_table(table_ref)
        
        # Vérifier si c'est une table (pas une vue)
            if not table_obj.view_query:
                table_id = table.table_id
                print(f"- {table_id}")
                self.tables_info[table_id] = {'rows': self.get_row_count(table_id)}
                tables_only.append(table_id)
    
        return tables_only
    
    def get_row_count(self, table_id):
        """Obtient le nombre de lignes dans une table"""
        query = f"""
        SELECT COUNT(*) as count 
        FROM `{self.project_id}.{self.source_dataset_id}.{table_id}`
        """
        query_job = self.client.query(query)
        result = query_job.result()
        count = list(result)[0].count
        return count
    
    def get_schema(self, table_id):
        """Obtient le schéma d'une table"""
        table_ref = self.client.dataset(self.source_dataset_id).table(table_id)
        table = self.client.get_table(table_ref)
        return table.schema
    
    def analyze_schemas(self):
        """Analyse le schéma de toutes les tables et identifie les champs communs"""
        for table_id in self.tables_info:
            schema = self.get_schema(table_id)
            fields = []
            
            for field in schema:
                field_name = field.name.lower()
                field_type = field.field_type
                
                fields.append({
                    'name': field_name,
                    'type': field_type,
                    'normalized_name': self.normalize_field_name(field_name)
                })
                
                # Enregistrer la correspondance champ -> table
                self.field_to_tables[self.normalize_field_name(field_name)].append({
                    'table_id': table_id,
                    'field_name': field_name,
                    'field_type': field_type
                })
            
            self.tables_schema[table_id] = fields
        
        # Identifier les correspondances possibles pour les jointures
        self.identify_potential_joins()
        
    def normalize_field_name(self, field_name):
        """Normalise le nom du champ pour faciliter la recherche de correspondances"""
        # Supprime les préfixes/suffixes courants (id_, _id, _key, etc.)
        normalized = re.sub(r'^(id_|pk_|fk_)', '', field_name)
        normalized = re.sub(r'(_id|_key|_pk|_fk)$', '', normalized)
        
        # Supprime les caractères non alphanumériques
        normalized = re.sub(r'[^a-z0-9]', '', normalized)
        
        return normalized
    
    def identify_potential_joins(self):
        """Identifie les jointures potentielles entre les tables"""
        # Trouver les champs qui apparaissent dans plus d'une table
        for normalized_field, tables in self.field_to_tables.items():
            if len(tables) > 1:
                # Créer des combinaisons de tables pour ce champ
                for i, table1_info in enumerate(tables):
                    for table2_info in tables[i+1:]:
                        # Vérifier que les types sont compatibles
                        if table1_info['field_type'] == table2_info['field_type']:
                            self.possible_joins.append({
                                'table1': table1_info['table_id'],
                                'field1': table1_info['field_name'],
                                'table2': table2_info['table_id'],
                                'field2': table2_info['field_name'],
                                'similarity': 'exact' if table1_info['field_name'] == table2_info['field_name'] else 'normalized',
                                'normalized_field': normalized_field
                            })
        
        print(f"Trouvé {len(self.possible_joins)} jointures potentielles.")
    
    def identify_fact_dimension_tables(self):
        """Identifie les tables de faits et les tables de dimensions"""
        # Pour simplifier, on considère que:
        # - Les grandes tables (plus de lignes) sont généralement des tables de faits
        # - Les tables qui sont référencées par plusieurs autres sont des dimensions
        
        # Compter le nombre de références entrantes pour chaque table
        inbound_refs = defaultdict(int)
        for join in self.possible_joins:
            inbound_refs[join['table2']] += 1
        
        # Tables les plus référencées sont probablement des dimensions
        sorted_refs = sorted(inbound_refs.items(), key=lambda x: x[1], reverse=True)
        
        # Première passe: les tables avec beaucoup de références entrantes sont des dimensions
        for table_id, ref_count in sorted_refs:
            if ref_count >= 2:  # Seuil arbitraire
                self.dimension_tables.append(table_id)
        
        # Deuxième passe: les grandes tables qui ne sont pas des dimensions sont des faits
        for table_id, info in sorted(self.tables_info.items(), key=lambda x: x[1]['rows'], reverse=True):
            if table_id not in self.dimension_tables:
                self.fact_tables.append(table_id)
                if len(self.fact_tables) >= 3:  # Limite arbitraire
                    break
        
        print("Tables de faits identifiées:", self.fact_tables)
        print("Tables de dimensions identifiées:", self.dimension_tables)
    
    def generate_star_schema_views(self):
        """Génère des vues de schéma en étoile"""
        created_views = []
        
        # Pour chaque table de faits, créer une vue qui la joint avec les dimensions appropriées
        for fact_table in self.fact_tables:
            joins = []
            
            # Trouver toutes les jointures possibles avec des tables de dimensions
            for join in self.possible_joins:
                if join['table1'] == fact_table and join['table2'] in self.dimension_tables:
                    joins.append({
                        'fact_table': fact_table,
                        'dim_table': join['table2'],
                        'fact_field': join['field1'],
                        'dim_field': join['field2']
                    })
            
            if joins:
                # Créer une vue avec toutes les jointures possibles
                view_name = f"star_{fact_table}"
                sql = self.generate_star_view_sql(fact_table, joins)
                self.create_view(view_name, sql)
                created_views.append(view_name)
        
        return created_views
    
    def generate_snowflake_schema_views(self):
        """Génère des vues de schéma en flocon"""
        created_views = []
        
        # Pour chaque table de faits, créer une vue plus complexe avec des jointures imbriquées
        for fact_table in self.fact_tables:
            # Trouver les dimensions de premier niveau
            first_level_dims = []
            for join in self.possible_joins:
                if join['table1'] == fact_table and join['table2'] in self.dimension_tables:
                    first_level_dims.append({
                        'dim_table': join['table2'],
                        'fact_field': join['field1'],
                        'dim_field': join['field2']
                    })
            
            # Pour chaque dimension de premier niveau, trouver les dimensions de second niveau
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
                # Créer une vue avec des jointures à plusieurs niveaux
                view_name = f"snowflake_{fact_table}"
                sql = self.generate_snowflake_view_sql(fact_table, first_level_dims, second_level_joins)
                self.create_view(view_name, sql)
                created_views.append(view_name)
        
        return created_views
    
    def generate_star_view_sql(self, fact_table, joins):
        """Génère le SQL pour une vue en schéma étoile"""
        # Sélectionner tous les champs de la table de faits
        select_clause = [f"f.* "]
        
        # Ajouter les champs des dimensions (en excluant les clés utilisées pour joindre)
        join_clauses = []
        
        for i, join in enumerate(joins):
            dim_alias = f"d{i}"
            
            # Ajouter les champs de la dimension au SELECT
            dim_table_ref = self.client.dataset(self.source_dataset_id).table(join['dim_table'])
            dim_schema = self.client.get_table(dim_table_ref).schema
            
            for field in dim_schema:
                if field.name != join['dim_field']:  # Exclure la clé de jointure
                    select_clause.append(f"{dim_alias}.{field.name} AS {join['dim_table']}_{field.name}")
            
            # Ajouter la clause JOIN
            join_clauses.append(f"""
            LEFT JOIN `{self.project_id}.{self.source_dataset_id}.{join['dim_table']}` AS {dim_alias}
            ON f.{join['fact_field']} = {dim_alias}.{join['dim_field']}
            """)
        
        # Construire la requête SQL complète
        sql = f"""
        SELECT
            {', '.join(select_clause)}
        FROM 
            `{self.project_id}.{self.source_dataset_id}.{fact_table}` AS f
            {' '.join(join_clauses)}
        """
        
        return sql
    
    def generate_snowflake_view_sql(self, fact_table, first_level_dims, second_level_joins):
        """Génère le SQL pour une vue en schéma flocon"""
        # Sélectionner tous les champs de la table de faits
        select_clause = [f"f.* "]
        
        # Ajouter les champs des dimensions de premier niveau
        join_clauses = []
        
        for i, dim in enumerate(first_level_dims):
            dim_alias = f"d{i}"
            
            # Ajouter les champs de la dimension au SELECT
            dim_table_ref = self.client.dataset(self.source_dataset_id).table(dim['dim_table'])
            dim_schema = self.client.get_table(dim_table_ref).schema
            
            for field in dim_schema:
                if field.name != dim['dim_field']:  # Exclure la clé de jointure
                    select_clause.append(f"{dim_alias}.{field.name} AS {dim['dim_table']}_{field.name}")
            
            # Ajouter la clause JOIN pour la dimension de premier niveau
            join_clauses.append(f"""
            LEFT JOIN `{self.project_id}.{self.source_dataset_id}.{dim['dim_table']}` AS {dim_alias}
            ON f.{dim['fact_field']} = {dim_alias}.{dim['dim_field']}
            """)
            
            # Ajouter les jointures de second niveau pour cette dimension
            for j, second_join in enumerate(second_level_joins):
                if second_join['parent_dim'] == dim['dim_table']:
                    second_dim_alias = f"d{i}_{j}"
                    
                    # Ajouter les champs de la dimension de second niveau au SELECT
                    second_dim_table_ref = self.client.dataset(self.source_dataset_id).table(second_join['child_dim'])
                    second_dim_schema = self.client.get_table(second_dim_table_ref).schema
                    
                    for field in second_dim_schema:
                        if field.name != second_join['child_field']:  # Exclure la clé de jointure
                            select_clause.append(f"{second_dim_alias}.{field.name} AS {second_join['child_dim']}_{field.name}")
                    
                    # Ajouter la clause JOIN pour la dimension de second niveau
                    join_clauses.append(f"""
                    LEFT JOIN `{self.project_id}.{self.source_dataset_id}.{second_join['child_dim']}` AS {second_dim_alias}
                    ON {dim_alias}.{second_join['parent_field']} = {second_dim_alias}.{second_join['child_field']}
                    """)
        
        # Construire la requête SQL complète
        sql = f"""
        SELECT
            {', '.join(select_clause)}
        FROM 
            `{self.project_id}.{self.source_dataset_id}.{fact_table}` AS f
            {' '.join(join_clauses)}
        """
        
        return sql
    
    def create_view(self, view_name, sql):
        """Crée une vue dans BigQuery"""
        view_id = f"{self.project_id}.{self.target_dataset_id}.{view_name}"
        view = bigquery.Table(view_id)
        view.view_query = sql
        
        try:
            # Supprimer la vue si elle existe déjà
            try:
                self.client.delete_table(view_id)
            except:
                pass
            
            # Créer la vue
            self.client.create_table(view)
            print(f"Vue créée: {view_id}")
            return True
        except Exception as e:
            print(f"Erreur lors de la création de la vue {view_name}: {e}")
            return False
    
    def analyze_and_create_views(self):
        """Analyse les tables et crée les vues"""
        print(f"Analyse du dataset {self.source_dataset_id} dans le projet {self.project_id}")
        
        # Étape 1: Lister toutes les tables
        self.list_tables()
        
        # Étape 2: Analyser les schémas
        self.analyze_schemas()
        
        # Étape 3: Identifier les tables de faits et de dimensions
        self.identify_fact_dimension_tables()
        
        # Étape 4: Créer des vues en schéma étoile
        star_views = self.generate_star_schema_views()
        print(f"Vues en schéma étoile créées: {star_views}")
        
        # Étape 5: Créer des vues en schéma flocon
        snowflake_views = self.generate_snowflake_schema_views()
        print(f"Vues en schéma flocon créées: {snowflake_views}")
        
        # Étape 6: Créer également des vues pour les jointures significatives
        self.create_custom_join_views()
        
        return {
            "star_views": star_views,
            "snowflake_views": snowflake_views
        }
    
    def create_custom_join_views(self):
        """Crée des vues pour des jointures significatives qui ne sont pas déjà dans les schémas étoile/flocon"""
        # Regrouper les jointures par paires de tables
        table_pairs = defaultdict(list)
        for join in self.possible_joins:
            pair_key = tuple(sorted([join['table1'], join['table2']]))
            table_pairs[pair_key].append(join)
        
        for pair, joins in table_pairs.items():
            table1, table2 = pair
            
            # Ne pas recréer des vues pour les tables déjà jointes dans les schémas étoile/flocon
            if (table1 in self.fact_tables and table2 in self.dimension_tables) or \
               (table2 in self.fact_tables and table1 in self.dimension_tables):
                continue
            
            if len(joins) > 0:
                # Utiliser la première jointure pour créer une vue simple
                join = joins[0]
                view_name = f"join_{table1}_{table2}"
                
                # Déterminer quelle table est à gauche et à droite de la jointure
                left_table, right_table = (join['table1'], join['table2'])
                left_field, right_field = (join['field1'], join['field2'])
                
                sql = f"""
                SELECT 
                    t1.*, 
                    t2.*
                FROM 
                    `{self.project_id}.{self.source_dataset_id}.{left_table}` AS t1
                LEFT JOIN 
                    `{self.project_id}.{self.source_dataset_id}.{right_table}` AS t2
                ON 
                    t1.{left_field} = t2.{right_field}
                """
                
                self.create_view(view_name, sql)


def main():
    """Fonction principale pour exécuter l'analyseur"""
    # Ces valeurs devraient être remplacées par les paramètres spécifiques du projet
    project_id = "sandbox-jndong"
    source_dataset_id = "DataOriginal"
    target_dataset_id = "DataOriginal_matching"
    
    # Créer et exécuter l'analyseur
    analyzer = BigQuerySchemaAnalyzer(
        project_id=project_id,
        source_dataset_id=source_dataset_id,
        target_dataset_id=target_dataset_id
    )
    
    # Lancer l'analyse et la création des vues
    results = analyzer.analyze_and_create_views()
    
    print("\nRésumé des vues créées:")
    print(f"Vues en schéma étoile: {len(results['star_views'])}")
    print(f"Vues en schéma flocon: {len(results['snowflake_views'])}")


if __name__ == "__main__":
    main()