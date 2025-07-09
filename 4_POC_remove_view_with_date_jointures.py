#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour supprimer les vues BigQuery d'un dataset qui ont été créées à partir de champs date.
Le script recherche les vues dont la définition SQL contient des fonctions de date ou des références
à des champs de type DATE ou TIMESTAMP.
"""

from google.cloud import bigquery
import re,os
import argparse
import logging


# Définition de la clé d'authentification Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./dbt_sa_bigquery.json"

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====================================================================
# CONFIGURATION EN DUR - Modifiez ces valeurs selon vos besoins
# ====================================================================
# Liste des datasets à traiter sous forme de tuples (project_id, dataset_id)
DATASETS = [
     ('sandbox-jndong', 'DataOriginal_matching')
     #,
    # ('votre-projet-id', 'autre-dataset-id'),
    # Ajoutez d'autres paires (project_id, dataset_id) selon vos besoins
]

# Mode simulation par défaut (ne supprime pas réellement les vues)  --> False = supprime ; True = Ne supprime pas 
DEFAULT_DRY_RUN = False #True
# ====================================================================

def get_views_in_dataset(client, project_id, dataset_id):
    """
    Récupère toutes les vues dans un dataset BigQuery spécifié.
    
    Args:
        client: Client BigQuery
        project_id: ID du projet
        dataset_id: ID du dataset
        
    Returns:
        Liste des vues trouvées
    """
    dataset_ref = client.dataset(dataset_id, project_id)
    
    # Récupérer la liste des tables et vues
    tables = list(client.list_tables(dataset_ref))
    
    # Filtrer pour ne garder que les vues
    views = [table for table in tables if table.table_type == 'VIEW']
    
    logger.info(f"Nombre total de vues trouvées dans {project_id}.{dataset_id}: {len(views)}")
    return views

def is_date_based_view(client, view):
    """
    Détermine si une vue est basée sur des champs date.
    
    Args:
        client: Client BigQuery
        view: Référence à la vue BigQuery
        
    Returns:
        Boolean indiquant si la vue est basée sur des champs date
    """
    # Récupérer les métadonnées de la vue pour avoir accès à la requête SQL
    view_details = client.get_table(view.reference)
    query = view_details.view_query
    
    if not query:
        logger.warning(f"Impossible de récupérer la requête pour la vue {view.table_id}")
        return False
    
    # Patterns pour identifier les vues basées sur des dates
    date_patterns = [
        # Fonctions de date
        r'DATE\(',
        r'TIMESTAMP\(',
        r'DATETIME\(',
        r'EXTRACT\(\s*(DATE|DATETIME|TIMESTAMP|YEAR|MONTH|DAY|HOUR|MINUTE|SECOND)',
        r'DATE_ADD',
        r'DATE_SUB',
        r'DATE_DIFF',
        r'FORMAT_DATE',
        r'PARSE_DATE',
        # Champs typiques de date
        r'\bdate\b',
        r'\btimestamp\b',
        r'\bdatetime\b',
        r'\bcreated_at\b',
        r'\bupdated_at\b',
        r'\bdate_creation\b',
        r'\bdate_modification\b'
    ]
    
    for pattern in date_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    
    return False

def delete_view(client, view):
    """
    Supprime une vue BigQuery.
    
    Args:
        client: Client BigQuery
        view: Référence à la vue à supprimer
        
    Returns:
        Boolean indiquant si la suppression a réussi
    """
    try:
        client.delete_table(view.reference)
        logger.info(f"Vue supprimée avec succès: {view.table_id}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de la vue {view.table_id}: {e}")
        return False

def process_dataset(client, project_id, dataset_id, dry_run):
    """
    Traite un dataset spécifique pour identifier et supprimer les vues basées sur des dates.
    
    Args:
        client: Client BigQuery
        project_id: ID du projet
        dataset_id: ID du dataset
        dry_run: Mode simulation (True) ou suppression réelle (False)
    
    Returns:
        Tuple (nombre de vues identifiées, nombre de vues supprimées)
    """
    logger.info(f"Traitement du dataset: {project_id}.{dataset_id}")
    
    # Récupérer les vues du dataset
    views = get_views_in_dataset(client, project_id, dataset_id)
    
    if not views:
        logger.info(f"Aucune vue trouvée dans le dataset {project_id}.{dataset_id}")
        return 0, 0
    
    # Identifier les vues basées sur des dates
    date_views = []
    for view in views:
        if is_date_based_view(client, view):
            date_views.append(view)
            logger.info(f"Vue basée sur date identifiée: {view.table_id}")
    
    logger.info(f"Nombre de vues basées sur date trouvées: {len(date_views)}")
    
    deleted_count = 0
    # Suppression des vues
    if dry_run:
        logger.info("Mode simulation activé. Aucune vue ne sera supprimée.")
        for view in date_views:
            logger.info(f"Simulation - Vue qui serait supprimée: {view.table_id}")
    else:
        logger.info("Début de la suppression des vues...")
        for view in date_views:
            if delete_view(client, view):
                deleted_count += 1
        
        logger.info(f"Suppression terminée. {deleted_count}/{len(date_views)} vues supprimées.")
    
    return len(date_views), deleted_count

def main():
    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Supprimer les vues BigQuery basées sur des champs date')
    parser.add_argument('--project', help='ID du projet Google Cloud')
    parser.add_argument('--dataset', help='ID du dataset BigQuery')
    parser.add_argument('--dry-run', action='store_true', help='Mode simulation (n\'effectue pas de suppression réelle)')
    parser.add_argument('--use-config', action='store_true', help='Utiliser la configuration en dur définie dans le script')
    args = parser.parse_args()
    
    # Définir si on utilise les arguments de ligne de commande ou la configuration en dur
    use_config = args.use_config or (not args.project and not args.dataset)
    
    # Initialiser le client BigQuery (utiliser le projet par défaut si aucun n'est spécifié)
    default_project = args.project if args.project else (DATASETS[0][0] if DATASETS else None)
    client = bigquery.Client(project=default_project)
    
    total_identified = 0
    total_deleted = 0
    
    if use_config:
        # Utiliser la configuration en dur
        logger.info("Utilisation de la configuration en dur définie dans le script")
        
        if not DATASETS:
            logger.error("Aucun dataset défini dans la configuration en dur. Veuillez modifier le script.")
            return
        
        dry_run = args.dry_run if args.dry_run else DEFAULT_DRY_RUN
        
        for project_id, dataset_id in DATASETS:
            identified, deleted = process_dataset(client, project_id, dataset_id, dry_run)
            total_identified += identified
            total_deleted += deleted
    else:
        # Utiliser les arguments de ligne de commande
        if not args.project or not args.dataset:
            logger.error("Les arguments --project et --dataset sont requis lorsque --use-config n'est pas utilisé.")
            return
        
        identified, deleted = process_dataset(client, args.project, args.dataset, args.dry_run)
        total_identified += identified
        total_deleted += deleted
    
    logger.info(f"Traitement terminé. Total: {total_identified} vues identifiées, {total_deleted} vues supprimées.")

if __name__ == "__main__":
    main()