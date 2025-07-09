import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
import plotly.express as px
import plotly.graph_objects as go
from pandas.plotting import scatter_matrix
import numpy as np




# Définition de la clé d'authentification Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Load_file_in_bigquery/dbt_sa_bigquery.json"

def authenticate_bigquery():
    """
    Authentification auprès de BigQuery.
    Assurez-vous d'avoir défini la variable d'environnement GOOGLE_APPLICATION_CREDENTIALS
    pointant vers votre fichier de clé JSON.
    """
    try:
        client = bigquery.Client()
        print("Authentification réussie à BigQuery!")
        return client
    except Exception as e:
        print(f"Erreur d'authentification: {e}")
        return None

def list_views(client, project_id, dataset_id):
    """
    Liste toutes les vues disponibles dans le dataset spécifié.
    """
    dataset_ref = client.dataset(dataset_id, project=project_id)
    try:
        tables = list(client.list_tables(dataset_ref))
        views = [table.table_id for table in tables if table.table_type == 'VIEW']
        if not views:
            print(f"Aucune vue trouvée dans le dataset {dataset_id}")
        return views
    except Exception as e:
        print(f"Erreur lors de la récupération des vues: {e}")
        return []

def get_view_data(client, project_id, dataset_id, view_id, limit=1000):
    """
    Récupère les données d'une vue spécifique.
    """
    query = f"""
    SELECT * 
    FROM `{project_id}.{dataset_id}.{view_id}`
    LIMIT {limit}
    """
    try:
        df = client.query(query).to_dataframe()
        print(f"Données récupérées avec succès: {len(df)} lignes")
        return df
    except Exception as e:
        print(f"Erreur lors de la récupération des données de la vue {view_id}: {e}")
        return None

def analyze_dataframe(df, view_name):
    """
    Analyse exploratoire des données et création de visualisations.
    """
    if df is None or df.empty:
        print("Aucune donnée à analyser")
        return

    print(f"\n--- Analyse de la vue: {view_name} ---")
    
    # Informations générales sur les données
    print("\nInformations sur le dataframe:")
    print(f"Dimensions: {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print("\nTypes de données:")
    print(df.dtypes)
    
    # Statistiques descriptives
    print("\nStatistiques descriptives:")
    try:
        desc = df.describe(include='all')
        print(desc)
    except Exception as e:
        print(f"Impossible de générer des statistiques descriptives: {e}")
    
    # Créer un dossier pour les visualisations
    viz_folder = f"visualisations_{view_name}"
    os.makedirs(viz_folder, exist_ok=True)
    
    # Analyse des colonnes numériques
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        print(f"\nAnalyse des {len(numeric_cols)} colonnes numériques...")
        
        # Histogrammes pour chaque colonne numérique
        for col in numeric_cols[:min(5, len(numeric_cols))]:  # Limiter à 5 colonnes pour éviter trop de graphiques
            try:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f'Distribution de {col}')
                plt.tight_layout()
                plt.savefig(f"{viz_folder}/hist_{col}.png")
                plt.close()
                
                # Boîte à moustaches
                plt.figure(figsize=(10, 6))
                sns.boxplot(y=df[col].dropna())
                plt.title(f'Boîte à moustaches de {col}')
                plt.tight_layout()
                plt.savefig(f"{viz_folder}/boxplot_{col}.png")
                plt.close()
            except Exception as e:
                print(f"Erreur lors de la création du graphique pour {col}: {e}")
        
        # Matrice de corrélation (uniquement si plus d'une colonne numérique)
        if len(numeric_cols) > 1:
            try:
                plt.figure(figsize=(12, 10))
                corr_matrix = df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
                plt.title('Matrice de corrélation')
                plt.tight_layout()
                plt.savefig(f"{viz_folder}/correlation_matrix.png")
                plt.close()
                
                # Scatter matrix (maximum 5 colonnes pour la lisibilité)
                if len(numeric_cols) >= 2 and len(numeric_cols) <= 5:
                    scatter_matrix(df[numeric_cols], figsize=(15, 15), diagonal='kde')
                    plt.suptitle('Matrice de dispersion', y=1.02)
                    plt.tight_layout()
                    plt.savefig(f"{viz_folder}/scatter_matrix.png")
                    plt.close()
            except Exception as e:
                print(f"Erreur lors de la création de la matrice de corrélation: {e}")
    
    # Analyse des colonnes catégorielles
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    if len(cat_cols) > 0:
        print(f"\nAnalyse des {len(cat_cols)} colonnes catégorielles...")
        
        for col in cat_cols[:min(5, len(cat_cols))]:  # Limiter à 5 colonnes
            try:
                # Vérifier le nombre de valeurs uniques
                value_counts = df[col].value_counts()
                if len(value_counts) <= 20:  # Limiter aux colonnes avec un nombre raisonnable de catégories
                    plt.figure(figsize=(12, 8))
                    sns.countplot(y=df[col], order=value_counts.index)
                    plt.title(f'Distribution de {col}')
                    plt.tight_layout()
                    plt.savefig(f"{viz_folder}/countplot_{col}.png")
                    plt.close()
                    
                    # Créer un graphique en camembert avec Plotly
                    fig = px.pie(
                        names=value_counts.index,
                        values=value_counts.values,
                        title=f'Répartition de {col}'
                    )
                    fig.write_html(f"{viz_folder}/pie_{col}.html")
            except Exception as e:
                print(f"Erreur lors de la création du graphique pour {col}: {e}")
    
    # Analyse temporelle (si colonnes datetime présentes)
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0:
        print(f"\nAnalyse des {len(date_cols)} colonnes temporelles...")
        
        for date_col in date_cols[:min(2, len(date_cols))]:
            try:
                # Sélectionner une colonne numérique pour l'analyse temporelle
                if len(numeric_cols) > 0:
                    num_col = numeric_cols[0]
                    
                    # Créer une série temporelle
                    df_time = df.copy()
                    df_time.set_index(date_col, inplace=True)
                    time_series = df_time[num_col].resample('D').mean()
                    
                    plt.figure(figsize=(14, 8))
                    plt.plot(time_series)
                    plt.title(f'Évolution de {num_col} dans le temps')
                    plt.xlabel('Date')
                    plt.ylabel(num_col)
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(f"{viz_folder}/timeseries_{date_col}_{num_col}.png")
                    plt.close()
            except Exception as e:
                print(f"Erreur lors de la création du graphique temporel pour {date_col}: {e}")
    
    # Analyse bivariée (entre une colonne numérique et une colonne catégorielle)
    if len(numeric_cols) > 0 and len(cat_cols) > 0:
        try:
            num_col = numeric_cols[0]
            cat_col = cat_cols[0]
            
            # Vérifier le nombre de catégories
            if df[cat_col].nunique() <= 10:
                plt.figure(figsize=(12, 8))
                sns.boxplot(x=cat_col, y=num_col, data=df)
                plt.title(f'Distribution de {num_col} par {cat_col}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"{viz_folder}/boxplot_{num_col}_by_{cat_col}.png")
                plt.close()
                
                # Violinplot
                plt.figure(figsize=(14, 8))
                sns.violinplot(x=cat_col, y=num_col, data=df)
                plt.title(f'Violinplot de {num_col} par {cat_col}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"{viz_folder}/violinplot_{num_col}_by_{cat_col}.png")
                plt.close()
        except Exception as e:
            print(f"Erreur lors de la création des graphiques bivariés: {e}")
    
    # Créer un tableau de bord interactif avec Plotly
    try:
        if len(numeric_cols) > 0:
            # Créer un histogramme interactif avec Plotly
            num_col = numeric_cols[0]
            fig = px.histogram(
                df, x=num_col, 
                marginal="box", 
                opacity=0.7,
                title=f"Distribution de {num_col}"
            )
            fig.write_html(f"{viz_folder}/interactive_hist_{num_col}.html")
        
        # Si nous avons deux colonnes numériques, créer un scatter plot
        if len(numeric_cols) >= 2:
            fig = px.scatter(
                df, x=numeric_cols[0], y=numeric_cols[1],
                color=cat_cols[0] if len(cat_cols) > 0 else None,
                title=f"Relation entre {numeric_cols[0]} et {numeric_cols[1]}"
            )
            fig.write_html(f"{viz_folder}/interactive_scatter.html")
    except Exception as e:
        print(f"Erreur lors de la création des graphiques interactifs: {e}")
    
    print(f"\nVisualisations sauvegardées dans le dossier '{viz_folder}'")
    return viz_folder

def main():
    # Configuration
   # project_id = input("Entrez votre Project ID BigQuery: ")
   # dataset_id = input("Entrez votre Dataset ID: ")

    project_id = "sandbox-jndong"
    #dataset_id = "DataOriginal"
    dataset_id = "DataOriginal_matching"
    
    # Authentification
    client = authenticate_bigquery()
    if not client:
        return
    
    # Récupération des vues disponibles
    views = list_views(client, project_id, dataset_id)
    if not views:
        return
    
    print("\nVues disponibles:")
    for i, view in enumerate(views, 1):
        print(f"{i}. {view}")
    
    # Sélection de la vue à analyser
    try:
        choice = int(input("\nEntrez le numéro de la vue à analyser (0 pour toutes): "))
        if choice == 0:
            selected_views = views
        elif 1 <= choice <= len(views):
            selected_views = [views[choice-1]]
        else:
            print("Choix invalide")
            return
    except ValueError:
        print("Veuillez entrer un nombre valide")
        return
    
    # Analyse des vues sélectionnées
    all_viz_folders = []
    for view in selected_views:
        print(f"\nRécupération des données de la vue: {view}")
        df = get_view_data(client, project_id, dataset_id, view)
        if df is not None:
            viz_folder = analyze_dataframe(df, view)
            if viz_folder:
                all_viz_folders.append(viz_folder)
    
    if all_viz_folders:
        print("\nRésumé des analyses:")
        for folder in all_viz_folders:
            print(f"- Visualisations pour {folder} générées avec succès")
        print("\nPour une analyse plus approfondie, vous pouvez explorer les visualisations statiques (.png) et interactives (.html) dans les dossiers mentionnés ci-dessus.")

if __name__ == "__main__":
    main()