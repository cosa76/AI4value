import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import panel as pn
from google.cloud import bigquery
from functools import partial

pn.extension('plotly')

# === Configuration en dur ===
PROJECT_ID = "sandbox-jndong"
DATASET_ID = "DataOriginal2"
FOLDER_PATH = os.path.join(os.path.dirname(__file__), "assets_panel")  # Dossier d'exports visuels
LOG_FILE = "./log_insights.log"
CREDENTIALS_JSON = "Load_file_in_bigquery/dbt_sa_bigquery.json"  # Chemin vers votre fichier JSON Google Cloud
MAX_TABLES = 5  # Limiter le nombre de tables traitées pour test
ROWS_PER_PAGE = 50  # Pagination

os.makedirs(FOLDER_PATH, exist_ok=True)

# === Configuration du logging ===
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === Connexion à BigQuery ===
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_JSON
client = bigquery.Client(project=PROJECT_ID)

# === Fonction : Charger les noms des tables ===
def get_tables_from_dataset():
    dataset_ref = client.dataset(DATASET_ID, project=PROJECT_ID)
    tables = list(client.list_tables(dataset_ref))
    return [table.table_id for table in tables[:MAX_TABLES]]

# === Fonction : Charger les données en DataFrame ===
def load_table_data(table_id):
    full_table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_id}"
    query = f"SELECT * FROM `{full_table_id}` LIMIT 10000"  # Plus de lignes pour pagination
    df = client.query(query).to_dataframe()
    return df

# === Pagination du DataFrame ===
class PaginatedDataFrame:
    def __init__(self, df):
        self.df = df
        self.total_pages = (len(df) // ROWS_PER_PAGE) + 1
        self.current_page = 0

    def get_page(self, page):
        start = page * ROWS_PER_PAGE
        end = start + ROWS_PER_PAGE
        return self.df.iloc[start:end]

    def prev(self):
        if self.current_page > 0:
            self.current_page -= 1
        return self.get_page(self.current_page)

    def next(self):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
        return self.get_page(self.current_page)

# === Fonction : Générer des insights visuels avec Panel ===
def generate_panel_insights(df_original, table_name):
    table_folder = os.path.join(FOLDER_PATH, table_name)
    os.makedirs(table_folder, exist_ok=True)

    insights = []

    insights.append(pn.pane.Markdown(f"# 📊 Insights pour la table: {table_name}"))

    # --- Widgets de filtrage ---
    numeric_cols = df_original.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df_original.select_dtypes(include=['object']).columns.tolist()

    filters = {}
    range_sliders = {}

    # Filtres catégoriels
    for col in categorical_cols:
        unique_values = ['Toutes'] + df_original[col].dropna().unique().tolist()
        filters[col] = pn.widgets.Select(name=col, options=unique_values)

    # Filtres numériques
    for col in numeric_cols:
        min_val, max_val = df_original[col].min(), df_original[col].max()
        if pd.notnull(min_val) and pd.notnull(max_val):
            range_sliders[col] = pn.widgets.RangeSlider(name=col, start=min_val, end=max_val, value=(min_val, max_val))

    filter_widgets = list(filters.values()) + list(range_sliders.values())
    reset_button = pn.widgets.Button(name="Réinitialiser", button_type="primary")

    # --- DataFrame paginé ---
    paginator = PaginatedDataFrame(df_original)
    df_pane = pn.pane.DataFrame(paginator.get_page(0), width=900)

    # Variable partagée
    class SharedState:
        filtered_df = df_original.copy()

    # Fonctions de mise à jour
    def update_df(event=None):
        # Appliquer les filtres
        filtered_df = df_original.copy()

        for col, widget in filters.items():
            if widget.value != 'Toutes':
                filtered_df = filtered_df[filtered_df[col] == widget.value]

        for col, widget in range_sliders.items():
            filtered_df = filtered_df[(filtered_df[col] >= widget.value[0]) & (filtered_df[col] <= widget.value[1])]

        SharedState.filtered_df = filtered_df
        paginator.df = filtered_df
        paginator.current_page = 0
        df_pane.object = paginator.get_page(0)

    def reset_filters(event):
        for w in filters.values():
            w.value = 'Toutes'
        for w in range_sliders.values():
            w.value = w.start, w.end
        update_df()

    reset_button.on_click(reset_filters)

    for widget in filter_widgets:
        widget.param.watch(update_df, 'value', onlychanged=False)

    # Pagination buttons
    prev_button = pn.widgets.Button(name="Précédent")
    next_button = pn.widgets.Button(name="Suivant")

    def prev_page(event):
        df_pane.object = paginator.prev()

    def next_page(event):
        df_pane.object = paginator.next()

    prev_button.on_click(prev_page)
    next_button.on_click(next_page)

    # Ajouter au dashboard
    insights.append(pn.pane.Markdown("## Filtres"))
    insights.append(pn.Column(*filter_widgets, reset_button))
    insights.append(pn.pane.Markdown("## Aperçu des données (Paginé)"))
    insights.append(pn.Row(prev_button, next_button))
    insights.append(df_pane)

    # Initialiser filtered_df avant utilisation
    update_df()

    # --- Visualisations ---
    insights.append(pn.pane.Markdown("## Statistiques descriptives"))
    try:
        insights.append(pn.pane.DataFrame(SharedState.filtered_df.describe().T, width=800))
    except Exception as e:
        logging.error(f"[{table_name}] Échec des statistiques descriptives : {str(e)}")

    if numeric_cols:
        insights.append(pn.pane.Markdown("## Distributions des variables numériques"))
        for col in numeric_cols[:5]:
            fig = px.histogram(SharedState.filtered_df, x=col, marginal="box", title=f"Distribution de {col}")
            insights.append(pn.pane.Plotly(fig, height=400))

    if categorical_cols:
        insights.append(pn.pane.Markdown("## Distributions des variables catégorielles"))
        for col in categorical_cols[:5]:
            value_counts = SharedState.filtered_df[col].value_counts().reset_index()
            value_counts.columns = [col, 'count']
            if len(value_counts) > 15:
                value_counts = pd.concat([
                    value_counts.head(10),
                    pd.DataFrame({col: ['Autres'], 'count': [value_counts.iloc[10:]['count'].sum()]})
                ])
            fig = px.bar(value_counts, x=col, y='count', title=f"Répartition de {col}")
            insights.append(pn.pane.Plotly(fig, height=400))

    if len(numeric_cols) >= 2:
        insights.append(pn.pane.Markdown("## Corrélations entre variables numériques"))
        corr = SharedState.filtered_df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Matrice de corrélation")
        insights.append(pn.pane.Plotly(fig, height=600))

    if len(numeric_cols) >= 2:
        insights.append(pn.pane.Markdown("## Relations entre variables"))
        fig = px.scatter(SharedState.filtered_df, x=numeric_cols[0], y=numeric_cols[1],
                         title=f"Relation entre {numeric_cols[0]} et {numeric_cols[1]}")
        insights.append(pn.pane.Plotly(fig, height=400))

    logging.info(f"[{table_name}] Insights Panel générés.")
    return insights

# === Fonction principale pour créer le tableau de bord Panel ===
def create_panel_dashboard(tables_with_insights, tables_data):
    tabs = pn.Tabs()

    current_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    welcome = pn.Column(
        pn.pane.Markdown(f"""
        # 🔍 Dashboard Automatique d'Insights Dataset BigQuery
        
        ## Projet : {PROJECT_ID}
        ## Dataset : {DATASET_ID}
        
        Ce tableau de bord présente une analyse automatique de vos tables BigQuery.
        Sélectionnez un onglet pour explorer les insights de chaque table.
        
        *Généré le : {current_date}*
        """),
        pn.pane.Markdown("### Tables analysées :"),
        pn.pane.Markdown("\n".join([f"- {table}" for table in tables_with_insights]))
    )

    tabs.append(("Accueil", welcome))

    for table_name in tables_with_insights:
        df = tables_data[table_name]
        insights = generate_panel_insights(df, table_name)
        tab_content = pn.Column(*insights, width=900)
        tabs.append((table_name, tab_content))

    dashboard = pn.template.MaterialTemplate(
        title="Insights BigQuery Dashboard",
        sidebar=["### Navigation", tabs],
        main=[pn.Column(
            pn.Row(
                pn.pane.Markdown(f"# 🔍 Dashboard Insights BigQuery"),
                pn.pane.Markdown(f"*Projet: {PROJECT_ID} | Dataset: {DATASET_ID}*")
            ),
            tabs[1:] if len(tabs) > 1 else pn.pane.Markdown("### Aucune donnée disponible")
        )]
    )
    return dashboard

# === Main pipeline ===
def main():
    logging.info("Début du traitement")
    tables = get_tables_from_dataset()
    logging.info(f"Tables trouvées : {tables}")

    tables_with_insights = []
    tables_data = {}

    for table_id in tables:
        logging.info(f"Traitement de la table : {table_id}")
        try:
            df = load_table_data(table_id)
            if not df.empty:
                tables_data[table_id] = df
                tables_with_insights.append(table_id)
            else:
                logging.warning(f"[{table_id}] DataFrame vide.")
        except Exception as e:
            logging.error(f"Échec traitement table {table_id} : {str(e)}")

    if tables_with_insights:
        dashboard = create_panel_dashboard(tables_with_insights, tables_data)
        pn.config.sizing_mode = "stretch_width"
        pn.config.allow_websocket_origin = ["127.0.0.1:5006", "localhost:5006"]
        pn.serve(
            dashboard,
            port=5006,
            address="127.0.0.1",
            show=True,
            websocket_origin=["127.0.0.1:5006", "localhost:5006"]
        )
    else:
        logging.error("Aucune table avec des insights à afficher.")

if __name__ == "__main__":
    main()