import os
import logging
import pandas as pd
import plotly.express as px
import panel as pn
from google.cloud import bigquery
from functools import partial

pn.extension('plotly')

# === CONFIGURATION ===
PROJECT_ID = "sandbox-jndong"
DATASET_ID = "DataOriginal2"
CREDENTIALS_JSON = "Load_file_in_bigquery/dbt_sa_bigquery.json"
MAX_TABLES = 5
ROWS_PER_PAGE = 50

# === Logging ===
logging.basicConfig(
    filename="log_insights.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === Auth BigQuery ===
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_JSON
client = bigquery.Client(project=PROJECT_ID)

# === Pagination Helper ===
class PaginatedDataFrame:
    def __init__(self, df):
        self.df = df
        self.current_page = 0
        self.total_pages = max((len(df) + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE, 1)

    def get_page(self):
        start = self.current_page * ROWS_PER_PAGE
        end = start + ROWS_PER_PAGE
        return self.df.iloc[start:end]

    def next_page(self, event=None):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
        return self.get_page()

    def prev_page(self, event=None):
        if self.current_page > 0:
            self.current_page -= 1
        return self.get_page()

# === BigQuery ===
def get_tables():
    dataset_ref = client.dataset(DATASET_ID)
    tables = list(client.list_tables(dataset_ref))
    return [t.table_id for t in tables[:MAX_TABLES]]

def load_data(table_id):
    full_id = f"{PROJECT_ID}.{DATASET_ID}.{table_id}"
    query = f"SELECT * FROM `{full_id}` LIMIT 10000"
    return client.query(query).to_dataframe()

# === Insights Generator ===
def generate_insights(df, table_name):
    tabs = []

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    filters = {}
    sliders = {}
    for col in categorical_cols:
        options = ['Toutes'] + sorted(df[col].dropna().unique().tolist())
        filters[col] = pn.widgets.Select(name=col, options=options)

    for col in numeric_cols:
        min_val, max_val = df[col].min(), df[col].max()
        sliders[col] = pn.widgets.RangeSlider(name=col, start=min_val, end=max_val, value=(min_val, max_val))

    filtered_df = df.copy()
    df_pane = pn.pane.DataFrame(filtered_df.head(ROWS_PER_PAGE), width=1200)
    paginator = PaginatedDataFrame(filtered_df)

    def update(event=None):
        fdf = df.copy()
        for col, widget in filters.items():
            if widget.value != 'Toutes':
                fdf = fdf[fdf[col] == widget.value]
        for col, widget in sliders.items():
            fdf = fdf[(fdf[col] >= widget.value[0]) & (fdf[col] <= widget.value[1])]
        paginator.df = fdf
        paginator.current_page = 0
        df_pane.object = paginator.get_page()

    for widget in list(filters.values()) + list(sliders.values()):
        widget.param.watch(update, 'value')

    prev_btn = pn.widgets.Button(name="◀️ Précédent")
    next_btn = pn.widgets.Button(name="Suivant ▶️")

    def prev(event): df_pane.object = paginator.prev_page()
    def next_(event): df_pane.object = paginator.next_page()
    prev_btn.on_click(prev)
    next_btn.on_click(next_)

    # === Layout ===
    controls = pn.Column(*filters.values(), *sliders.values())
    pagination = pn.Row(prev_btn, next_btn)
    data_table = pn.Column(pagination, df_pane)

    tabs.append(("📄 Données", pn.Column(controls, data_table)))

    # === Stats
    try:
        desc = df.describe().T
        tabs.append(("📊 Statistiques", pn.pane.DataFrame(desc, width=1000)))
    except Exception as e:
        logging.error(f"{table_name}: erreur stats — {str(e)}")

    # === Graphs
    plots = []
    if numeric_cols:
        for col in numeric_cols[:3]:
            fig = px.histogram(df, x=col, title=f"Distribution de {col}")
            plots.append(pn.pane.Plotly(fig, height=300))

    if len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
        plots.append(pn.pane.Plotly(fig, height=300))

    if plots:
        tabs.append(("📈 Graphiques", pn.Column(*plots)))

    return pn.Tabs(*tabs)

# === Create Panel App ===
def create_dashboard(tables, dataframes):
    template = pn.template.FastListTemplate(
        title="📊 Dashboard BigQuery Insights",
        sidebar=[pn.pane.Markdown("## Navigation"), *[pn.widgets.Button(name=t) for t in tables]],
        main=[],
        header_background="#0078D4",
    )

    welcome = pn.pane.Markdown(f"""
    # 🔍 Dashboard Automatique
    **Projet**: `{PROJECT_ID}`  
    **Dataset**: `{DATASET_ID}`  
    **Tables disponibles**:  
    """ + "\n".join([f"- {t}" for t in tables]), width=900)

    template.main.append(welcome)

    for t in tables:
        df = dataframes[t]
        insights = generate_insights(df, t)
        template.main.append(pn.Column(pn.pane.Markdown(f"# 📘 Table: `{t}`"), insights))

    return template

# === MAIN ===
def main():
    logging.info("Chargement des tables...")
    tables = get_tables()
    tables_data = {}

    for t in tables:
        try:
            df = load_data(t)
            if not df.empty:
                tables_data[t] = df
                logging.info(f"{t}: chargée")
            else:
                logging.warning(f"{t}: vide")
        except Exception as e:
            logging.error(f"{t}: erreur — {str(e)}")

    if tables_data:
        app = create_dashboard(list(tables_data.keys()), tables_data)
        pn.serve(app, port=5006, show=True)
    else:
        logging.error("Aucune table à afficher.")

if __name__ == "__main__":
    main()
