import os
import base64
import io
from PIL import Image
from flask import Flask, send_from_directory
import dash
from dash import html, Input, Output

# === Définir ici le chemin du dossier à visualiser ===
folder_path = r"/Users/jndong/AI4value_good/visualisations_star_order_items"  # ← MODIFIE CETTE LIGNE AVEC TON DOSSIER

# Vérification que le dossier existe
if not os.path.isdir(folder_path):
    raise NotADirectoryError(f"Le dossier {folder_path} n'existe pas ou n'est pas valide.")

# === Initialisation de l'app Dash avec serveur Flask personnalisé ===
server = Flask(__name__)
app = dash.Dash(__name__, server=server)

# === Route Flask pour servir les fichiers statiques (HTML, JS, CSS...) ===
@server.route('/files/<path:filename>')
def serve_file(filename):
    return send_from_directory(folder_path, filename)

# === Fonctions utilitaires ===
def list_files(folder):
    return sorted([f for f in os.listdir(folder) if f.endswith(('.png', '.html'))])

def get_file_content(path):
    if path.endswith('.png'):
        return display_image(path)
    elif path.endswith('.html'):
        return display_html(path)
    return html.Div("Type de fichier non supporté")

def display_image(path):
    try:
        img = Image.open(path)
        img.thumbnail((800, 600))
        buffer = io.BytesIO()
        img.save(buffer, format=img.format)
        encoded_img = base64.b64encode(buffer.getvalue()).decode()
        return html.Img(src=f"data:image/png;base64,{encoded_img}", style={"max-width": "100%"})
    except Exception as e:
        return html.Div(f"Erreur lors du chargement de l'image : {e}")

def display_html(path):
    rel_path = os.path.relpath(path, folder_path)
    iframe_src = f'/files/{rel_path}'
    return html.Iframe(
        src=iframe_src,
        style={'width': '100%', 'height': '600px', 'border': 'none', 'marginTop': '20px'}
    )

# === Layout principal ===
app.layout = html.Div([
    html.H1("Visualiseur de Fichiers PNG et HTML", style={'textAlign': 'center'}),

    html.Div(id='content-display')
])

# === Callback pour charger les fichiers au lancement ===
@app.callback(
    Output('content-display', 'children'),
    Input('content-display', 'id')  # Se déclenche dès que l'élément est chargé
)
def load_files(_):
    files = list_files(folder_path)

    if not files:
        return html.Div("Aucun fichier .png ou .html trouvé.", style={'textAlign': 'center'})

    elements = []
    for file in files:
        full_path = os.path.join(folder_path, file)
        elements.append(html.H3(file))
        elements.append(get_file_content(full_path))

    return elements

# === Démarrage de l'application ===
if __name__ == '__main__':
    app.run(debug=True)