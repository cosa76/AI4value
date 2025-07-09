import subprocess

def run_script(script_name):
    print(f"🚀 Lancement de {script_name}...")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    
    print(f"📄 Sortie de {script_name} :")
    print(result.stdout)

    if result.returncode != 0:
        print(f"❌ Erreur dans {script_name} :")
        print(result.stderr)
        raise Exception(f"Le script {script_name} a échoué avec le code {result.returncode}")
    else:
        print(f"✅ {script_name} exécuté avec succès.\n")

# Lancement séquentiel
run_script("1_POC_read_file_folder_create_tables_v4.py")
run_script("2_POC_creation_modele_etoile_via_tables_bigquery_v2.py")
run_script("3_POC_generator_schema_etoile_or_flocons.py")
run_script("4_POC_recreation_vues_bigquery.py")
#run_script("5_POC_visualisation_view_data.py")
