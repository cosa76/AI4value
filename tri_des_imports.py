import ast
import os
import sys
import importlib.util

STANDARD_MODULES = sys.builtin_module_names

def is_standard_lib(module):
    """Vérifie si un module est dans la bibliothèque standard."""
    if module in STANDARD_MODULES:
        return True
    try:
        spec = importlib.util.find_spec(module)
        if spec and 'site-packages' not in (spec.origin or ''):
            return True
    except ModuleNotFoundError:
        pass
    return False

def extract_top_imports(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    import_lines = []
    other_lines = []
    inside_import_block = True

    for line in lines:
        stripped = line.strip()
        if inside_import_block and (stripped.startswith('import ') or stripped.startswith('from ')):
            import_lines.append(line)
        elif stripped == '' and inside_import_block:
            import_lines.append(line)
        else:
            inside_import_block = False
            other_lines.append(line)

    return import_lines, other_lines

def categorize_imports(import_lines):
    std, third_party, local = [], [], []

    for line in import_lines:
        if line.strip() == '' or line.strip().startswith('#'):
            continue
        try:
            root_module = line.split()[1].split('.')[0]
        except IndexError:
            continue
        if is_standard_lib(root_module):
            std.append(line)
        elif root_module in sys.modules or importlib.util.find_spec(root_module):
            third_party.append(line)
        else:
            local.append(line)

    return sorted(std), sorted(third_party), sorted(local)

def rewrite_file(file_path):
    import_lines, other_lines = extract_top_imports(file_path)
    std, third_party, local = categorize_imports(import_lines)

    with open(file_path, 'w', encoding='utf-8') as f:
        if std:
            f.writelines(std)
            f.write('\n')
        if third_party:
            f.writelines(third_party)
            f.write('\n')
        if local:
            f.writelines(local)
            f.write('\n')
        f.writelines(other_lines)

    print(f"Imports triés dans : {file_path}")

# --- UTILISATION ---

# Exemple : trie les imports d’un fichier script.py dans le même dossier
if __name__ == '__main__':
    file_to_process = '/Users/jndong/AI4value_good/Load_file_in_bigquery/1_POC_read_file_folder_create_tables_v5.py'  # remplace par le fichier à trier
    if os.path.isfile(file_to_process):
        rewrite_file(file_to_process)
    else:
        print(f"Fichier non trouvé : {file_to_process}")
