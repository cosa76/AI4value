#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour convertir automatiquement tous les fichiers XML d'un dossier en CSV
Reconnaissance automatique des balises et structures XML
"""

import xml.etree.ElementTree as ET
import csv
import argparse
import os
import glob
from typing import List, Dict, Any, Tuple, Set
from collections import Counter, defaultdict

class XMLStructureAnalyzer:
    """Classe pour analyser automatiquement la structure des fichiers XML"""
    
    def __init__(self):
        self.tag_patterns = defaultdict(int)
        self.element_depths = defaultdict(list)
        self.repeated_elements = defaultdict(int)
        self.sample_data = {}
    
    def analyze_element(self, element: ET.Element, depth: int = 0, parent_path: str = ""):
        """Analyse récursivement un élément XML"""
        current_path = f"{parent_path}/{element.tag}" if parent_path else element.tag
        
        # Compter les occurrences de chaque balise
        self.tag_patterns[element.tag] += 1
        self.element_depths[element.tag].append(depth)
        
        # Sauvegarder un échantillon de données
        if element.tag not in self.sample_data:
            sample = {}
            if element.text and element.text.strip():
                sample['text'] = element.text.strip()
            if element.attrib:
                sample['attributes'] = element.attrib
            sample['children'] = [child.tag for child in element]
            self.sample_data[element.tag] = sample
        
        # Analyser les enfants
        for child in element:
            self.analyze_element(child, depth + 1, current_path)
    
    def get_main_record_element(self) -> str:
        """Détermine automatiquement l'élément principal qui représente un enregistrement"""
        # Chercher les éléments qui apparaissent plusieurs fois au même niveau
        candidates = {}
        
        for tag, count in self.tag_patterns.items():
            if count > 1 and tag not in ['root', 'data', 'document']:
                avg_depth = sum(self.element_depths[tag]) / len(self.element_depths[tag])
                candidates[tag] = {'count': count, 'avg_depth': avg_depth}
        
        if not candidates:
            # Si pas de répétitions, prendre l'élément avec le plus d'enfants
            best_tag = max(self.sample_data.keys(), 
                          key=lambda x: len(self.sample_data[x].get('children', [])))
            return best_tag
        
        # Privilégier l'élément avec le plus d'occurrences et une profondeur raisonnable
        best_tag = max(candidates.keys(), 
                      key=lambda x: candidates[x]['count'] * (1 / (candidates[x]['avg_depth'] + 1)))
        
        return best_tag
    
    def print_analysis(self):
        """Affiche l'analyse de la structure XML"""
        print("\n=== ANALYSE DE LA STRUCTURE XML ===")
        print(f"Balises trouvées : {len(self.tag_patterns)}")
        
        print("\nOccurrences par balise :")
        for tag, count in sorted(self.tag_patterns.items(), key=lambda x: x[1], reverse=True):
            avg_depth = sum(self.element_depths[tag]) / len(self.element_depths[tag])
            print(f"  {tag}: {count} fois (profondeur moyenne: {avg_depth:.1f})")
        
        main_element = self.get_main_record_element()
        print(f"\nÉlément principal détecté : '{main_element}'")
        
        if main_element in self.sample_data:
            sample = self.sample_data[main_element]
            print(f"Structure de '{main_element}' :")
            if sample.get('attributes'):
                print(f"  Attributs : {list(sample['attributes'].keys())}")
            if sample.get('children'):
                print(f"  Éléments enfants : {sample['children']}")
            if sample.get('text'):
                print(f"  Contient du texte : Oui")

def analyze_xml_file(xml_file: str) -> XMLStructureAnalyzer:
    """Analyse un fichier XML et retourne l'analyseur"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        analyzer = XMLStructureAnalyzer()
        analyzer.analyze_element(root)
        
        return analyzer
    except Exception as e:
        print(f"Erreur lors de l'analyse de {xml_file}: {e}")
        return None

def flatten_element(element: ET.Element, parent_key: str = '', max_depth: int = 10) -> Dict[str, str]:
    """
    Aplatit récursivement un élément XML en dictionnaire avec contrôle de profondeur
    """
    if max_depth <= 0:
        return {}
    
    result = {}
    
    # Ajouter les attributs de l'élément
    for attr_name, attr_value in element.attrib.items():
        key = f"{parent_key}@{attr_name}" if parent_key else f"@{attr_name}"
        result[key] = str(attr_value)
    
    # Ajouter le texte de l'élément s'il existe et n'a pas d'enfants
    if element.text and element.text.strip() and len(element) == 0:
        key = parent_key if parent_key else 'text'
        result[key] = element.text.strip()
    
    # Traiter les éléments enfants
    child_counts = Counter(child.tag for child in element)
    child_indices = defaultdict(int)
    
    for child in element:
        if child_counts[child.tag] > 1:
            # Élément répété - ajouter un index
            child_key = f"{parent_key}.{child.tag}[{child_indices[child.tag]}]" if parent_key else f"{child.tag}[{child_indices[child.tag]}]"
            child_indices[child.tag] += 1
        else:
            # Élément unique
            child_key = f"{parent_key}.{child.tag}" if parent_key else child.tag
        
        # Récursion avec contrôle de profondeur
        if len(child) > 0:
            child_data = flatten_element(child, child_key, max_depth - 1)
            result.update(child_data)
        else:
            # Feuille avec texte
            if child.text and child.text.strip():
                result[child_key] = child.text.strip()
            
            # Ajouter les attributs de l'enfant
            for attr_name, attr_value in child.attrib.items():
                attr_key = f"{child_key}@{attr_name}"
                result[attr_key] = str(attr_value)
    
    return result

def convert_xml_to_csv(xml_file: str, csv_file: str, main_element: str = None, max_depth: int = 10) -> bool:
    """
    Convertit un fichier XML en CSV avec détection automatique
    """
    try:
        print(f"\nTraitement de : {xml_file}")
        
        # Analyser la structure si l'élément principal n'est pas spécifié
        if not main_element:
            analyzer = analyze_xml_file(xml_file)
            if not analyzer:
                return False
            main_element = analyzer.get_main_record_element()
            print(f"Élément principal détecté : '{main_element}'")
        
        # Parser le fichier XML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Trouver tous les éléments du type principal
        if main_element == root.tag:
            elements = [root]
        else:
            elements = root.findall(f".//{main_element}")
        
        if not elements:
            print(f"Aucun élément '{main_element}' trouvé dans {xml_file}")
            return False
        
        print(f"Nombre d'enregistrements trouvés : {len(elements)}")
        
        # Collecter toutes les données
        all_data = []
        all_keys = set()
        
        for i, element in enumerate(elements):
            try:
                row_data = flatten_element(element, max_depth=max_depth)
                if row_data:  # Ignorer les éléments vides
                    all_data.append(row_data)
                    all_keys.update(row_data.keys())
            except Exception as e:
                print(f"Erreur lors du traitement de l'élément {i+1}: {e}")
                continue
        
        if not all_data:
            print(f"Aucune donnée valide trouvée dans {xml_file}")
            return False
        
        # Trier les clés pour un ordre logique
        fieldnames = sorted(all_keys, key=lambda x: (
            0 if x.startswith('@') else 1,  # Attributs en premier
            x.count('.'),  # Puis par profondeur
            x  # Puis alphabétique
        ))
        
        # Écrire le fichier CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row_data in all_data:
                # Remplir les valeurs manquantes avec des chaînes vides
                complete_row = {key: row_data.get(key, '') for key in fieldnames}
                writer.writerow(complete_row)
        
        print(f"✓ Conversion réussie : {len(all_data)} enregistrements → {csv_file}")
        print(f"  Colonnes créées : {len(fieldnames)}")
        return True
        
    except ET.ParseError as e:
        print(f"✗ Erreur de parsing XML dans {xml_file}: {e}")
        return False
    except Exception as e:
        print(f"✗ Erreur lors de la conversion de {xml_file}: {e}")
        return False

def process_folder(folder_path: str, output_dir: str = None, element: str = None, analyze_first: bool = True):
    """
    Traite tous les fichiers XML d'un dossier
    """
    if not os.path.exists(folder_path):
        print(f"Erreur : Le dossier {folder_path} n'existe pas")
        return
    
    # Trouver tous les fichiers XML
    xml_files = []
    for pattern in ['*.xml', '*.XML']:
        xml_files.extend(glob.glob(os.path.join(folder_path, pattern)))
        xml_files.extend(glob.glob(os.path.join(folder_path, '**', pattern), recursive=True))
    
    if not xml_files:
        print(f"Aucun fichier XML trouvé dans {folder_path}")
        return
    
    print(f"Fichiers XML trouvés : {len(xml_files)}")
    for xml_file in xml_files:
        print(f"  - {os.path.relpath(xml_file, folder_path)}")
    
    # Définir le dossier de sortie
    if not output_dir:
        output_dir = os.path.join(folder_path, 'csv_output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyse globale si demandée
    if analyze_first and not element:
        print("\n=== ANALYSE GLOBALE DES STRUCTURES ===")
        global_analyzer = XMLStructureAnalyzer()
        
        for xml_file in xml_files[:3]:  # Analyser les 3 premiers fichiers
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                global_analyzer.analyze_element(root)
            except Exception as e:
                print(f"Erreur lors de l'analyse de {xml_file}: {e}")
        
        if global_analyzer.tag_patterns:
            global_analyzer.print_analysis()
            element = global_analyzer.get_main_record_element()
    
    # Traiter chaque fichier
    print(f"\n=== CONVERSION DES FICHIERS ===")
    success_count = 0
    
    for xml_file in xml_files:
        try:
            # Générer le nom du fichier CSV
            base_name = os.path.splitext(os.path.basename(xml_file))[0]
            csv_file = os.path.join(output_dir, f"{base_name}.csv")
            
            # Convertir
            if convert_xml_to_csv(xml_file, csv_file, element):
                success_count += 1
                
        except Exception as e:
            print(f"✗ Erreur lors du traitement de {xml_file}: {e}")
    
    print(f"\n=== RÉSUMÉ ===")
    print(f"Fichiers traités avec succès : {success_count}/{len(xml_files)}")
    print(f"Fichiers CSV générés dans : {output_dir}")

def main():
    """Fonction principale avec interface en ligne de commande"""
    parser = argparse.ArgumentParser(description='Convertit automatiquement tous les fichiers XML d\'un dossier en CSV')
    parser.add_argument('folder_path', help='Dossier contenant les fichiers XML')
    parser.add_argument('--output', '-o', help='Dossier de sortie pour les fichiers CSV')
    parser.add_argument('--element', '-e', help='Nom de l\'élément principal (détection auto si non spécifié)')
    parser.add_argument('--no-analyze', action='store_true', help='Désactiver l\'analyse préalable')
    parser.add_argument('--max-depth', type=int, default=10, help='Profondeur maximale d\'aplatissement (défaut: 10)')
    parser.add_argument('--recursive', '-r', action='store_true', help='Recherche récursive dans les sous-dossiers')
    
    args = parser.parse_args()
    
    print("=== CONVERTISSEUR XML VERS CSV - VERSION DOSSIER ===")
    print(f"Dossier source : {args.folder_path}")
    
    # Traiter le dossier
    process_folder(
        folder_path=args.folder_path,
        output_dir=args.output,
        element=args.element,
        analyze_first=not args.no_analyze
    )

def create_test_files():
    """Crée des fichiers XML d'exemple pour tester le script"""
    test_dir = "test_xml_files"
    os.makedirs(test_dir, exist_ok=True)
    
    # Fichier 1 : Liste de personnes
    xml1 = """<?xml version="1.0" encoding="UTF-8"?>
<personnes>
    <personne id="1" actif="true">
        <nom>Dupont</nom>
        <prenom>Jean</prenom>
        <age>30</age>
        <contact>
            <email>jean.dupont@email.com</email>
            <telephone>0123456789</telephone>
        </contact>
    </personne>
    <personne id="2" actif="false">
        <nom>Martin</nom>
        <prenom>Marie</prenom>
        <age>25</age>
        <contact>
            <email>marie.martin@email.com</email>
            <telephone>0987654321</telephone>
        </contact>
    </personne>
</personnes>"""
    
    # Fichier 2 : Catalogue de produits
    xml2 = """<?xml version="1.0" encoding="UTF-8"?>
<catalogue>
    <produit ref="P001">
        <nom>Ordinateur portable</nom>
        <prix devise="EUR">999.99</prix>
        <stock>15</stock>
        <categories>
            <categorie>Informatique</categorie>
            <categorie>Bureautique</categorie>
        </categories>
    </produit>
    <produit ref="P002">
        <nom>Souris sans fil</nom>
        <prix devise="EUR">29.99</prix>
        <stock>50</stock>
        <categories>
            <categorie>Informatique</categorie>
            <categorie>Accessoires</categorie>
        </categories>
    </produit>
</catalogue>"""
    
    with open(os.path.join(test_dir, "personnes.xml"), 'w', encoding='utf-8') as f:
        f.write(xml1)
    
    with open(os.path.join(test_dir, "produits.xml"), 'w', encoding='utf-8') as f:
        f.write(xml2)
    
    print(f"Fichiers de test créés dans le dossier : {test_dir}")
    return test_dir

if __name__ == "__main__":
    import sys
    
    # ============================================
    # 📁 CONFIGURATION - MODIFIEZ CES CHEMINS
    # ============================================
    
    # Chemin vers le dossier contenant vos fichiers XML
    FOLDER_PATH = "/Users/jndong/xml_file/"  # ⬅️ MODIFIEZ CE CHEMIN
    
    # Dossier de sortie pour les fichiers CSV (optionnel)
    OUTPUT_DIR = None  # None = créera automatiquement un dossier 'csv_output'
    
    # Élément principal à traiter (None = détection automatique)
    MAIN_ELEMENT = None  # Ex: "produit", "personne", "record", etc.
    
    # Options avancées
    ANALYZE_FIRST = True      # Analyser la structure avant conversion
    MAX_DEPTH = 10           # Profondeur maximale d'aplatissement
    RECURSIVE_SEARCH = True   # Chercher dans les sous-dossiers
    
    # ============================================
    # 🚀 EXÉCUTION AUTOMATIQUE
    # ============================================
    
    print("=== CONVERTISSEUR XML VERS CSV - MODE AUTOMATIQUE ===")
    
    # Vérifier si le dossier existe
    if os.path.exists(FOLDER_PATH):
        print(f"📁 Dossier source : {FOLDER_PATH}")
        print(f"📤 Dossier sortie : {OUTPUT_DIR or 'Auto (csv_output)'}")
        print(f"🔍 Élément principal : {MAIN_ELEMENT or 'Détection automatique'}")
        print(f"📊 Analyse préalable : {'Oui' if ANALYZE_FIRST else 'Non'}")
        print(f"🔄 Recherche récursive : {'Oui' if RECURSIVE_SEARCH else 'Non'}")
        print("-" * 60)
        
        # Traitement automatique
        process_folder(
            folder_path=FOLDER_PATH,
            output_dir=OUTPUT_DIR,
            element=MAIN_ELEMENT,
            analyze_first=ANALYZE_FIRST
        )
        
        print("\n✅ Traitement terminé !")
        
    else:
        print(f"❌ Erreur : Le dossier '{FOLDER_PATH}' n'existe pas.")
        print("\n🛠️  Solutions :")
        print("1. Modifiez la variable FOLDER_PATH dans le script")
        print("2. Créez des fichiers de test automatiquement")
        
        response = input("\nVoulez-vous créer des fichiers de test ? (o/n) : ").lower().strip()
        if response in ['o', 'oui', 'y', 'yes']:
            print("\n📝 Création de fichiers de test...")
            test_dir = create_test_files()
            print(f"\n🚀 Traitement du dossier de test : {test_dir}")
            process_folder(test_dir, analyze_first=True)
            print(f"\n✅ Test terminé ! Vérifiez le dossier : {test_dir}/csv_output")
            print(f"\n💡 Pour vos propres fichiers, modifiez FOLDER_PATH = '{test_dir}' dans le script")
        else:
            print("\n📋 Pour utiliser en ligne de commande :")
            print(f"python {sys.argv[0]} /chemin/vers/votre/dossier")
    
    # Possibilité d'utiliser aussi en ligne de commande
    if len(sys.argv) > 1:
        print("\n" + "="*60)
        print("🖥️  MODE LIGNE DE COMMANDE DÉTECTÉ")
        print("="*60)
        main()