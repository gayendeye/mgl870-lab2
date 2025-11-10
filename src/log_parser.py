import os
import re
import pandas as pd
import hashlib
from collections import defaultdict
import numpy as np

class LogCluster:
    def __init__(self, log_template='', log_id=0):
        self.log_template = log_template
        self.ids = [log_id]
        self.idx = 0

class Node:
    def __init__(self, key='', depth=0):
        self.key = key
        self.depth = depth
        self.children = {}
        self.clusters = []

class Drain:
    def __init__(self, log_format, indir='./', outdir='./', depth=4, st=0.5, rex=[]):
        self.log_format = log_format
        self.indir = indir
        self.outdir = outdir
        self.depth = depth - 2
        self.st = st
        self.rex = rex
        self.root = Node()
        self.log_cluster_list = []
        self.df_log = None
        self.log_columns = ['Content', 'EventTemplate', 'EventId']

    def has_numbers(self, s):
        return any(char.isdigit() for char in s)

    def tree_search(self, root_node, tokens, depth=0):
        if depth >= len(tokens):
            return root_node.clusters

        token = tokens[depth]
        if token in root_node.children:
            return self.tree_search(root_node.children[token], tokens, depth + 1)
        else:
            for child in root_node.children.values():
                if self.has_numbers(child.key) and self.has_numbers(token):
                    return self.tree_search(child, tokens, depth + 1)
            return root_node.clusters

    def add_log_message(self, content, log_id):
        content = content.strip()
        for current_rex in self.rex:
            content = re.sub(current_rex, '<*>', content)

        tokens = content.split()
        if len(tokens) < self.depth:
            tokens = [''] * (self.depth - len(tokens)) + tokens

        tokens = tokens[:self.depth] + [tokens[-1]] if len(tokens) > self.depth else tokens
        clusters = self.tree_search(self.root, tokens)

        max_sim = -1
        max_cluster = None

        for cluster in clusters:
            template_tokens = cluster.log_template.split()
            if len(template_tokens) != len(tokens):
                continue

            sim = sum(1 for i in range(len(tokens)) if template_tokens[i] == tokens[i] or 
                     (self.has_numbers(template_tokens[i]) and self.has_numbers(tokens[i])))
            sim_ratio = sim / len(tokens)

            if sim_ratio > max_sim and sim_ratio >= self.st:
                max_sim = sim_ratio
                max_cluster = cluster

        if max_cluster is None:
            new_cluster = LogCluster(content, log_id)
            self.log_cluster_list.append(new_cluster)
            current_node = self.root
            for token in tokens:
                if token not in current_node.children:
                    new_node = Node(token, current_node.depth + 1)
                    current_node.children[token] = new_node
                current_node = current_node.children[token]
            current_node.clusters.append(new_cluster)
            return content
        else:
            template_tokens = max_cluster.log_template.split()
            new_template = []
            for i in range(len(tokens)):
                if template_tokens[i] == tokens[i]:
                    new_template.append(tokens[i])
                else:
                    new_template.append('<*>')
            new_template = ' '.join(new_template)
            max_cluster.log_template = new_template
            max_cluster.ids.append(log_id)
            return new_template

    def log_to_dataframe(self, log_file, max_lines=10000):
        log_messages = []
        print(f"Ouverture du fichier: {log_file}")
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                print(f"Fichier ouvert. Lecture des {max_lines} premières lignes...")
                for i, line in enumerate(f):
                    if i >= max_lines:
                        print(f"Limite de {max_lines} lignes atteinte.")
                        break
                    if i % 1000 == 0:
                        print(f"Traitement de la ligne {i}...")
                    log_messages.append({'LineId': i, 'Content': line.strip()})
                print(f"Lecture terminée. {len(log_messages)} lignes lues.")
            return pd.DataFrame(log_messages)
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {log_file}: {str(e)}")
            raise

    def parse(self, log_file, max_lines=10000):
        log_file_path = os.path.join(self.indir, log_file)
        print(f"Début du parsing de {log_file_path}")
        
        try:
            # Charger uniquement max_lines lignes
            self.df_log = self.log_to_dataframe(log_file_path, max_lines=max_lines)
            print(f"Fichier chargé avec succès. {len(self.df_log)} entrées à traiter.")
            
            templates = []
            total = len(self.df_log)
            
            # Traiter par lots pour économiser la mémoire
            batch_size = 1000
            for start_idx in range(0, total, batch_size):
                end_idx = min(start_idx + batch_size, total)
                print(f"Traitement des lignes {start_idx} à {end_idx-1}...")
                
                for idx in range(start_idx, end_idx):
                    row = self.df_log.iloc[idx]
                    template = self.add_log_message(row['Content'], idx)
                    templates.append(template)
                
        except Exception as e:
            print(f"Erreur lors du parsing: {str(e)}")
            import traceback
            traceback.print_exc()
        
        self.df_log['EventTemplate'] = templates
        self.df_log['EventId'] = self.df_log['EventTemplate'].map(
            lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[0:8]
        )
        
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        
        output_file = os.path.join(self.outdir, f'{os.path.splitext(log_file)[0]}_structured.csv')
        self.df_log.to_csv(output_file, index=False)
        print(f"Logs parsés enregistrés dans {output_file}")

def parse_logs(input_dir, output_dir, log_type):
    """
    Parse les fichiers de logs en utilisant l'algorithme Drain
    """
    # Configuration du parseur en fonction du type de log
    log_format = {
        'HDFS': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'BGL': '<Date> <Time> <Pid> <Level> <Component>: <Content>'
    }[log_type]
    
    # Mapping des noms de fichiers de log
    log_files = {
        'HDFS': 'HDFS_2k.log',
        'BGL': 'BGL.log'  # Nom corrigé du fichier BGL
    }
    
    # Création du parseur Drain
    parser = Drain(
        log_format=log_format,
        indir=input_dir,
        outdir=output_dir,
        depth=4,          # Profondeur de l'arbre de parsing
        st=0.5,           # Seuil de similarité
        rex=[r'\d+']     # Expressions régulières pour les motifs numériques
    )
    
    # Parsing du fichier de log
    input_file = log_files[log_type]
    print(f"Traitement de {input_file}...")
    parser.parse(input_file)

def main():
    # Dossiers de base
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # Création des dossiers de sortie
    os.makedirs(os.path.join(data_dir, 'hdfs_parsed'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'bgl_parsed'), exist_ok=True)
    
    # Parser les logs HDFS
    print("\n" + "="*50)
    print("TRAITEMENT DES LOGS HDFS")
    print("="*50)
    parse_logs(
        os.path.join(data_dir, 'hdfs'),
        os.path.join(data_dir, 'hdfs_parsed'),
        'HDFS'
    )
    
    # Parser les logs BGL
    print("\n" + "="*50)
    print("TRAITEMENT DES LOGS BGL")
    print("="*50)
    parse_logs(
        os.path.join(data_dir, 'bgl'),
        os.path.join(data_dir, 'bgl_parsed'),
        'BGL'
    )
    
    print("\nTraitement terminé avec succès !")

if __name__ == "__main__":
    main()