import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_parsed_logs(file_path):
    """Charge les logs parsés depuis un fichier CSV"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier {file_path}: {e}")
        return None

def analyze_logs(df, log_type):
    """Analyse les logs et génère des statistiques"""
    print(f"\nAnalyse des logs {log_type}")
    print("=" * 50)
    
    if df is None or df.empty:
        print("Aucune donnée à analyser")
        return {}
    
    # Afficher les premières lignes
    print("\nAperçu des données :")
    print(df.head())
    
    # Statistiques de base
    print("\nStatistiques descriptives :")
    print(df.describe(include='all'))
    
    # Nombre d'événements uniques
    if 'EventId' in df.columns:
        unique_events = df['EventId'].nunique()
        print(f"\nNombre d'événements uniques : {unique_events}")
        
        # Distribution des événements
        event_distribution = df['EventId'].value_counts()
        print("\nTop 10 des événements les plus fréquents :")
        print(event_distribution.head(10))
    
    # Vérifier les valeurs manquantes
    print("\nValeurs manquantes par colonne :")
    print(df.isnull().sum())
    
    return {
        'total_entries': len(df),
        'unique_events': unique_events if 'EventId' in df.columns else 0,
        'event_distribution': event_distribution.to_dict() if 'EventId' in df.columns else {}
    }

def plot_event_distribution(df, log_type, top_n=20):
    """Affiche la distribution des événements"""
    if df is None or 'EventId' not in df.columns:
        print("Données non valides pour la génération du graphique")
        return None
        
    plt.figure(figsize=(12, 8))
    event_counts = df['EventId'].value_counts().head(top_n)
    
    # Créer un graphique à barres horizontales
    ax = sns.barplot(x=event_counts.values, y=event_counts.index, palette="viridis")
    
    # Ajouter des étiquettes et un titre
    plt.title(f'Top {top_n} des événements les plus fréquents - {log_type}', pad=20)
    plt.xlabel("Nombre d'occurrences")
    plt.ylabel("ID d'événement")
    
    # Ajuster les marges
    plt.tight_layout()
    
    # Créer le dossier de sortie s'il n'existe pas
    output_dir = os.path.join("..", "results", "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder le graphique
    plot_path = os.path.join(output_dir, f"{log_type.lower()}_event_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nGraphique sauvegardé sous : {plot_path}")
    return plot_path

if __name__ == "__main__":
    # Chemins des fichiers de logs parsés
    data_dir = os.path.join("..", "data")
    
    # Analyser les logs HDFS
    print("\n" + "="*50)
    print("ANALYSE DES LOGS HDFS")
    print("="*50)
    hdfs_df = load_parsed_logs(os.path.join(data_dir, "hdfs_parsed", "HDFS_2k_structured.csv"))
    hdfs_stats = analyze_logs(hdfs_df, "HDFS")
    if hdfs_df is not None and not hdfs_df.empty:
        hdfs_plot = plot_event_distribution(hdfs_df, "HDFS")
    
    # Analyser les logs BGL
    print("\n" + "="*50)
    print("ANALYSE DES LOGS BGL")
    print("="*50)
    bgl_df = load_parsed_logs(os.path.join(data_dir, "bgl_parsed", "BGL_structured.csv"))
    bgl_stats = analyze_logs(bgl_df, "BGL")
    if bgl_df is not None and not bgl_df.empty:
        bgl_plot = plot_event_distribution(bgl_df, "BGL")
    
    print("\nAnalyse terminée. Les graphiques ont été sauvegardés dans le dossier results/figures/")