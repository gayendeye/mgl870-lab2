import os
import pandas as pd
from pathlib import Path
from log_parser import Drain

def setup_drain_parser(log_format, indir, outdir, log_file):
    """
    Configure et retourne un parseur Drain.
    
    Args:
        log_format (str): Format du log (spécifique à chaque type de log)
        indir (str): Répertoire d'entrée des logs
        outdir (str): Répertoire de sortie pour les résultats
        log_file (str): Nom du fichier de log à parser
        
    Returns:
        Drain: Instance du parseur configuré
    """
    # Configuration du parseur Drain
    parser = Drain(
        log_format=log_format,  # Format du log
        indir=indir,           # Répertoire d'entrée
        outdir=outdir,         # Répertoire de sortie
        depth=4,               # Profondeur de l'arbre
        st=0.5,                # Seuil de similarité
        rex=[r'\d+']          # Expressions régulières pour les nombres
    )
    return parser

def parse_bgl_logs():
    """Parse les logs BGL avec Drain et retourne un DataFrame des résultats."""
    # Format spécifique des logs BGL
    log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'
    indir = os.path.join('..', 'data', 'bgl')
    outdir = os.path.join('..', 'results', 'bgl_parsed')
    
    # Créer le dossier de sortie s'il n'existe pas
    Path(outdir).mkdir(parents=True, exist_ok=True)
    
    # Configurer et exécuter le parseur
    parser = setup_drain_parser(log_format, indir, outdir, 'BGL.log')
    parser.parse('BGL.log')
    
    # Lire et retourner les résultats
    output_file = os.path.join(outdir, 'BGL_structured.csv')
    if os.path.exists(output_file):
        return pd.read_csv(output_file)
    else:
        print(f"Fichier de sortie non trouvé : {output_file}")
        return pd.DataFrame()

def parse_hdfs_logs():
    """Parse les logs HDFS avec Drain et retourne un DataFrame des résultats."""
    # Format spécifique des logs HDFS
    log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
    indir = os.path.join('..', 'data', 'hdfs')
    outdir = os.path.join('..', 'results', 'hdfs_parsed')
    
    # Créer le dossier de sortie s'il n'existe pas
    Path(outdir).mkdir(parents=True, exist_ok=True)
    
    # Configurer et exécuter le parseur
    parser = setup_drain_parser(log_format, indir, outdir, 'HDFS_2k.log')
    parser.parse('HDFS_2k.log')
    
    # Lire et retourner les résultats
    output_file = os.path.join(outdir, 'HDFS_2k_structured.csv')
    if os.path.exists(output_file):
        return pd.read_csv(output_file)
    else:
        print(f"Fichier de sortie non trouvé : {output_file}")
        return pd.DataFrame()

def main():
    """Fonction principale pour tester le parsing des logs."""
    print("Démarrage du parsing des logs...")
    
    # Parser les logs BGL
    print("\n=== Parsing des logs BGL ===")
    try:
        bgl_df = parse_bgl_logs()
        print(f"✅ Logs BGL parsés avec succès : {len(bgl_df)} entrées")
        print("\nAperçu des données BGL parsées :")
        print(bgl_df.head())
        
        # Afficher les statistiques des templates
        print("\nStatistiques des templates BGL :")
        print(f"Nombre de templates uniques : {bgl_df['EventTemplate'].nunique()}")
        print("\nTemplates les plus courants :")
        print(bgl_df['EventTemplate'].value_counts().head())
        
    except Exception as e:
        print(f"❌ Erreur lors du parsing des logs BGL : {e}")
    
    # Parser les logs HDFS
    print("\n=== Parsing des logs HDFS ===")
    try:
        hdfs_df = parse_hdfs_logs()
        print(f"✅ Logs HDFS parsés avec succès : {len(hdfs_df)} entrées")
        print("\nAperçu des données HDFS parsées :")
        print(hdfs_df.head())
        
        # Afficher les statistiques des templates
        print("\nStatistiques des templates HDFS :")
        print(f"Nombre de templates uniques : {hdfs_df['EventTemplate'].nunique()}")
        print("\nTemplates les plus courants :")
        print(hdfs_df['EventTemplate'].value_counts().head())
        
    except Exception as e:
        print(f"❌ Erreur lors du parsing des logs HDFS : {e}")

if __name__ == "__main__":
    main()