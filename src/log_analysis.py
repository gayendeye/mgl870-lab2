import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_bgl_logs(file_path, sample_size=1000):
    """Analyse les logs BGL et retourne des statistiques."""
    with open(file_path, 'r', encoding='latin-1') as f:
        logs = [next(f).strip() for _ in range(sample_size)]
    
    # Analyse des logs BGL
    bgl_pattern = r'^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$'
    parsed_logs = []
    
    for log in logs:
        match = re.match(bgl_pattern, log)
        if match:
            timestamp, date, node, component, message = match.groups()
            parsed_logs.append({
                'timestamp': timestamp,
                'date': date,
                'node': node,
                'component': component,
                'message': message
            })
    
    # Calcul des statistiques
    stats = {
        'total_logs': len(parsed_logs),
        'unique_nodes': len(set(log['node'] for log in parsed_logs)),
        'components': dict(Counter(log['component'] for log in parsed_logs))
    }
    
    return stats

def analyze_hdfs_logs(file_path, sample_size=1000):
    """Analyse les logs HDFS et retourne des statistiques."""
    with open(file_path, 'r', encoding='latin-1') as f:
        logs = [next(f).strip() for _ in range(sample_size)]
    
    # Analyse des logs HDFS
    hdfs_pattern = r'^(\d{6})\s+(\d{6})\s+(\d+)\s+(\w+)\s+([\w.$]+):\s+(.*)$'
    parsed_logs = []
    
    for log in logs:
        match = re.match(hdfs_pattern, log)
        if match:
            date, time, pid, level, clazz, message = match.groups()
            parsed_logs.append({
                'date': date,
                'time': time,
                'pid': pid,
                'level': level,
                'class': clazz,
                'message': message
            })
    
    # Calcul des statistiques
    stats = {
        'total_logs': len(parsed_logs),
        'log_levels': dict(Counter(log['level'] for log in parsed_logs)),
        'classes': dict(Counter(log['class'] for log in parsed_logs))
    }
    
    return stats

def main():
    # Chemins des fichiers de logs
    bgl_path = Path('../data/bgl/BGL.log')
    hdfs_path = Path('../data/hdfs/HDFS_2k.log')
    
    # Analyse des logs
    print("Analyse des logs BGL...")
    bgl_stats = analyze_bgl_logs(bgl_path)
    print("\nStatistiques BGL:")
    print(f"Total des logs analysés: {bgl_stats['total_logs']}")
    print(f"Nombre de nœuds uniques: {bgl_stats['unique_nodes']}")
    print("\nComposants les plus courants:")
    for comp, count in sorted(bgl_stats['components'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {comp}: {count} occurrences")
    
    print("\nAnalyse des logs HDFS...")
    hdfs_stats = analyze_hdfs_logs(hdfs_path)
    print("\nStatistiques HDFS:")
    print(f"Total des logs analysés: {hdfs_stats['total_logs']}")
    print("\nNiveaux de log:")
    for level, count in sorted(hdfs_stats['log_levels'].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {level}: {count} occurrences")
    
    print("\nClasses les plus courantes:")
    for clazz, count in sorted(hdfs_stats['classes'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {clazz}: {count} occurrences")

if __name__ == "__main__":
    main()