import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score, 
    confusion_matrix, ConfusionMatrixDisplay, f1_score,
    precision_recall_curve, average_precision_score, roc_curve, auc,
    make_scorer, classification_report
)
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os
import json
from pathlib import Path
from collections import Counter
from time import time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Configuration des paramètres d'affichage
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
plt.style.use('seaborn-v0_8')

def load_data(log_type='bgl'):
    """Charge les données préparées."""
    data_dir = os.path.join('..', 'results', 'features')
    X = np.load(os.path.join(data_dir, f'X_{log_type}.npy'))
    y = np.load(os.path.join(data_dir, f'y_{log_type}.npy'))
    
    # Vérification des étiquettes
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"Classes trouvées : {dict(zip(unique_classes, counts))}")
    print(f"Nombre d'échantillons : {len(y)}")
    
    return X, y

def remove_low_variance_features(X, threshold=0.01):
    """Élimine les caractéristiques à faible variance."""
    selector = VarianceThreshold(threshold=threshold)
    X_reduced = selector.fit_transform(X)
    return X_reduced, selector

def evaluate_model(y_true, y_pred, y_proba, model_name, log_type):
    """Évalue un modèle et retourne les métriques."""
    if np.isnan(y_proba).any():
        print("Avertissement: Des valeurs NaN détectées dans les probabilités prédites. Remplacement par 0.5")
        y_proba = np.nan_to_num(y_proba, nan=0.5)
    
    if np.isinf(y_proba).any():
        print("Avertissement: Des valeurs infinies détectées dans les probabilités prédites. Remplacement par 0 ou 1")
        y_proba = np.clip(y_proba, -1e10, 1e10)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-10)
    
    metrics = {}
    
    try:
        metrics.update({
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        })
    except Exception as e:
        print(f"Erreur lors du calcul des métriques de classification: {e}")
        metrics.update({'precision': 0, 'recall': 0, 'f1': 0})
    
    try:
        if len(np.unique(y_true)) > 1:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
            metrics['ap'] = average_precision_score(y_true, y_proba)
        else:
            print("Avertissement: Une seule classe détectée, AUC et AP non calculables")
            metrics.update({'auc': 0.5, 'ap': 0.5})
    except Exception as e:
        print(f"Erreur lors du calcul des métriques de probabilité: {e}")
        metrics.update({'auc': 0.5, 'ap': 0.5})
    
    print(f"\nRésultats pour {model_name} - {log_type.upper()}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return metrics

def plot_roc_curve(y_true, y_proba, model_name, log_type, save_dir):
    """Trace et sauvegarde la courbe ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Courbe ROC - {model_name} - {log_type.upper()}')
    plt.legend(loc="lower right")
    
    # Sauvegarde de la figure
    plt.savefig(os.path.join(save_dir, f'roc_curve_{model_name}_{log_type}.png'))
    plt.close()

def train_models(X, y, log_type):
    """Entraîne et évalue les modèles avec gestion du déséquilibre de classes."""
    # Vérification des classes
    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(unique_classes)
    min_samples = min(class_counts) if n_classes > 1 else 0
    
    # Pour BGL, on utilise les modèles supervisés avec gestion du déséquilibre
    if log_type == 'bgl' and n_classes > 1 and min_samples >= 5:
        print(f"\nUtilisation de modèles supervisés avec gestion du déséquilibre...")
        
        # Division stratifiée pour conserver la proportion d'anomalies
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Définition des modèles avec gestion du déséquilibre
        models = {
            'logistic_regression': {
                'model': LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42,
                    n_jobs=-1
                ),
                'params': {
                    'C': [0.1, 1, 10],
                    'solver': ['lbfgs', 'liblinear']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    class_weight='balanced_subsample',
                    random_state=42,
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            }
        }
    else:
        # Pour HDFS ou si très peu d'anomalies, on utilise des méthodes non supervisées
        print(f"\nTrop peu d'anomalies détectées ({min_samples} échantillons), utilisation de méthodes non supervisées...")
        
        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Modèles non supervisés
        models = {
            'isolation_forest': {
                'model': IsolationForest(
                    random_state=42, 
                    contamination=min(0.1, max(0.01, (y == 1).mean())),
                    n_estimators=200, 
                    n_jobs=-1,
                    verbose=1
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'max_samples': ['auto', 0.5, 0.8],
                    'contamination': [min(0.1, max(0.01, (y == 1).mean()))]
                }
            },
            'one_class_svm': {
                'model': OneClassSVM(
                    nu=min(0.5, max(0.01, (y == 1).mean() * 2)),
                    kernel='rbf',
                    gamma='scale',
                    max_iter=1000
                ),
                'params': {
                    'nu': [0.01, 0.05, 0.1],
                    'kernel': ['rbf'],
                    'gamma': ['scale', 'auto']
                }
            }
        }
    
    # Détection du type de modèles (supervisé ou non supervisé)
    is_unsupervised = log_type == 'hdfs' or n_classes < 2 or min_samples < 5
    
    # Entraînement et évaluation
    results = {}
    vis_dir = os.path.join('..', 'results', 'visualizations')
    model_dir = os.path.join('..', 'results', 'models')
    Path(vis_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    for name, model_info in models.items():
        print(f"\n=== Entraînement du modèle {name} ===")
        
        try:
            if is_unsupervised:
                # Entraînement des modèles non supervisés
                model = model_info['model']
                param_grid = model_info['params']
                
                def custom_scorer(estimator, X, y_true=None):
                    if hasattr(estimator, 'decision_function'):
                        y_scores = -estimator.score_samples(X)
                    else:
                        y_scores = -estimator.decision_function(X)
                    
                    if y_true is not None and len(np.unique(y_true)) > 1:
                        return roc_auc_score(y_true, y_scores)
                    return 0.5
                
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    scoring=custom_scorer,
                    n_jobs=-1,
                    verbose=1
                )
                
                # Entraînement
                start_time = time()
                grid_search.fit(X_train)
                training_time = time() - start_time
                best_model = grid_search.best_estimator_
                
                # Prédictions pour les modèles non supervisés
                if hasattr(best_model, 'decision_function'):
                    y_scores = -best_model.score_samples(X_test)
                    threshold = np.percentile(y_scores, 85)
                    y_pred = (y_scores > threshold).astype(int)
                    y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-10)
                else:
                    y_scores = best_model.decision_function(X_test)
                    threshold = np.percentile(y_scores, 10)
                    y_pred = (y_scores < threshold).astype(int)
                    y_proba = 1 - (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-10)
            else:
                # Entraînement des modèles supervisés
                cv = StratifiedKFold(n_splits=min(5, np.min(np.bincount(y_train.astype(int)))), 
                                   shuffle=True, random_state=42)
                
                grid_search = GridSearchCV(
                    estimator=model_info['model'],
                    param_grid=model_info['params'],
                    cv=cv,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=1,
                    error_score='raise'
                )
                
                # Entraînement
                start_time = time()
                grid_search.fit(X_train, y_train)
                training_time = time() - start_time
                best_model = grid_search.best_estimator_
                
                # Prédictions
                y_pred = best_model.predict(X_test)
                
                # Probabilités de prédiction
                if hasattr(best_model, 'predict_proba'):
                    y_proba = best_model.predict_proba(X_test)[:, 1]
                else:
                    y_scores = best_model.decision_function(X_test)
                    y_proba = 1 / (1 + np.exp(-y_scores))
            
            # Gestion des cas où toutes les prédictions sont identiques
            if len(np.unique(y_pred)) == 1:
                print(f"Avertissement: Toutes les prédictions sont {y_pred[0]}. Ajustement des probabilités.")
                y_proba = np.linspace(0.1, 0.9, len(y_pred))
                
            # Conversion en tableaux numpy si nécessaire
            y_pred = np.array(y_pred).flatten()
            y_proba = np.array(y_proba).flatten()
            
            # Vérification finale des valeurs NaN/Inf
            if np.isnan(y_proba).any() or np.isinf(y_proba).any():
                print("Avertissement: Correction des valeurs NaN/Inf dans les probabilités prédites")
                y_proba = np.nan_to_num(y_proba, nan=0.5, posinf=1.0, neginf=0.0)
                y_proba = np.clip(y_proba, 0, 1)
            
            # Métriques d'évaluation
            metrics = evaluate_model(y_test, y_pred, y_proba, name, log_type)
            metrics['training_time'] = training_time
            
        except Exception as e:
            print(f"\nErreur lors de l'entraînement du modèle {name}:")
            print(f"Type d'erreur: {type(e).__name__}")
            print(f"Message: {str(e)}")
            print("\nPoursuite avec le modèle suivant...\n")
            continue
        
        # Rapport de classification
        print("\nRapport de classification détaillé:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Matrice de confusion - {name} - {log_type.upper()}\nF1: {metrics["f1"]:.3f}, AUC: {metrics["auc"]:.3f}')
        plt.savefig(os.path.join(vis_dir, f'confusion_matrix_{name}_{log_type}.png'), bbox_inches='tight')
        plt.close()
        
        # Sauvegarde du modèle
        model_path = os.path.join(model_dir, f'{name}_{log_type}.joblib')
        joblib.dump(best_model, model_path, compress=3)
        print(f"Modèle sauvegardé dans {model_path}")

        # Sauvegarde des métriques
        metrics_path = os.path.join(model_dir, f'metrics_{name}_{log_type}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Ajout du modèle et des métriques aux résultats
        results[name] = {
            'model': best_model,
            'metrics': metrics,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'best_params': grid_search.best_params_
        }
    
    return results, X_test, y_test

def main():
    # Configuration des dossiers de sortie
    for folder in ['models', 'visualizations', 'optimization', 'error_analysis']:
        Path(os.path.join('..', 'results', folder)).mkdir(parents=True, exist_ok=True)
    
    # Traitement pour chaque type de logs
    for log_type in ['bgl', 'hdfs']:
        print(f"\n{'='*50}")
        print(f"=== TRAITEMENT DES LOGS {log_type.upper()} ===")
        print(f"{'='*50}")
        
        # 1. Chargement des données
        print("\n1. Chargement des données...")
        X, y = load_data(log_type)
        
        # 2. Prétraitement des caractéristiques
        print("\n2. Prétraitement des caractéristiques...")
        X_clean, _ = remove_low_variance_features(X)
        print(f"   - Nombre de caractéristiques initial: {X.shape[1]}")
        print(f"   - Nombre après suppression de la faible variance: {X_clean.shape[1]}")
        
        # 3. Entraînement et évaluation des modèles
        print("\n3. Entraînement et évaluation des modèles...")
        results, X_test, y_test = train_models(X_clean, y, log_type)
        
        # 4. Affichage des résultats
        print("\n4. Résumé des performances:")
        for model_name, result in results.items():
            print(f"\nModèle: {model_name}")
            print("-" * 50)
            for metric, value in result['metrics'].items():
                if metric != 'confusion_matrix':
                    print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()