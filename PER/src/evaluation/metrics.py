"""
Metrics Calculator - Calcul des métriques d'évaluation
=====================================================

Ce module fournit des outils pour calculer et agréger les métriques
d'évaluation des modèles ML et Transformers.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import json
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculateur de métriques pour l'évaluation des modèles.
    
    Calcule:
    - Métriques de classification (accuracy, precision, recall, F1)
    - Métriques de performance (temps, mémoire)
    - Courbes ROC et PR
    """
    
    def __init__(self):
        """Initialise le calculateur de métriques."""
        self.results_history = []
    
    def calculate_classification_metrics(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_pred: Union[np.ndarray, List[int]],
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calcule les métriques de classification.
        
        Args:
            y_true: Labels réels
            y_pred: Labels prédits
            y_proba: Probabilités (optionnel, pour AUC)
            
        Returns:
            Dictionnaire des métriques
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        }
        
        # Calculer l'AUC si les probabilités sont fournies
        if y_proba is not None:
            try:
                # Probabilité de la classe positive
                if len(y_proba.shape) > 1:
                    y_proba_positive = y_proba[:, 1]
                else:
                    y_proba_positive = y_proba
                
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba_positive)
                metrics['average_precision'] = average_precision_score(y_true, y_proba_positive)
            except Exception as e:
                logger.warning(f"Impossible de calculer AUC: {e}")
        
        return metrics
    
    def calculate_confusion_matrix(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_pred: Union[np.ndarray, List[int]]
    ) -> Dict[str, Any]:
        """
        Calcule la matrice de confusion.
        
        Args:
            y_true: Labels réels
            y_pred: Labels prédits
            
        Returns:
            Dictionnaire avec la matrice et les métriques dérivées
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Extraire TN, FP, FN, TP
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,  # = recall
        }
    
    def calculate_performance_metrics(
        self,
        training_time: float,
        inference_time: float,
        n_samples_train: int,
        n_samples_test: int,
        memory_usage_mb: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calcule les métriques de performance.
        
        Args:
            training_time: Temps d'entraînement (secondes)
            inference_time: Temps d'inférence (secondes)
            n_samples_train: Nombre d'échantillons d'entraînement
            n_samples_test: Nombre d'échantillons de test
            memory_usage_mb: Utilisation mémoire (MB)
            
        Returns:
            Dictionnaire des métriques
        """
        metrics = {
            'training_time_seconds': training_time,
            'inference_time_seconds': inference_time,
            'training_time_per_sample_ms': (training_time / max(n_samples_train, 1)) * 1000,
            'inference_time_per_sample_ms': (inference_time / max(n_samples_test, 1)) * 1000,
            'throughput_samples_per_second': n_samples_test / max(inference_time, 0.001),
        }
        
        if memory_usage_mb is not None:
            metrics['memory_usage_mb'] = memory_usage_mb
        
        return metrics
    
    def get_classification_report(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_pred: Union[np.ndarray, List[int]],
        target_names: Optional[List[str]] = None
    ) -> str:
        """
        Génère un rapport de classification textuel.
        
        Args:
            y_true: Labels réels
            y_pred: Labels prédits
            target_names: Noms des classes
            
        Returns:
            Rapport textuel
        """
        if target_names is None:
            target_names = ['Flaky/False Positive', 'Non Flaky (Possible Regression)']
        
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def calculate_roc_curve(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_proba: np.ndarray
    ) -> Dict[str, List[float]]:
        """
        Calcule la courbe ROC.
        
        Args:
            y_true: Labels réels
            y_proba: Probabilités
            
        Returns:
            Dictionnaire avec FPR, TPR et seuils
        """
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
    
    def calculate_pr_curve(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_proba: np.ndarray
    ) -> Dict[str, List[float]]:
        """
        Calcule la courbe Precision-Recall.
        
        Args:
            y_true: Labels réels
            y_proba: Probabilités
            
        Returns:
            Dictionnaire avec Precision, Recall et seuils
        """
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1]
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist()
        }
    
    def aggregate_metrics(
        self,
        metrics_list: List[Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Agrège les métriques de plusieurs runs.
        
        Args:
            metrics_list: Liste des dictionnaires de métriques
            
        Returns:
            Dictionnaire avec mean, std, min, max pour chaque métrique
        """
        if not metrics_list:
            return {}
        
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())
        
        aggregated = {}
        
        for key in all_keys:
            values = [m.get(key) for m in metrics_list if key in m and isinstance(m.get(key), (int, float))]
            
            if values:
                aggregated[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'n_runs': len(values)
                }
        
        return aggregated
    
    def evaluate_model(
        self,
        model_name: str,
        y_true: Union[np.ndarray, List[int]],
        y_pred: Union[np.ndarray, List[int]],
        y_proba: Optional[np.ndarray] = None,
        training_stats: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Évalue un modèle de manière complète.
        
        Args:
            model_name: Nom du modèle
            y_true: Labels réels
            y_pred: Labels prédits
            y_proba: Probabilités (optionnel)
            training_stats: Statistiques d'entraînement (optionnel)
            
        Returns:
            Dictionnaire complet des résultats
        """
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(y_true),
        }
        
        # Métriques de classification
        results['classification_metrics'] = self.calculate_classification_metrics(
            y_true, y_pred, y_proba
        )
        
        # Matrice de confusion
        results['confusion_matrix'] = self.calculate_confusion_matrix(y_true, y_pred)
        
        # Rapport textuel
        results['classification_report'] = self.get_classification_report(y_true, y_pred)
        
        # Courbes si probabilités disponibles
        if y_proba is not None:
            results['roc_curve'] = self.calculate_roc_curve(y_true, y_proba)
            results['pr_curve'] = self.calculate_pr_curve(y_true, y_proba)
        
        # Statistiques d'entraînement
        if training_stats:
            results['training_stats'] = training_stats
        
        # Ajouter à l'historique
        self.results_history.append(results)
        
        return results
    
    def save_results(self, filepath: str) -> None:
        """
        Sauvegarde l'historique des résultats.
        
        Args:
            filepath: Chemin du fichier
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results_history, f, indent=2, default=str)
        
        logger.info(f"Résultats sauvegardés: {filepath}")
    
    def load_results(self, filepath: str) -> List[Dict]:
        """
        Charge des résultats sauvegardés.
        
        Args:
            filepath: Chemin du fichier
            
        Returns:
            Liste des résultats
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            self.results_history = json.load(f)
        
        return self.results_history
    
    def results_to_dataframe(self) -> pd.DataFrame:
        """
        Convertit l'historique en DataFrame.
        
        Returns:
            DataFrame des résultats
        """
        rows = []
        
        for result in self.results_history:
            row = {
                'model_name': result.get('model_name'),
                'timestamp': result.get('timestamp'),
                'n_samples': result.get('n_samples'),
            }
            
            # Aplatir les métriques de classification
            if 'classification_metrics' in result:
                for key, value in result['classification_metrics'].items():
                    row[key] = value
            
            # Aplatir les stats d'entraînement
            if 'training_stats' in result:
                for key, value in result['training_stats'].items():
                    if isinstance(value, (int, float)):
                        row[f'train_{key}'] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)


def main():
    """Fonction de test du calculateur de métriques."""
    import numpy as np
    
    print("Test du MetricsCalculator")
    print("=" * 50)
    
    # Données de test
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = y_true.copy()
    # Ajouter du bruit
    noise_idx = np.random.choice(n_samples, size=10, replace=False)
    y_pred[noise_idx] = 1 - y_pred[noise_idx]
    
    y_proba = np.random.rand(n_samples, 2)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    # Calculer les métriques
    calculator = MetricsCalculator()
    
    results = calculator.evaluate_model(
        model_name="TestModel",
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        training_stats={
            'training_time_seconds': 10.5,
            'memory_usage_mb': 256.0,
            'n_samples': 800,
        }
    )
    
    print("\nMétriques de classification:")
    for key, value in results['classification_metrics'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nMatrice de confusion:")
    cm = results['confusion_matrix']
    print(f"  TN={cm['true_negatives']}, FP={cm['false_positives']}")
    print(f"  FN={cm['false_negatives']}, TP={cm['true_positives']}")
    
    print("\nRapport de classification:")
    print(results['classification_report'])
    
    # Test d'agrégation
    print("\nTest d'agrégation (3 runs):")
    metrics_list = [
        {'accuracy': 0.90, 'f1': 0.88},
        {'accuracy': 0.92, 'f1': 0.90},
        {'accuracy': 0.91, 'f1': 0.89},
    ]
    
    agg = calculator.aggregate_metrics(metrics_list)
    for key, stats in agg.items():
        print(f"  {key}: mean={stats['mean']:.4f} ± {stats['std']:.4f}")


if __name__ == "__main__":
    main()
