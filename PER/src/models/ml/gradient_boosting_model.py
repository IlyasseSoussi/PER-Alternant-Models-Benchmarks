"""
Gradient Boosting Model - Classification de logs avec Gradient Boosting
======================================================================

Implémentation du modèle Gradient Boosting pour la classification binaire
de logs CI/CD.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from typing import Optional, Dict, Any, Union
import logging
import pickle
from pathlib import Path
import yaml
import time
import psutil
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradientBoostingModel:
    """
    Wrapper pour Gradient Boosting adapté à la classification de logs.
    
    Attributes:
        model: Instance de GradientBoostingClassifier
        is_fitted: Indique si le modèle a été entraîné
        training_stats: Statistiques d'entraînement
    """
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialise le classifieur Gradient Boosting.
        
        Args:
            config_path: Chemin vers le fichier de configuration
            **kwargs: Paramètres supplémentaires pour GradientBoostingClassifier
        """
        self.config = self._load_config(config_path)
        self.model = self._create_model(**kwargs)
        self.is_fitted = False
        self.training_stats = {}
        self.model_name = "Gradient Boosting"
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Charge la configuration Gradient Boosting."""
        default_config = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10,
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                if 'ml' in yaml_config and 'gradient_boosting' in yaml_config['ml']:
                    default_config.update(yaml_config['ml']['gradient_boosting'])
        
        return default_config
    
    def _create_model(self, **kwargs) -> GradientBoostingClassifier:
        """Crée le modèle Gradient Boosting."""
        params = {
            'n_estimators': self.config.get('n_estimators', 100),
            'max_depth': self.config.get('max_depth', 6),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'subsample': self.config.get('subsample', 0.8),
            'random_state': self.config.get('random_state', 42),
            'validation_fraction': self.config.get('validation_fraction', 0.1),
            'n_iter_no_change': self.config.get('n_iter_no_change', 10),
            'verbose': 0,
        }
        
        # Override avec kwargs
        params.update(kwargs)
        
        return GradientBoostingClassifier(**params)
    
    def _get_memory_usage(self) -> float:
        """Retourne l'utilisation mémoire en MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        verbose: bool = True
    ) -> 'GradientBoostingModel':
        """
        Entraîne le modèle.
        
        Args:
            X: Features d'entraînement
            y: Labels d'entraînement
            verbose: Afficher les logs
            
        Returns:
            self
        """
        if verbose:
            logger.info(f"Entraînement Gradient Boosting sur {X.shape[0]} échantillons...")
        
        # Convertir sparse matrix si nécessaire
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Mesures
        start_time = time.time()
        mem_before = self._get_memory_usage()
        
        # Entraînement
        self.model.fit(X, y)
        
        # Statistiques
        training_time = time.time() - start_time
        mem_after = self._get_memory_usage()
        
        self.training_stats = {
            'training_time_seconds': training_time,
            'memory_usage_mb': mem_after - mem_before,
            'n_samples': X.shape[0],
            'n_features': X.shape[1] if len(X.shape) > 1 else 1,
            'n_estimators_actual': self.model.n_estimators_,
            'train_score': self.model.train_score_[-1] if hasattr(self.model, 'train_score_') else None,
        }
        
        self.is_fitted = True
        
        if verbose:
            logger.info(f"Entraînement terminé en {training_time:.2f}s")
            logger.info(f"Nombre d'estimateurs: {self.training_stats['n_estimators_actual']}")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Prédit les classes.
        
        Args:
            X: Features
            
        Returns:
            Prédictions
        """
        if not self.is_fitted:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez fit() d'abord.")
        
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Prédit les probabilités.
        
        Args:
            X: Features
            
        Returns:
            Probabilités par classe
        """
        if not self.is_fitted:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez fit() d'abord.")
        
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Dict[str, float]:
        """
        Évalue le modèle sur un ensemble de données.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dictionnaire des métriques
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Mesure du temps d'inférence
        start_time = time.time()
        y_pred = self.predict(X)
        inference_time = time.time() - start_time
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='binary'),
            'recall': recall_score(y, y_pred, average='binary'),
            'f1': f1_score(y, y_pred, average='binary'),
            'inference_time_seconds': inference_time,
            'inference_time_per_sample_ms': (inference_time / len(y)) * 1000,
        }
    
    def cross_validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Validation croisée.
        
        Args:
            X: Features
            y: Labels
            cv: Nombre de folds
            
        Returns:
            Scores moyens
        """
        if hasattr(X, 'toarray'):
            X = X.toarray()
            
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1')
        
        return {
            'mean_f1': scores.mean(),
            'std_f1': scores.std(),
            'scores': scores.tolist(),
        }
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> Dict[str, float]:
        """
        Retourne l'importance des features.
        
        Args:
            feature_names: Noms des features
            
        Returns:
            Dictionnaire feature -> importance
        """
        if not self.is_fitted:
            return {}
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Trier par importance
        sorted_idx = np.argsort(importances)[::-1]
        
        return {feature_names[i]: importances[i] for i in sorted_idx}
    
    def get_staged_scores(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> list:
        """
        Retourne les scores à chaque étape du boosting.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Liste des scores
        """
        from sklearn.metrics import f1_score
        
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        scores = []
        for y_pred in self.model.staged_predict(X):
            score = f1_score(y, y_pred, average='binary')
            scores.append(score)
        
        return scores
    
    def get_training_deviance(self) -> list:
        """
        Retourne l'évolution de la deviance pendant l'entraînement.
        
        Returns:
            Liste des deviances
        """
        if not self.is_fitted:
            return []
        
        return list(self.model.train_score_)
    
    def save(self, filepath: str) -> None:
        """Sauvegarde le modèle."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'is_fitted': self.is_fitted,
                'training_stats': self.training_stats,
                'config': self.config
            }, f)
        
        logger.info(f"Modèle sauvegardé: {filepath}")
    
    def load(self, filepath: str) -> 'GradientBoostingModel':
        """Charge un modèle sauvegardé."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.is_fitted = data['is_fitted']
        self.training_stats = data.get('training_stats', {})
        self.config = data.get('config', self.config)
        
        logger.info(f"Modèle chargé: {filepath}")
        
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """Retourne les paramètres du modèle."""
        return self.model.get_params()
    
    def hyperparameter_search(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        param_grid: Optional[Dict] = None,
        cv: int = 3
    ) -> Dict[str, Any]:
        """
        Recherche d'hyperparamètres.
        
        Args:
            X: Features
            y: Labels
            param_grid: Grille de paramètres
            cv: Nombre de folds
            
        Returns:
            Meilleurs paramètres
        """
        if hasattr(X, 'toarray'):
            X = X.toarray()
            
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
            }
        
        logger.info("Recherche d'hyperparamètres Gradient Boosting...")
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Mettre à jour avec les meilleurs paramètres
        self.model = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
        }


def main():
    """Fonction de test du modèle Gradient Boosting."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("Test du GradientBoostingModel")
    print("=" * 50)
    
    # Données de test
    X, y = make_classification(
        n_samples=1000,
        n_features=100,
        n_informative=20,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entraîner
    model = GradientBoostingModel()
    model.fit(X_train, y_train)
    
    # Évaluer
    metrics = model.evaluate(X_test, y_test)
    
    print("\nMétriques:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    print("\nStatistiques d'entraînement:")
    for name, value in model.training_stats.items():
        if value is not None:
            print(f"  {name}: {value}")
    
    # Deviance
    deviance = model.get_training_deviance()
    print(f"\nDeviance (première/dernière): {deviance[0]:.4f} -> {deviance[-1]:.4f}")
    
    # Staged scores
    staged = model.get_staged_scores(X_test, y_test)
    print(f"Scores staged (premiers 5): {[f'{s:.4f}' for s in staged[:5]]}")
    print(f"Score final: {staged[-1]:.4f}")


if __name__ == "__main__":
    main()
