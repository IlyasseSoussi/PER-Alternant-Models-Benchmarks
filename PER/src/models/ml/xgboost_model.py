"""
XGBoost Model - Classification de logs avec XGBoost
===================================================

Implémentation du modèle XGBoost pour la classification binaire
de logs CI/CD.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from typing import Optional, Dict, Any, Union, Tuple
import logging
import pickle
from pathlib import Path
import yaml
import time
import psutil
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostClassifier:
    """
    Wrapper pour XGBoost adapté à la classification de logs.
    
    Attributes:
        model: Instance de XGBClassifier
        is_fitted: Indique si le modèle a été entraîné
        training_stats: Statistiques d'entraînement
    """
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialise le classifieur XGBoost.
        
        Args:
            config_path: Chemin vers le fichier de configuration
            **kwargs: Paramètres supplémentaires pour XGBClassifier
        """
        self.config = self._load_config(config_path)
        self.model = self._create_model(**kwargs)
        self.is_fitted = False
        self.training_stats = {}
        self.model_name = "XGBoost"
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Charge la configuration XGBoost."""
        default_config = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss',
            'early_stopping_rounds': 10,
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                if 'ml' in yaml_config and 'xgboost' in yaml_config['ml']:
                    default_config.update(yaml_config['ml']['xgboost'])
        
        return default_config
    
    def _create_model(self, **kwargs) -> XGBClassifier:
        """Crée le modèle XGBoost."""
        params = {
            'n_estimators': self.config.get('n_estimators', 100),
            'max_depth': self.config.get('max_depth', 6),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'subsample': self.config.get('subsample', 0.8),
            'colsample_bytree': self.config.get('colsample_bytree', 0.8),
            'random_state': self.config.get('random_state', 42),
            'eval_metric': self.config.get('eval_metric', 'logloss'),
            'use_label_encoder': False,
            'verbosity': 0,
        }
        
        # Override avec kwargs
        params.update(kwargs)
        
        return XGBClassifier(**params)
    
    def _get_memory_usage(self) -> float:
        """Retourne l'utilisation mémoire en MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        verbose: bool = True
    ) -> 'XGBoostClassifier':
        """
        Entraîne le modèle.
        
        Args:
            X: Features d'entraînement
            y: Labels d'entraînement
            X_val: Features de validation (optionnel)
            y_val: Labels de validation (optionnel)
            verbose: Afficher les logs
            
        Returns:
            self
        """
        if verbose:
            logger.info(f"Entraînement XGBoost sur {X.shape[0]} échantillons...")
        
        # Mesures
        start_time = time.time()
        mem_before = self._get_memory_usage()
        
        # Préparer eval_set si validation fournie
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.model.set_params(early_stopping_rounds=self.config.get('early_stopping_rounds', 10))
        
        # Entraînement
        if eval_set:
            self.model.fit(X, y, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X, y)
        
        # Statistiques
        training_time = time.time() - start_time
        mem_after = self._get_memory_usage()
        
        self.training_stats = {
            'training_time_seconds': training_time,
            'memory_usage_mb': mem_after - mem_before,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'best_iteration': getattr(self.model, 'best_iteration', self.config.get('n_estimators')),
        }
        
        self.is_fitted = True
        
        if verbose:
            logger.info(f"Entraînement terminé en {training_time:.2f}s")
            logger.info(f"Mémoire utilisée: {self.training_stats['memory_usage_mb']:.2f} MB")
        
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
    
    def load(self, filepath: str) -> 'XGBoostClassifier':
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
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
            }
        
        logger.info("Recherche d'hyperparamètres XGBoost...")
        
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
    """Fonction de test du modèle XGBoost."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("Test du XGBoostClassifier")
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
    model = XGBoostClassifier()
    model.fit(X_train, y_train)
    
    # Évaluer
    metrics = model.evaluate(X_test, y_test)
    
    print("\nMétriques:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    print("\nStatistiques d'entraînement:")
    for name, value in model.training_stats.items():
        print(f"  {name}: {value}")
    
    # Feature importance (top 5)
    importance = model.get_feature_importance()
    print("\nTop 5 features:")
    for i, (feat, imp) in enumerate(list(importance.items())[:5]):
        print(f"  {i+1}. {feat}: {imp:.4f}")


if __name__ == "__main__":
    main()
