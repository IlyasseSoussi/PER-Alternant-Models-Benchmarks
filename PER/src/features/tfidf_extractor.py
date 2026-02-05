"""
TF-IDF Extractor - Vectorisation pour les modèles ML
====================================================

Ce module gère l'extraction de features TF-IDF pour les modèles
de Machine Learning (Boosting).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional, Dict, Any, Tuple, Union
import logging
import pickle
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFIDFExtractor:
    """
    Extracteur de features TF-IDF pour les logs CI/CD.
    
    Attributes:
        vectorizer: Instance de TfidfVectorizer
        is_fitted: Indique si le vectorizer a été entraîné
    """
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialise l'extracteur TF-IDF.
        
        Args:
            config_path: Chemin vers le fichier de configuration
            **kwargs: Paramètres supplémentaires pour TfidfVectorizer
        """
        self.config = self._load_config(config_path)
        self.vectorizer = self._create_vectorizer(**kwargs)
        self.is_fitted = False
        self.feature_names = None
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Charge la configuration TF-IDF."""
        default_config = {
            'max_features': 5000,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95,
            'sublinear_tf': True,
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                if 'ml' in yaml_config and 'tfidf' in yaml_config['ml']:
                    tfidf_config = yaml_config['ml']['tfidf']
                    # Convertir ngram_range de liste à tuple
                    if 'ngram_range' in tfidf_config:
                        tfidf_config['ngram_range'] = tuple(tfidf_config['ngram_range'])
                    default_config.update(tfidf_config)
        
        return default_config
    
    def _create_vectorizer(self, **kwargs) -> TfidfVectorizer:
        """Crée le vectorizer TF-IDF."""
        params = {
            'max_features': self.config.get('max_features', 5000),
            'ngram_range': self.config.get('ngram_range', (1, 2)),
            'min_df': self.config.get('min_df', 2),
            'max_df': self.config.get('max_df', 0.95),
            'sublinear_tf': self.config.get('sublinear_tf', True),
            'strip_accents': 'unicode',
            'lowercase': True,
            'analyzer': 'word',
            'token_pattern': r'\b[a-zA-Z][a-zA-Z0-9_]*\b',  # Mots commençant par une lettre
        }
        
        # Override avec kwargs
        params.update(kwargs)
        
        return TfidfVectorizer(**params)
    
    def fit(self, texts: Union[pd.Series, list]) -> 'TFIDFExtractor':
        """
        Entraîne le vectorizer sur un corpus.
        
        Args:
            texts: Textes d'entraînement
            
        Returns:
            self
        """
        logger.info(f"Entraînement TF-IDF sur {len(texts)} documents...")
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        self.vectorizer.fit(texts)
        self.is_fitted = True
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        logger.info(f"Vocabulaire créé: {len(self.feature_names)} features")
        
        return self
    
    def transform(self, texts: Union[pd.Series, list]) -> np.ndarray:
        """
        Transforme des textes en vecteurs TF-IDF.
        
        Args:
            texts: Textes à transformer
            
        Returns:
            Matrice sparse ou dense de features
        """
        if not self.is_fitted:
            raise ValueError("Le vectorizer n'a pas été entraîné. Appelez fit() d'abord.")
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts: Union[pd.Series, list]) -> np.ndarray:
        """
        Entraîne et transforme en une seule opération.
        
        Args:
            texts: Textes à traiter
            
        Returns:
            Matrice de features
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_top_features(self, n: int = 20) -> list:
        """
        Retourne les features les plus fréquentes.
        
        Args:
            n: Nombre de features à retourner
            
        Returns:
            Liste des top features
        """
        if not self.is_fitted:
            return []
        
        # Calculer l'importance moyenne de chaque feature
        idf_scores = self.vectorizer.idf_
        indices = np.argsort(idf_scores)[:n]  # IDF bas = plus fréquent
        
        return [self.feature_names[i] for i in indices]
    
    def get_feature_importance_for_text(self, text: str, n: int = 10) -> Dict[str, float]:
        """
        Retourne les features les plus importantes pour un texte donné.
        
        Args:
            text: Texte à analyser
            n: Nombre de features à retourner
            
        Returns:
            Dictionnaire feature -> score
        """
        if not self.is_fitted:
            return {}
        
        vector = self.vectorizer.transform([text]).toarray()[0]
        
        # Trouver les indices des features non nulles
        nonzero_indices = np.nonzero(vector)[0]
        
        # Trier par score
        sorted_indices = sorted(nonzero_indices, key=lambda i: vector[i], reverse=True)[:n]
        
        return {self.feature_names[i]: vector[i] for i in sorted_indices}
    
    def save(self, filepath: str) -> None:
        """
        Sauvegarde le vectorizer.
        
        Args:
            filepath: Chemin de destination
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'is_fitted': self.is_fitted,
                'feature_names': self.feature_names,
                'config': self.config
            }, f)
        
        logger.info(f"Vectorizer sauvegardé: {filepath}")
    
    def load(self, filepath: str) -> 'TFIDFExtractor':
        """
        Charge un vectorizer sauvegardé.
        
        Args:
            filepath: Chemin du fichier
            
        Returns:
            self
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.vectorizer = data['vectorizer']
        self.is_fitted = data['is_fitted']
        self.feature_names = data['feature_names']
        self.config = data.get('config', self.config)
        
        logger.info(f"Vectorizer chargé: {filepath}")
        
        return self
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur le vocabulaire.
        
        Returns:
            Dictionnaire de statistiques
        """
        if not self.is_fitted:
            return {}
        
        return {
            'vocabulary_size': len(self.feature_names),
            'max_features': self.config.get('max_features'),
            'ngram_range': self.config.get('ngram_range'),
            'min_df': self.config.get('min_df'),
            'max_df': self.config.get('max_df'),
        }


def main():
    """Fonction de test de l'extracteur TF-IDF."""
    extractor = TFIDFExtractor()
    
    # Corpus de test
    corpus = [
        "Test failed with NullPointerException in UserService",
        "Connection timeout, retrying operation",
        "Build successful, all tests passed",
        "FLAKY: Test intermittently fails due to race condition",
        "REGRESSION: Expected 42 but got 0 in calculation",
        "Network error: connection refused to database server",
        "Fatal error: segmentation fault in memory allocator",
        "Test flapping, marked as unstable",
    ]
    
    print("Test du TFIDFExtractor")
    print("=" * 50)
    
    # Fit et transform
    X = extractor.fit_transform(corpus)
    
    print(f"\nCorpus size: {len(corpus)} documents")
    print(f"Feature matrix shape: {X.shape}")
    
    # Statistiques
    stats = extractor.get_vocabulary_stats()
    print(f"\nVocabulary stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Top features
    print(f"\nTop 10 features (par fréquence):")
    for feat in extractor.get_top_features(10):
        print(f"  - {feat}")
    
    # Analyse d'un texte
    test_text = "NullPointerException caused test failure in production"
    print(f"\nAnalyse de: '{test_text}'")
    importance = extractor.get_feature_importance_for_text(test_text, 5)
    for feat, score in importance.items():
        print(f"  {feat}: {score:.4f}")


if __name__ == "__main__":
    main()
