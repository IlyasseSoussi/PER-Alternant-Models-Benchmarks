"""
Preprocessor - Nettoyage et normalisation des logs
==================================================

Ce module gère le prétraitement des logs CI/CD :
- Suppression des timestamps, IDs, chemins systèmes
- Normalisation du texte
- Filtrage des lignes pertinentes
"""

import re
import pandas as pd
from typing import List, Optional, Dict, Any
import logging
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogPreprocessor:
    """
    Classe pour le nettoyage et la normalisation des logs CI/CD.
    
    Attributes:
        config: Configuration du prétraitement
    """
    
    # Patterns par défaut à supprimer
    DEFAULT_REMOVE_PATTERNS = [
        r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?',  # ISO timestamps
        r'\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2}',  # US timestamps
        r'\d{2}-\d{2}-\d{4}\s\d{2}:\d{2}:\d{2}',  # EU timestamps
        r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}',  # UUIDs
        r'(?:/[a-zA-Z0-9_\-\.]+)+/?',  # Unix paths
        r'[A-Z]:\\(?:[a-zA-Z0-9_\-\.]+\\)*[a-zA-Z0-9_\-\.]*',  # Windows paths
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  # IP addresses
        r'0x[a-fA-F0-9]+',  # Hex addresses
        r'\b[a-fA-F0-9]{32}\b',  # MD5 hashes
        r'\b[a-fA-F0-9]{40}\b',  # SHA1 hashes
        r'\b[a-fA-F0-9]{64}\b',  # SHA256 hashes
        r'\[\d+\]',  # Process IDs in brackets
        r'#\d+',  # Issue/PR numbers
        r'@[a-zA-Z0-9_]+',  # Mentions
    ]
    
    # Mots-clés importants pour les logs CI/CD
    IMPORTANT_KEYWORDS = [
        'error', 'fail', 'failed', 'failure', 'exception', 'assert',
        'timeout', 'null', 'undefined', 'crash', 'fatal', 'warning',
        'flaky', 'retry', 'unstable', 'intermittent', 'skipped',
        'success', 'passed', 'pass', 'build', 'test', 'deploy',
        'connection', 'refused', 'denied', 'permission', 'access',
        'memory', 'heap', 'stack', 'overflow', 'leak',
        'race', 'condition', 'deadlock', 'concurrent'
    ]
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise le preprocessor.
        
        Args:
            config_path: Chemin vers le fichier de configuration YAML
        """
        self.config = self._load_config(config_path)
        self.remove_patterns = self._compile_patterns()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Charge la configuration depuis un fichier YAML."""
        default_config = {
            'lowercase': True,
            'remove_special_chars': True,
            'min_line_length': 10,
            'max_line_length': 512,
            'remove_patterns': self.DEFAULT_REMOVE_PATTERNS,
            'important_keywords': self.IMPORTANT_KEYWORDS
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                if 'preprocessing' in yaml_config:
                    default_config.update(yaml_config['preprocessing'])
        
        return default_config
    
    def _compile_patterns(self) -> List[re.Pattern]:
        """Compile les patterns regex."""
        patterns = self.config.get('remove_patterns', self.DEFAULT_REMOVE_PATTERNS)
        compiled = []
        for pattern in patterns:
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Pattern invalide ignoré: {pattern} - {e}")
        return compiled
    
    def clean_text(self, text: str) -> str:
        """
        Nettoie un texte de log.
        
        Args:
            text: Texte brut du log
            
        Returns:
            Texte nettoyé
        """
        if not isinstance(text, str) or not text:
            return ""
        
        # Supprimer les patterns non informatifs
        for pattern in self.remove_patterns:
            text = pattern.sub(' ', text)
        
        # Supprimer les caractères spéciaux si configuré
        if self.config.get('remove_special_chars', True):
            # Garder lettres, chiffres, espaces et quelques caractères utiles
            text = re.sub(r'[^\w\s\.\,\:\;\!\?\-\(\)]', ' ', text)
        
        # Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)
        
        # Lowercase si configuré
        if self.config.get('lowercase', True):
            text = text.lower()
        
        return text.strip()
    
    def filter_relevant_lines(self, text: str) -> bool:
        """
        Vérifie si une ligne est pertinente pour l'analyse.
        
        Args:
            text: Texte du log
            
        Returns:
            True si la ligne est pertinente
        """
        if not text:
            return False
        
        # Vérifier la longueur
        min_len = self.config.get('min_line_length', 10)
        max_len = self.config.get('max_line_length', 512)
        
        if len(text) < min_len or len(text) > max_len:
            return False
        
        # Vérifier la présence de mots-clés importants
        keywords = self.config.get('important_keywords', self.IMPORTANT_KEYWORDS)
        text_lower = text.lower()
        
        return any(keyword in text_lower for keyword in keywords)
    
    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'log_text',
        filter_relevant: bool = True
    ) -> pd.DataFrame:
        """
        Prétraite un DataFrame de logs.
        
        Args:
            df: DataFrame avec les logs
            text_column: Nom de la colonne de texte
            filter_relevant: Filtrer uniquement les lignes pertinentes
            
        Returns:
            DataFrame prétraité
        """
        logger.info(f"Prétraitement de {len(df)} logs...")
        
        # Copie pour éviter les modifications in-place
        result = df.copy()
        
        # Nettoyage du texte
        result['cleaned_text'] = result[text_column].apply(self.clean_text)
        
        # Filtrage des lignes pertinentes
        if filter_relevant:
            mask = result['cleaned_text'].apply(self.filter_relevant_lines)
            initial_count = len(result)
            result = result[mask].reset_index(drop=True)
            logger.info(f"Filtrage: {initial_count} -> {len(result)} logs ({len(result)/max(initial_count,1)*100:.1f}%)")
        
        # Supprimer les doublons
        initial_count = len(result)
        result = result.drop_duplicates(subset=['cleaned_text']).reset_index(drop=True)
        if initial_count != len(result):
            logger.info(f"Doublons supprimés: {initial_count} -> {len(result)}")
        
        logger.info(f"Prétraitement terminé: {len(result)} logs")
        
        return result
    
    def extract_features_manual(self, text: str) -> Dict[str, Any]:
        """
        Extrait des features manuelles d'un log.
        
        Args:
            text: Texte du log
            
        Returns:
            Dictionnaire de features
        """
        text_lower = text.lower()
        
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'has_error': 'error' in text_lower,
            'has_fail': 'fail' in text_lower,
            'has_exception': 'exception' in text_lower,
            'has_warning': 'warning' in text_lower,
            'has_timeout': 'timeout' in text_lower,
            'has_retry': 'retry' in text_lower or 'flaky' in text_lower,
            'has_success': 'success' in text_lower or 'passed' in text_lower,
            'exclamation_count': text.count('!'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
        }
        
        return features
    
    def add_manual_features(self, df: pd.DataFrame, text_column: str = 'cleaned_text') -> pd.DataFrame:
        """
        Ajoute des features manuelles au DataFrame.
        
        Args:
            df: DataFrame avec les logs
            text_column: Colonne de texte à analyser
            
        Returns:
            DataFrame avec les features ajoutées
        """
        result = df.copy()
        
        features_list = result[text_column].apply(self.extract_features_manual)
        features_df = pd.DataFrame(features_list.tolist())
        
        # Préfixer les colonnes
        features_df.columns = [f'feat_{col}' for col in features_df.columns]
        
        return pd.concat([result, features_df], axis=1)


def main():
    """Fonction de test du preprocessor."""
    preprocessor = LogPreprocessor()
    
    # Test avec des exemples
    test_logs = [
        "2024-01-15T10:30:45.123Z [ERROR] Test failed: NullPointerException at com.example.MyClass",
        "Build successful - all 42 tests passed",
        "[INFO] Starting deployment to production server 192.168.1.100",
        "FLAKY: Test intermittent failure detected, retrying...",
        "===",  # Ligne non pertinente
    ]
    
    print("Test du LogPreprocessor")
    print("=" * 50)
    
    for log in test_logs:
        cleaned = preprocessor.clean_text(log)
        relevant = preprocessor.filter_relevant_lines(cleaned)
        print(f"\nOriginal: {log[:60]}...")
        print(f"Nettoyé:  {cleaned[:60]}...")
        print(f"Pertinent: {relevant}")


if __name__ == "__main__":
    main()
