"""
Loader - Chargement des logs CI/CD
==================================

Ce module gère le chargement des logs depuis différentes sources :
- Fichiers texte (.log, .txt)
- Fichiers CSV
- Répertoires de logs
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogLoader:
    """
    Classe pour charger des logs CI/CD depuis différents formats.
    
    Attributes:
        supported_extensions: Extensions de fichiers supportées
    """
    
    supported_extensions = ['.log', '.txt', '.csv']
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialise le loader.
        
        Args:
            base_path: Chemin de base pour la recherche de fichiers
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
    
    def load_file(self, filepath: str) -> pd.DataFrame:
        """
        Charge un fichier de logs.
        
        Args:
            filepath: Chemin vers le fichier
            
        Returns:
            DataFrame avec les logs
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
        
        extension = filepath.suffix.lower()
        
        if extension == '.csv':
            return self._load_csv(filepath)
        elif extension in ['.log', '.txt']:
            return self._load_text(filepath)
        else:
            raise ValueError(f"Extension non supportée: {extension}")
    
    def _load_csv(self, filepath: Path) -> pd.DataFrame:
        """Charge un fichier CSV."""
        logger.info(f"Chargement CSV: {filepath}")
        
        # Essayer différents séparateurs
        for sep in [',', ';', '\t']:
            try:
                df = pd.read_csv(filepath, sep=sep, encoding='utf-8')
                if len(df.columns) > 1:
                    return df
            except Exception:
                continue
        
        # Fallback: lire comme texte
        return self._load_text(filepath)
    
    def _load_text(self, filepath: Path) -> pd.DataFrame:
        """Charge un fichier texte ligne par ligne."""
        logger.info(f"Chargement texte: {filepath}")
        
        lines = []
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                break
            except UnicodeDecodeError:
                continue
        
        # Nettoyer les lignes
        lines = [line.strip() for line in lines if line.strip()]
        
        return pd.DataFrame({'log_text': lines})
    
    def load_directory(self, dirpath: str, recursive: bool = True) -> pd.DataFrame:
        """
        Charge tous les logs d'un répertoire.
        
        Args:
            dirpath: Chemin du répertoire
            recursive: Recherche récursive
            
        Returns:
            DataFrame combiné de tous les logs
        """
        dirpath = Path(dirpath)
        
        if not dirpath.exists():
            raise FileNotFoundError(f"Répertoire non trouvé: {dirpath}")
        
        all_logs = []
        
        pattern = '**/*' if recursive else '*'
        
        for filepath in dirpath.glob(pattern):
            if filepath.is_file() and filepath.suffix.lower() in self.supported_extensions:
                try:
                    df = self.load_file(str(filepath))
                    df['source_file'] = filepath.name
                    all_logs.append(df)
                    logger.info(f"Chargé: {filepath.name} ({len(df)} lignes)")
                except Exception as e:
                    logger.warning(f"Erreur chargement {filepath}: {e}")
        
        if not all_logs:
            logger.warning(f"Aucun fichier log trouvé dans {dirpath}")
            return pd.DataFrame(columns=['log_text'])
        
        combined = pd.concat(all_logs, ignore_index=True)
        logger.info(f"Total: {len(combined)} lignes chargées")
        
        return combined
    
    def load_from_dataframe(self, df: pd.DataFrame, text_column: str = 'log_text') -> pd.DataFrame:
        """
        Charge des logs depuis un DataFrame existant.
        
        Args:
            df: DataFrame source
            text_column: Nom de la colonne contenant le texte
            
        Returns:
            DataFrame formaté
        """
        if text_column not in df.columns:
            # Chercher une colonne de texte
            text_cols = [col for col in df.columns if 'text' in col.lower() or 'log' in col.lower() or 'message' in col.lower()]
            if text_cols:
                text_column = text_cols[0]
            else:
                text_column = df.columns[0]
        
        result = pd.DataFrame({'log_text': df[text_column].astype(str)})
        
        # Copier les colonnes de label si présentes
        label_cols = [col for col in df.columns if 'label' in col.lower() or 'target' in col.lower() or 'class' in col.lower()]
        for col in label_cols:
            result[col] = df[col]
        
        return result


def main():
    """Fonction de test du loader."""
    loader = LogLoader()
    
    # Test avec un fichier exemple
    print("LogLoader initialisé avec succès")
    print(f"Extensions supportées: {loader.supported_extensions}")


if __name__ == "__main__":
    main()
