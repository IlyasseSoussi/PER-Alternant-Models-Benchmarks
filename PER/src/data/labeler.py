"""
Labeler - Labellisation des logs CI/CD
======================================

Ce module gère la labellisation des logs pour la classification binaire :
- Classe 0: Flaky / False Positive
- Classe 1: Non Flaky (Possible Regression)
"""

import re
import pandas as pd
from typing import Optional, Dict, List, Tuple
import logging
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogLabeler:
    """
    Classe pour labelliser les logs CI/CD en classification binaire.
    
    Labels:
        0: Flaky / False Positive (tests instables, problèmes transitoires)
        1: Non Flaky / Possible Regression (erreurs potentiellement réelles)
    """
    
    # Patterns indiquant un log FLAKY (classe 0)
    FLAKY_PATTERNS = [
        r'\bflaky\b',
        r'\bintermittent\b',
        r'\bunstable\b',
        r'\bretry\b',
        r'\bretrying\b',
        r'\bflapping\b',
        r'\btimeout\b(?!.*\bfixed\b)',
        r'\bconnection\s+(?:refused|reset|timed?\s*out)\b',
        r'\bnetwork\s+(?:error|issue|problem)\b',
        r'\brace\s+condition\b',
        r'\bconcurren(?:t|cy)\s+(?:error|issue)\b',
        r'\bresource\s+(?:busy|unavailable)\b',
        r'\btemporary\s+(?:failure|error)\b',
        r'\bspurious\b',
        r'\bnon[\-\s]?deterministic\b',
        r'\brandom(?:ly)?\s+fail',
        r'\bskipped\b',
        r'\bignored\b',
        r'\bknown\s+issue\b',
        r'\bwont\s*fix\b',
        r'\bexpected\s+(?:failure|to\s+fail)\b',
    ]
    
    # Patterns indiquant NON FLAKY / POSSIBLE REGRESSION (classe 1)
    REGRESSION_PATTERNS = [
        r'\bregression\b',
        r'\bbug\b',
        r'\bdefect\b',
        r'\bbroken\b',
        r'\bbreak(?:s|ing)?\b',
        r'\bfatal\b',
        r'\bcritical\b',
        r'\bsevere\b',
        r'\bnull\s*pointer\b',
        r'\bsegmentation\s+fault\b',
        r'\bsegfault\b',
        r'\bstack\s*overflow\b',
        r'\bheap\s+(?:corruption|overflow)\b',
        r'\bmemory\s+leak\b',
        r'\bout\s+of\s+memory\b',
        r'\bindex\s+out\s+of\s+(?:bounds|range)\b',
        r'\bassertion\s+(?:failed|error)\b',
        r'\bunhandled\s+exception\b',
        r'\binvalid\s+(?:argument|parameter|input)\b',
        r'\btype\s*error\b',
        r'\bvalue\s*error\b',
        r'\bsyntax\s*error\b',
        r'\bcompilation?\s+(?:error|failed)\b',
        r'\blink(?:er|ing)?\s+error\b',
        r'\bpermission\s+denied\b',
        r'\baccess\s+denied\b',
        r'\bauthentication\s+(?:failed|error)\b',
        r'\bauthorization\s+(?:failed|error)\b',
        r'\bincorrect\s+(?:result|output)\b',
        r'\bwrong\s+(?:result|output|value)\b',
        r'\bmismatch\b',
        r'\bexpected.*but\s+(?:got|was|received)\b',
    ]
    
    # Patterns neutres (ni flaky ni regression clairement)
    NEUTRAL_PATTERNS = [
        r'\bsuccess(?:ful(?:ly)?)?\b',
        r'\bpass(?:ed)?\b',
        r'\bcompleted?\b',
        r'\binfo\b',
        r'\bdebug\b',
        r'\bstarting\b',
        r'\binitialized?\b',
    ]
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise le labeler.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config = self._load_config(config_path)
        self.flaky_patterns = [re.compile(p, re.IGNORECASE) for p in self.FLAKY_PATTERNS]
        self.regression_patterns = [re.compile(p, re.IGNORECASE) for p in self.REGRESSION_PATTERNS]
        self.neutral_patterns = [re.compile(p, re.IGNORECASE) for p in self.NEUTRAL_PATTERNS]
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Charge la configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def compute_label_scores(self, text: str) -> Tuple[float, float]:
        """
        Calcule les scores pour chaque classe.
        
        Args:
            text: Texte du log
            
        Returns:
            Tuple (score_flaky, score_regression)
        """
        flaky_score = 0.0
        regression_score = 0.0
        
        # Compter les matches flaky
        for pattern in self.flaky_patterns:
            matches = pattern.findall(text)
            flaky_score += len(matches)
        
        # Compter les matches regression
        for pattern in self.regression_patterns:
            matches = pattern.findall(text)
            regression_score += len(matches)
        
        return flaky_score, regression_score
    
    def label_single(self, text: str) -> int:
        """
        Labellise un seul log.
        
        Args:
            text: Texte du log
            
        Returns:
            0 pour flaky/false positive, 1 pour true regression
        """
        if not isinstance(text, str) or not text:
            return 0  # Par défaut flaky si pas de texte
        
        flaky_score, regression_score = self.compute_label_scores(text)
        
        # Si les deux scores sont nuls, vérifier les patterns neutres
        if flaky_score == 0 and regression_score == 0:
            # Vérifier si c'est un message de succès
            for pattern in self.neutral_patterns:
                if pattern.search(text):
                    return 0  # Pas une regression
            # Sinon, se baser sur des heuristiques
            text_lower = text.lower()
            if 'error' in text_lower or 'fail' in text_lower or 'exception' in text_lower:
                return 1  # Probablement regression
            return 0  # Par défaut flaky
        
        # Décider en fonction des scores
        if regression_score > flaky_score:
            return 1  # Non flaky (possible regression)
        elif flaky_score > regression_score:
            return 0  # Flaky
        else:
            # Scores égaux, utiliser des heuristiques
            return 1 if 'error' in text.lower() else 0
    
    def label_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'cleaned_text',
        include_scores: bool = False
    ) -> pd.DataFrame:
        """
        Labellise un DataFrame de logs.
        
        Args:
            df: DataFrame avec les logs
            text_column: Colonne de texte à analyser
            include_scores: Inclure les scores dans le résultat
            
        Returns:
            DataFrame avec les labels
        """
        logger.info(f"Labellisation de {len(df)} logs...")
        
        result = df.copy()
        
        # Calculer les labels
        result['label'] = result[text_column].apply(self.label_single)
        
        if include_scores:
            scores = result[text_column].apply(
                lambda x: self.compute_label_scores(x) if isinstance(x, str) else (0.0, 0.0)
            )
            result['flaky_score'] = scores.apply(lambda x: x[0])
            result['regression_score'] = scores.apply(lambda x: x[1])
        
        # Statistiques
        label_counts = result['label'].value_counts()
        logger.info(f"Distribution des labels:")
        logger.info(f"  - Flaky (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(result)*100:.1f}%)")
        logger.info(f"  - Regression (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(result)*100:.1f}%)")
        
        return result
    
    def balance_dataset(
        self,
        df: pd.DataFrame,
        target_ratio: float = 0.5,
        method: str = 'undersample'
    ) -> pd.DataFrame:
        """
        Équilibre le dataset entre les classes.
        
        Args:
            df: DataFrame labellisé
            target_ratio: Ratio cible de la classe minoritaire
            method: 'undersample' ou 'oversample'
            
        Returns:
            DataFrame équilibré
        """
        if 'label' not in df.columns:
            raise ValueError("DataFrame doit contenir une colonne 'label'")
        
        class_0 = df[df['label'] == 0]
        class_1 = df[df['label'] == 1]
        
        logger.info(f"Avant équilibrage: Classe 0={len(class_0)}, Classe 1={len(class_1)}")
        
        if method == 'undersample':
            # Sous-échantillonner la classe majoritaire
            min_size = min(len(class_0), len(class_1))
            class_0_balanced = class_0.sample(n=min_size, random_state=42)
            class_1_balanced = class_1.sample(n=min_size, random_state=42)
        
        elif method == 'oversample':
            # Sur-échantillonner la classe minoritaire
            max_size = max(len(class_0), len(class_1))
            if len(class_0) < max_size:
                class_0_balanced = class_0.sample(n=max_size, replace=True, random_state=42)
                class_1_balanced = class_1
            else:
                class_0_balanced = class_0
                class_1_balanced = class_1.sample(n=max_size, replace=True, random_state=42)
        else:
            raise ValueError(f"Méthode inconnue: {method}")
        
        result = pd.concat([class_0_balanced, class_1_balanced], ignore_index=True)
        result = result.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        logger.info(f"Après équilibrage: {len(result)} logs (50/50)")
        
        return result
    
    def get_label_name(self, label: int) -> str:
        """Retourne le nom lisible d'un label."""
        return "Flaky/False Positive" if label == 0 else "True Regression"


def main():
    """Fonction de test du labeler."""
    labeler = LogLabeler()
    
    test_logs = [
        "Test failed intermittently, marking as flaky",
        "FATAL: NullPointerException in MainService.java:42",
        "Connection timeout, retrying in 5 seconds...",
        "REGRESSION: Expected output 42 but got 0",
        "Build successful, all tests passed",
        "WARNING: Test is unstable, may fail randomly",
        "AssertionError: values mismatch in calculation",
    ]
    
    print("Test du LogLabeler")
    print("=" * 60)
    
    for log in test_logs:
        label = labeler.label_single(log)
        flaky_score, reg_score = labeler.compute_label_scores(log)
        print(f"\nLog: {log[:50]}...")
        print(f"  Label: {label} ({labeler.get_label_name(label)})")
        print(f"  Scores: flaky={flaky_score:.1f}, regression={reg_score:.1f}")


if __name__ == "__main__":
    main()
