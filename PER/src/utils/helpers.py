"""
Helpers - Fonctions utilitaires
===============================

Ce module contient des fonctions utilitaires utilisées dans tout le projet.
"""

import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Fixe la graine aléatoire pour la reproductibilité.
    
    Args:
        seed: Valeur de la graine
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Graine aléatoire fixée à {seed}")


def get_device() -> torch.device:
    """
    Retourne le device disponible (GPU si disponible, sinon CPU).
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Utilisation du CPU")
    
    return device


def format_time(seconds: float) -> str:
    """
    Formate un temps en secondes en format lisible.
    
    Args:
        seconds: Temps en secondes
        
    Returns:
        Chaîne formatée
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    Charge la configuration depuis un fichier YAML.
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        Dictionnaire de configuration
    """
    # Chercher le fichier de config dans plusieurs emplacements
    possible_paths = [
        config_path,
        f'../{config_path}',
        f'../../{config_path}',
        Path(__file__).parent.parent.parent / config_path,
    ]
    
    for path in possible_paths:
        path = Path(path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration chargée: {path}")
            return config
    
    logger.warning(f"Configuration non trouvée, utilisation des valeurs par défaut")
    return {}


def create_directory_structure(base_path: str = '.') -> None:
    """
    Crée la structure de répertoires du projet.
    
    Args:
        base_path: Chemin de base
    """
    base_path = Path(base_path)
    
    directories = [
        'data/raw',
        'data/processed',
        'data/small_dataset',
        'data/large_dataset',
        'models',
        'results/ml',
        'results/transformers',
        'results/comparative',
        'results/figures',
        'notebooks',
        'logs',
    ]
    
    for dir_path in directories:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Répertoire créé: {full_path}")


def get_project_root() -> Path:
    """
    Retourne le chemin racine du projet.
    
    Returns:
        Path vers la racine du projet
    """
    # Remonter depuis le fichier actuel
    current = Path(__file__).resolve()
    
    # Chercher le répertoire contenant config/
    for parent in [current] + list(current.parents):
        if (parent / 'config').exists() or (parent / 'requirements.txt').exists():
            return parent
    
    # Fallback
    return Path.cwd()


def print_gpu_info() -> None:
    """Affiche les informations sur le GPU."""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Mémoire totale: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Mémoire allouée: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
        print(f"Mémoire en cache: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")
    else:
        print("Pas de GPU disponible")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Compte les paramètres d'un modèle PyTorch.
    
    Args:
        model: Modèle PyTorch
        
    Returns:
        Dictionnaire avec le nombre de paramètres
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable,
    }


def memory_usage_mb() -> float:
    """
    Retourne l'utilisation mémoire actuelle en MB.
    
    Returns:
        Utilisation mémoire en MB
    """
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class Timer:
    """Context manager pour mesurer le temps d'exécution."""
    
    def __init__(self, name: str = "Opération"):
        """
        Initialise le timer.
        
        Args:
            name: Nom de l'opération à mesurer
        """
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        import time
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        logger.info(f"{self.name}: {format_time(self.elapsed)}")
    
    @property
    def elapsed_seconds(self) -> float:
        """Retourne le temps écoulé en secondes."""
        return self.elapsed if self.end_time else 0


def main():
    """Fonction de test des utilitaires."""
    print("Test des utilitaires")
    print("=" * 50)
    
    # Test set_seed
    set_seed(42)
    print(f"Random: {random.random():.4f}")
    print(f"Numpy: {np.random.rand():.4f}")
    
    # Test device
    device = get_device()
    print(f"Device: {device}")
    
    # Test format_time
    print(f"10s: {format_time(10)}")
    print(f"90s: {format_time(90)}")
    print(f"3700s: {format_time(3700)}")
    
    # Test Timer
    import time
    with Timer("Test sleep") as t:
        time.sleep(0.5)
    print(f"Elapsed: {t.elapsed_seconds:.2f}s")
    
    # Test memory
    print(f"Memory usage: {memory_usage_mb():.1f} MB")
    
    # Test project root
    print(f"Project root: {get_project_root()}")


if __name__ == "__main__":
    main()
