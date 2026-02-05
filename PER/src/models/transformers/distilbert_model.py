"""
DistilBERT Model - Classification de logs avec DistilBERT
========================================================

Implémentation du modèle DistilBERT pour la classification binaire
de logs CI/CD. DistilBERT est une version plus légère de BERT.
"""

from typing import Optional
import logging
from pathlib import Path

from .base_transformer import BaseTransformerClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistilBertClassifier(BaseTransformerClassifier):
    """
    Classifieur basé sur DistilBERT pour la classification de logs CI/CD.
    
    DistilBERT est une version distillée de BERT qui conserve 97% de sa
    performance tout en étant 60% plus rapide et 40% plus léger.
    
    Avantages:
    - Plus rapide à l'entraînement et à l'inférence
    - Moins de mémoire requise
    - Performances proches de BERT
    
    Attributes:
        model_name: Nom du modèle Hugging Face (distilbert-base-uncased par défaut)
    """
    
    DEFAULT_MODEL_NAME = "distilbert-base-uncased"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        num_labels: int = 2,
        config_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialise le classifieur DistilBERT.
        
        Args:
            model_name: Nom du modèle (défaut: distilbert-base-uncased)
            num_labels: Nombre de classes
            config_path: Chemin vers la configuration
            device: Device PyTorch
        """
        model_name = model_name or self.DEFAULT_MODEL_NAME
        
        # Charger la config pour voir si un modèle spécifique est défini
        if config_path and Path(config_path).exists():
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                if 'transformers' in yaml_config and 'distilbert' in yaml_config['transformers']:
                    distilbert_config = yaml_config['transformers']['distilbert']
                    if 'model_name' in distilbert_config:
                        model_name = distilbert_config['model_name']
        
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            config_path=config_path,
            device=device
        )
        
        self.model_display_name = "DistilBERT"
        logger.info(f"DistilBertClassifier initialisé avec {model_name}")
    
    def get_model_info(self) -> dict:
        """
        Retourne des informations sur le modèle.
        
        Returns:
            Dictionnaire d'informations
        """
        return {
            'model_name': self.model_name,
            'model_type': 'DistilBERT',
            'num_labels': self.num_labels,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'max_length': self.config.get('max_length', 256),
            'vocab_size': self.tokenizer.vocab_size,
            'distillation_info': {
                'compression_ratio': '~60%',
                'performance_retention': '~97%',
            }
        }


def main():
    """Fonction de test du modèle DistilBERT."""
    print("Test du DistilBertClassifier")
    print("=" * 50)
    
    # Données de test
    train_texts = [
        "Test failed with NullPointerException in UserService",
        "Connection timeout, retrying operation",
        "Build successful, all tests passed",
        "FLAKY: Test intermittently fails due to race condition",
        "REGRESSION: Expected 42 but got 0 in calculation",
        "Network error: connection refused to database server",
        "Fatal error: segmentation fault in memory allocator",
        "Test flapping, marked as unstable",
    ] * 10
    
    train_labels = [1, 0, 0, 0, 1, 0, 1, 0] * 10
    
    test_texts = [
        "NullPointerException caused test failure",
        "Timeout during network call, retrying",
    ]
    test_labels = [1, 0]
    
    try:
        classifier = DistilBertClassifier()
        
        print("\nInfo modèle:")
        info = classifier.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\nEntraînement (1 epoch pour le test)...")
        classifier.fit(
            train_texts, train_labels,
            epochs=1,
            batch_size=4,
            verbose=True
        )
        
        print("\nÉvaluation:")
        metrics = classifier.evaluate(test_texts, test_labels)
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")
            
    except Exception as e:
        print(f"Erreur: {e}")


if __name__ == "__main__":
    main()
