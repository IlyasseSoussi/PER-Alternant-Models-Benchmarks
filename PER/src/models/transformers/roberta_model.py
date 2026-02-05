"""
RoBERTa Model - Classification de logs avec RoBERTa
==================================================

Implémentation du modèle RoBERTa pour la classification binaire
de logs CI/CD. RoBERTa est une version robustement optimisée de BERT.
"""

from typing import Optional
import logging
from pathlib import Path

from .base_transformer import BaseTransformerClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobertaClassifier(BaseTransformerClassifier):
    """
    Classifieur basé sur RoBERTa pour la classification de logs CI/CD.
    
    RoBERTa (Robustly Optimized BERT Approach) améliore BERT avec:
    - Plus de données d'entraînement
    - Entraînement plus long
    - Pas de Next Sentence Prediction
    - Séquences plus longues
    - Batch size plus grand
    
    Avantages:
    - Meilleures performances que BERT sur de nombreuses tâches
    - Plus robuste aux variations de texte
    
    Attributes:
        model_name: Nom du modèle Hugging Face (roberta-base par défaut)
    """
    
    DEFAULT_MODEL_NAME = "roberta-base"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        num_labels: int = 2,
        config_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialise le classifieur RoBERTa.
        
        Args:
            model_name: Nom du modèle (défaut: roberta-base)
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
                if 'transformers' in yaml_config and 'roberta' in yaml_config['transformers']:
                    roberta_config = yaml_config['transformers']['roberta']
                    if 'model_name' in roberta_config:
                        model_name = roberta_config['model_name']
        
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            config_path=config_path,
            device=device
        )
        
        self.model_display_name = "RoBERTa"
        logger.info(f"RobertaClassifier initialisé avec {model_name}")
    
    def get_model_info(self) -> dict:
        """
        Retourne des informations sur le modèle.
        
        Returns:
            Dictionnaire d'informations
        """
        return {
            'model_name': self.model_name,
            'model_type': 'RoBERTa',
            'num_labels': self.num_labels,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'max_length': self.config.get('max_length', 256),
            'vocab_size': self.tokenizer.vocab_size,
            'improvements_over_bert': [
                'Dynamic masking',
                'No NSP task',
                'Larger batch sizes',
                'More training data',
                'Longer training'
            ]
        }


def main():
    """Fonction de test du modèle RoBERTa."""
    print("Test du RobertaClassifier")
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
        classifier = RobertaClassifier()
        
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
