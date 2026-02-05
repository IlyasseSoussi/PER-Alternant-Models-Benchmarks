"""
BERT Model - Classification de logs avec BERT
=============================================

Implémentation du modèle BERT pour la classification binaire
de logs CI/CD.
"""

from typing import Optional
import logging
from pathlib import Path

from .base_transformer import BaseTransformerClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BertClassifier(BaseTransformerClassifier):
    """
    Classifieur basé sur BERT pour la classification de logs CI/CD.
    
    BERT (Bidirectional Encoder Representations from Transformers) est
    particulièrement adapté pour la compréhension du contexte bidirectionnel
    dans les logs.
    
    Attributes:
        model_name: Nom du modèle Hugging Face (bert-base-uncased par défaut)
    """
    
    DEFAULT_MODEL_NAME = "bert-base-uncased"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        num_labels: int = 2,
        config_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialise le classifieur BERT.
        
        Args:
            model_name: Nom du modèle (défaut: bert-base-uncased)
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
                if 'transformers' in yaml_config and 'bert' in yaml_config['transformers']:
                    bert_config = yaml_config['transformers']['bert']
                    if 'model_name' in bert_config:
                        model_name = bert_config['model_name']
        
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            config_path=config_path,
            device=device
        )
        
        self.model_display_name = "BERT"
        logger.info(f"BertClassifier initialisé avec {model_name}")
    
    def get_model_info(self) -> dict:
        """
        Retourne des informations sur le modèle.
        
        Returns:
            Dictionnaire d'informations
        """
        return {
            'model_name': self.model_name,
            'model_type': 'BERT',
            'num_labels': self.num_labels,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'max_length': self.config.get('max_length', 256),
            'vocab_size': self.tokenizer.vocab_size,
        }


def main():
    """Fonction de test du modèle BERT."""
    print("Test du BertClassifier")
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
    ] * 10  # Répéter pour avoir plus de données
    
    train_labels = [1, 0, 0, 0, 1, 0, 1, 0] * 10
    
    test_texts = [
        "NullPointerException caused test failure",
        "Timeout during network call, retrying",
    ]
    test_labels = [1, 0]
    
    # Créer et entraîner
    try:
        classifier = BertClassifier()
        
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
        
        print("\nPrédictions:")
        preds = classifier.predict(test_texts)
        for text, pred in zip(test_texts, preds):
            label = "Regression" if pred == 1 else "Flaky"
            print(f"  '{text[:40]}...' -> {label}")
            
    except Exception as e:
        print(f"Erreur (probablement pas de GPU ou modèle non téléchargé): {e}")


if __name__ == "__main__":
    main()
