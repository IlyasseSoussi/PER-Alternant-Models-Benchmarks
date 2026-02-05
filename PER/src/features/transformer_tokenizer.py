"""
Transformer Tokenizer - Tokenisation pour les modèles Deep Learning
===================================================================

Ce module gère la tokenisation des logs pour les modèles Transformers
(BERT, DistilBERT, RoBERTa, GPT).
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import Optional, Dict, Any, List, Union, Tuple
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogDataset(Dataset):
    """Dataset PyTorch pour les logs tokenisés."""
    
    def __init__(
        self,
        encodings: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ):
        """
        Initialise le dataset.
        
        Args:
            encodings: Dictionnaire avec 'input_ids', 'attention_mask'
            labels: Tenseur des labels (optionnel)
        """
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item


class TransformerTokenizer:
    """
    Tokenizer pour les modèles Transformers.
    
    Supporte: BERT, DistilBERT, RoBERTa, GPT-2
    """
    
    # Mapping des noms de modèles vers les noms Hugging Face
    MODEL_MAPPING = {
        'bert': 'bert-base-uncased',
        'distilbert': 'distilbert-base-uncased',
        'roberta': 'roberta-base',
        'gpt': 'gpt2',
        'gpt2': 'gpt2',
    }
    
    def __init__(
        self,
        model_name: str = 'bert',
        config_path: Optional[str] = None,
        max_length: int = 256
    ):
        """
        Initialise le tokenizer.
        
        Args:
            model_name: Nom du modèle ('bert', 'distilbert', 'roberta', 'gpt')
            config_path: Chemin vers le fichier de configuration
            max_length: Longueur maximale des séquences
        """
        self.config = self._load_config(config_path)
        self.model_name = model_name.lower()
        self.max_length = self.config.get('max_length', max_length)
        
        # Résoudre le nom du modèle
        self.hf_model_name = self._resolve_model_name()
        
        # Charger le tokenizer
        self.tokenizer = self._load_tokenizer()
        
        logger.info(f"Tokenizer chargé: {self.hf_model_name}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Charge la configuration."""
        default_config = {
            'max_length': 256,
            'batch_size': 16,
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                if 'transformers' in yaml_config:
                    if 'common' in yaml_config['transformers']:
                        default_config.update(yaml_config['transformers']['common'])
        
        return default_config
    
    def _resolve_model_name(self) -> str:
        """Résout le nom du modèle Hugging Face."""
        if self.model_name in self.MODEL_MAPPING:
            return self.MODEL_MAPPING[self.model_name]
        
        # Supposer que c'est déjà un nom HF complet
        return self.model_name
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Charge le tokenizer depuis Hugging Face."""
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        
        # GPT-2 n'a pas de pad token par défaut
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def tokenize(
        self,
        texts: Union[List[str], pd.Series],
        return_tensors: bool = True
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Tokenise une liste de textes.
        
        Args:
            texts: Liste de textes à tokeniser
            return_tensors: Retourner des tenseurs PyTorch
            
        Returns:
            Dictionnaire avec 'input_ids' et 'attention_mask'
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # Tokenisation
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt' if return_tensors else None,
        )
        
        return encodings
    
    def create_dataset(
        self,
        texts: Union[List[str], pd.Series],
        labels: Optional[Union[List[int], pd.Series, np.ndarray]] = None
    ) -> LogDataset:
        """
        Crée un Dataset PyTorch.
        
        Args:
            texts: Textes à tokeniser
            labels: Labels correspondants (optionnel)
            
        Returns:
            LogDataset
        """
        encodings = self.tokenize(texts, return_tensors=True)
        
        if labels is not None:
            if isinstance(labels, (pd.Series, np.ndarray)):
                labels = labels.tolist() if isinstance(labels, pd.Series) else labels.tolist()
            labels = torch.tensor(labels, dtype=torch.long)
        
        return LogDataset(encodings, labels)
    
    def create_dataloader(
        self,
        texts: Union[List[str], pd.Series],
        labels: Optional[Union[List[int], pd.Series]] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Crée un DataLoader PyTorch.
        
        Args:
            texts: Textes
            labels: Labels
            batch_size: Taille des batches
            shuffle: Mélanger les données
            
        Returns:
            DataLoader
        """
        dataset = self.create_dataset(texts, labels)
        batch_size = batch_size or self.config.get('batch_size', 16)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
    
    def prepare_train_val_test(
        self,
        train_texts: Union[List[str], pd.Series],
        train_labels: Union[List[int], pd.Series],
        val_texts: Union[List[str], pd.Series],
        val_labels: Union[List[int], pd.Series],
        test_texts: Union[List[str], pd.Series],
        test_labels: Union[List[int], pd.Series],
        batch_size: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prépare les DataLoaders pour train/val/test.
        
        Returns:
            Tuple (train_loader, val_loader, test_loader)
        """
        batch_size = batch_size or self.config.get('batch_size', 16)
        
        train_loader = self.create_dataloader(
            train_texts, train_labels, batch_size, shuffle=True
        )
        val_loader = self.create_dataloader(
            val_texts, val_labels, batch_size, shuffle=False
        )
        test_loader = self.create_dataloader(
            test_texts, test_labels, batch_size, shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def decode(self, input_ids: torch.Tensor) -> List[str]:
        """
        Décode des input_ids en texte.
        
        Args:
            input_ids: Tenseur d'IDs
            
        Returns:
            Liste de textes décodés
        """
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    
    def get_vocab_size(self) -> int:
        """Retourne la taille du vocabulaire."""
        return self.tokenizer.vocab_size
    
    def get_special_tokens(self) -> Dict[str, str]:
        """Retourne les tokens spéciaux."""
        return {
            'pad_token': self.tokenizer.pad_token,
            'unk_token': self.tokenizer.unk_token,
            'cls_token': getattr(self.tokenizer, 'cls_token', None),
            'sep_token': getattr(self.tokenizer, 'sep_token', None),
            'mask_token': getattr(self.tokenizer, 'mask_token', None),
            'bos_token': getattr(self.tokenizer, 'bos_token', None),
            'eos_token': getattr(self.tokenizer, 'eos_token', None),
        }
    
    def get_stats(self, texts: List[str]) -> Dict[str, Any]:
        """
        Calcule des statistiques sur les textes tokenisés.
        
        Args:
            texts: Liste de textes
            
        Returns:
            Dictionnaire de statistiques
        """
        encodings = self.tokenize(texts, return_tensors=False)
        
        lengths = [len(ids) for ids in encodings['input_ids']]
        
        return {
            'model_name': self.hf_model_name,
            'vocab_size': self.get_vocab_size(),
            'max_length_config': self.max_length,
            'num_texts': len(texts),
            'avg_length': np.mean(lengths),
            'max_length_actual': max(lengths),
            'min_length_actual': min(lengths),
            'truncated_count': sum(1 for l in lengths if l == self.max_length),
        }


def main():
    """Fonction de test du tokenizer."""
    print("Test du TransformerTokenizer")
    print("=" * 50)
    
    # Corpus de test
    corpus = [
        "Test failed with NullPointerException in UserService",
        "Connection timeout, retrying operation",
        "Build successful, all tests passed",
        "FLAKY: Test intermittently fails due to race condition",
    ]
    labels = [1, 0, 0, 0]
    
    # Tester chaque modèle
    for model_name in ['bert', 'distilbert', 'roberta', 'gpt']:
        print(f"\n{'='*50}")
        print(f"Testing: {model_name}")
        print("=" * 50)
        
        try:
            tokenizer = TransformerTokenizer(model_name=model_name, max_length=128)
            
            # Stats
            stats = tokenizer.get_stats(corpus)
            print(f"Stats: {stats}")
            
            # Créer un dataset
            dataset = tokenizer.create_dataset(corpus, labels)
            print(f"Dataset size: {len(dataset)}")
            
            # Créer un dataloader
            dataloader = tokenizer.create_dataloader(corpus, labels, batch_size=2)
            print(f"Number of batches: {len(dataloader)}")
            
            # Afficher un batch
            batch = next(iter(dataloader))
            print(f"Batch keys: {batch.keys()}")
            print(f"Input IDs shape: {batch['input_ids'].shape}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nTest terminé!")


if __name__ == "__main__":
    main()
