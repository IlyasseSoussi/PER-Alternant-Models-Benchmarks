"""
Base Transformer Classifier - Classe de base pour les modèles Transformers
=========================================================================

Ce module fournit une classe de base abstraite pour tous les
classifieurs Transformer (BERT, DistilBERT, RoBERTa, GPT).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    PreTrainedModel,
    PreTrainedTokenizer
)
from typing import Optional, Dict, Any, List, Union, Tuple
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import yaml
import time
import psutil
import os
from abc import ABC, abstractmethod
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTransformerClassifier(ABC):
    """
    Classe de base abstraite pour les classifieurs Transformer.
    
    Cette classe fournit l'infrastructure commune pour:
    - Le chargement des modèles
    - L'entraînement
    - L'évaluation
    - La sauvegarde/chargement
    """
    
    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        config_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialise le classifieur.
        
        Args:
            model_name: Nom du modèle Hugging Face
            num_labels: Nombre de classes
            config_path: Chemin vers le fichier de configuration
            device: Device PyTorch ('cuda' ou 'cpu')
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.config = self._load_config(config_path)
        
        # Déterminer le device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Device: {self.device}")
        
        # Charger le modèle et le tokenizer
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.model.to(self.device)
        
        self.is_fitted = False
        self.training_stats = {}
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Charge la configuration."""
        default_config = {
            'max_length': 256,
            'batch_size': 16,
            'epochs': 3,
            'learning_rate': 2e-5,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'seed': 42,
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                if 'transformers' in yaml_config:
                    if 'common' in yaml_config['transformers']:
                        default_config.update(yaml_config['transformers']['common'])
        
        return default_config
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Charge le tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Gérer le pad token pour GPT-2
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def _load_model(self) -> PreTrainedModel:
        """Charge le modèle."""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        
        # Pour GPT-2, définir le pad token id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = self.tokenizer.pad_token_id
        
        return model
    
    def _get_memory_usage(self) -> float:
        """Retourne l'utilisation mémoire en MB."""
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 / 1024
        
        if torch.cuda.is_available():
            mem += torch.cuda.memory_allocated() / 1024 / 1024
        
        return mem
    
    def _tokenize(
        self,
        texts: List[str],
        max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Tokenise une liste de textes."""
        max_length = max_length or self.config.get('max_length', 256)
        
        encodings = self.tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encodings
    
    def _create_dataloader(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = True
    ) -> DataLoader:
        """Crée un DataLoader."""
        batch_size = batch_size or self.config.get('batch_size', 16)
        
        encodings = self._tokenize(texts)
        
        if labels is not None:
            labels = torch.tensor(labels, dtype=torch.long)
            dataset = torch.utils.data.TensorDataset(
                encodings['input_ids'],
                encodings['attention_mask'],
                labels
            )
        else:
            dataset = torch.utils.data.TensorDataset(
                encodings['input_ids'],
                encodings['attention_mask']
            )
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def fit(
        self,
        train_texts: Union[List[str], pd.Series],
        train_labels: Union[List[int], pd.Series],
        val_texts: Optional[Union[List[str], pd.Series]] = None,
        val_labels: Optional[Union[List[int], pd.Series]] = None,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        verbose: bool = True
    ) -> 'BaseTransformerClassifier':
        """
        Entraîne le modèle.
        
        Args:
            train_texts: Textes d'entraînement
            train_labels: Labels d'entraînement
            val_texts: Textes de validation (optionnel)
            val_labels: Labels de validation (optionnel)
            epochs: Nombre d'époques
            learning_rate: Taux d'apprentissage
            batch_size: Taille des batches
            verbose: Afficher les logs
            
        Returns:
            self
        """
        # Convertir en listes si nécessaire
        if isinstance(train_texts, pd.Series):
            train_texts = train_texts.tolist()
        if isinstance(train_labels, pd.Series):
            train_labels = train_labels.tolist()
        if val_texts is not None and isinstance(val_texts, pd.Series):
            val_texts = val_texts.tolist()
        if val_labels is not None and isinstance(val_labels, pd.Series):
            val_labels = val_labels.tolist()
        
        # Paramètres
        epochs = epochs or self.config.get('epochs', 3)
        learning_rate = learning_rate or self.config.get('learning_rate', 2e-5)
        batch_size = batch_size or self.config.get('batch_size', 16)
        
        if verbose:
            logger.info(f"Entraînement {self.__class__.__name__} sur {len(train_texts)} échantillons...")
            logger.info(f"Epochs: {epochs}, LR: {learning_rate}, Batch size: {batch_size}")
        
        # Créer les DataLoaders
        train_loader = self._create_dataloader(train_texts, train_labels, batch_size, shuffle=True)
        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_loader = self._create_dataloader(val_texts, val_labels, batch_size, shuffle=False)
        
        # Optimizer et scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * self.config.get('warmup_ratio', 0.1))
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Mesures
        start_time = time.time()
        mem_before = self._get_memory_usage()
        
        # Entraînement
        self.model.train()
        training_losses = []
        val_scores = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else train_loader
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            training_losses.append(avg_loss)
            
            # Validation
            if val_loader is not None:
                val_metrics = self._evaluate_loader(val_loader)
                val_scores.append(val_metrics['f1'])
                
                if verbose:
                    logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val F1={val_metrics['f1']:.4f}")
            else:
                if verbose:
                    logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
        
        # Statistiques
        training_time = time.time() - start_time
        mem_after = self._get_memory_usage()
        
        self.training_stats = {
            'training_time_seconds': training_time,
            'memory_usage_mb': mem_after - mem_before,
            'n_samples': len(train_texts),
            'epochs': epochs,
            'final_loss': training_losses[-1],
            'training_losses': training_losses,
            'val_scores': val_scores if val_loader else None,
        }
        
        self.is_fitted = True
        
        if verbose:
            logger.info(f"Entraînement terminé en {training_time:.2f}s")
        
        return self
    
    def _evaluate_loader(self, dataloader: DataLoader) -> Dict[str, float]:
        """Évalue sur un DataLoader."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2]
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        self.model.train()
        
        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='binary', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='binary', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='binary', zero_division=0),
        }
    
    def predict(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Prédit les classes.
        
        Args:
            texts: Textes à classifier
            
        Returns:
            Prédictions
        """
        if not self.is_fitted:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez fit() d'abord.")
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        self.model.eval()
        all_preds = []
        
        dataloader = self._create_dataloader(texts, batch_size=self.config.get('batch_size', 16), shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
        
        return np.array(all_preds)
    
    def predict_proba(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Prédit les probabilités.
        
        Args:
            texts: Textes à classifier
            
        Returns:
            Probabilités par classe
        """
        if not self.is_fitted:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez fit() d'abord.")
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        self.model.eval()
        all_probs = []
        
        dataloader = self._create_dataloader(texts, batch_size=self.config.get('batch_size', 16), shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                all_probs.extend(probs)
        
        return np.array(all_probs)
    
    def evaluate(
        self,
        texts: Union[List[str], pd.Series],
        labels: Union[List[int], pd.Series]
    ) -> Dict[str, float]:
        """
        Évalue le modèle.
        
        Args:
            texts: Textes
            labels: Labels
            
        Returns:
            Dictionnaire des métriques
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        if isinstance(labels, pd.Series):
            labels = labels.tolist()
        
        # Mesure du temps d'inférence
        start_time = time.time()
        y_pred = self.predict(texts)
        inference_time = time.time() - start_time
        
        return {
            'accuracy': accuracy_score(labels, y_pred),
            'precision': precision_score(labels, y_pred, average='binary', zero_division=0),
            'recall': recall_score(labels, y_pred, average='binary', zero_division=0),
            'f1': f1_score(labels, y_pred, average='binary', zero_division=0),
            'inference_time_seconds': inference_time,
            'inference_time_per_sample_ms': (inference_time / len(labels)) * 1000,
        }
    
    def save(self, filepath: str) -> None:
        """Sauvegarde le modèle."""
        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le modèle et le tokenizer
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        
        # Sauvegarder les métadonnées
        import json
        metadata = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'config': self.config,
            'training_stats': self.training_stats,
            'is_fitted': self.is_fitted,
        }
        
        with open(filepath / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Modèle sauvegardé: {filepath}")
    
    def load(self, filepath: str) -> 'BaseTransformerClassifier':
        """Charge un modèle sauvegardé."""
        filepath = Path(filepath)
        
        # Charger les métadonnées
        import json
        with open(filepath / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.model_name = metadata['model_name']
        self.num_labels = metadata['num_labels']
        self.config = metadata['config']
        self.training_stats = metadata['training_stats']
        self.is_fitted = metadata['is_fitted']
        
        # Charger le modèle et le tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(filepath)
        self.tokenizer = AutoTokenizer.from_pretrained(filepath)
        self.model.to(self.device)
        
        logger.info(f"Modèle chargé: {filepath}")
        
        return self
    
    def get_embeddings(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Extrait les embeddings des textes.
        
        Args:
            texts: Textes
            
        Returns:
            Embeddings (matrice)
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        self.model.eval()
        all_embeddings = []
        
        dataloader = self._create_dataloader(texts, batch_size=self.config.get('batch_size', 16), shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                
                # Obtenir les hidden states
                outputs = self.model.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Utiliser le [CLS] token ou la moyenne
                if hasattr(outputs, 'last_hidden_state'):
                    # Prendre la moyenne des embeddings (excluant le padding)
                    mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    summed = (outputs.last_hidden_state * mask).sum(dim=1)
                    count = mask.sum(dim=1).clamp(min=1e-9)
                    embeddings = (summed / count).cpu().numpy()
                else:
                    embeddings = outputs[0][:, 0, :].cpu().numpy()
                
                all_embeddings.extend(embeddings)
        
        return np.array(all_embeddings)
