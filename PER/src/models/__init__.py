# src/models/__init__.py
"""
Module des modèles
==================

Ce module contient les implémentations des modèles :
- ML (Boosting): XGBoost, AdaBoost, Gradient Boosting
- Transformers: BERT, DistilBERT, RoBERTa
"""

from .ml import XGBoostClassifier, AdaBoostClassifier, GradientBoostingModel
from .transformers import (
    BertClassifier,
    DistilBertClassifier,
    RobertaClassifier,
)

__all__ = [
    # ML
    "XGBoostClassifier",
    "AdaBoostClassifier",
    "GradientBoostingModel",
    # Transformers
    "BertClassifier",
    "DistilBertClassifier",
    "RobertaClassifier",
]
