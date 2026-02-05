# src/models/ml/__init__.py
"""
Module des modèles Machine Learning (Boosting)
==============================================

Implémentation des algorithmes de boosting pour la classification de logs.
"""

from .xgboost_model import XGBoostClassifier
from .adaboost_model import AdaBoostClassifier
from .gradient_boosting_model import GradientBoostingModel

__all__ = [
    "XGBoostClassifier",
    "AdaBoostClassifier",
    "GradientBoostingModel"
]
