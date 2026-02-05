# src/evaluation/__init__.py
"""
Module d'évaluation
==================

Ce module contient les outils d'évaluation et de comparaison des modèles.
"""

from .metrics import MetricsCalculator
from .comparator import ModelComparator

__all__ = [
    "MetricsCalculator",
    "ModelComparator"
]
