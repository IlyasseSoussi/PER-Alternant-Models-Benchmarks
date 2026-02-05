# src/features/__init__.py
"""
Module d'extraction de features
===============================

Ce module contient :
- Extraction TF-IDF pour les modèles ML
- Tokenisation pour les modèles Transformers
"""

from .tfidf_extractor import TFIDFExtractor
from .transformer_tokenizer import TransformerTokenizer

__all__ = [
    "TFIDFExtractor",
    "TransformerTokenizer"
]
