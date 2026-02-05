# PER - Benchmark ML vs Transformers pour Classification de Logs CI/CD

## But du Projet

Ce **Projet d'Étude et de Recherche (PER)** compare les performances de modèles **Machine Learning classiques** (Boosting) face aux modèles **Deep Learning** (Transformers) pour la **classification automatique de logs CI/CD**.

### Problématique

Dans les pipelines CI/CD, les échecs de tests peuvent être :
- **Flaky (Classe 0)** : Échecs intermittents dus à des problèmes d'infrastructure (timeouts, race conditions, ressources temporairement indisponibles)
- **Régression (Classe 1)** : Vrais bugs introduits par le code (NullPointerException, erreurs logiques, crashes)

**Objectif** : Déterminer quelle approche (ML vs DL) est la plus adaptée pour classifier automatiquement ces logs.

### Modèles Comparés

| Type | Modèles | Caractéristiques |
|------|---------|------------------|
| **ML (Boosting)** | XGBoost, AdaBoost, GradientBoosting | Rapides, interprétables, features TF-IDF |
| **DL (Transformers)** | BERT, DistilBERT, RoBERTa | Compréhension contextuelle, pré-entraînés |

---

## Architecture du Code

```
PER/
├── config/
│   └── config.yaml                 # Configuration centralisée (hyperparamètres, chemins)
│
├── data/
│   ├── ci_logs_dataset.csv         # Dataset principal (5000 logs)
│   ├── ci_logs_dataset_large.csv   # Dataset large (10000 logs)
│   ├── small_dataset/              # Logs format simple
│   └── large_dataset/              # Logs format étendu
│
├── src/                            # Code source modulaire
│   ├── data/
│   │   ├── loader.py               # Chargement CSV/texte
│   │   ├── preprocessor.py         # Nettoyage (timestamps, IPs, UUIDs...)
│   │   ├── labeler.py              # Labellisation automatique
│   │   └── generator.py            # Génération de logs synthétiques
│   │
│   ├── features/
│   │   ├── tfidf_extractor.py      # Vectorisation TF-IDF pour ML
│   │   └── transformer_tokenizer.py # Tokenisation pour Transformers
│   │
│   ├── models/
│   │   ├── ml/                     # Modèles Machine Learning
│   │   │   ├── xgboost_model.py
│   │   │   ├── adaboost_model.py
│   │   │   └── gradient_boosting_model.py
│   │   │
│   │   └── transformers/           # Modèles Deep Learning
│   │       ├── base_transformer.py # Classe abstraite commune
│   │       ├── bert_model.py
│   │       ├── distilbert_model.py
│   │       └── roberta_model.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py              # Accuracy, F1, Precision, Recall, AUC
│   │   └── comparator.py           # Comparaison multi-modèles
│   │
│   └── utils/
│       └── helpers.py              # Fonctions utilitaires
│
├── scripts/                        # Scripts exécutables
│   ├── benchmark_ml.py             # Benchmark des modèles ML
│   ├── benchmark_transformers.py   # Benchmark des modèles DL
│   └── generate_dataset.py         # Génération de données synthétiques
│
├── results/                        # Résultats des benchmarks
│   ├── ml/
│   │   ├── benchmark_results.csv   # Métriques détaillées ML
│   │   ├── benchmark_scores.csv    # Scores pondérés ML
│   │   └── figures/                # Graphiques ML
│   │
│   └── dl/
│       ├── benchmark_results.csv   # Métriques détaillées DL
│       ├── benchmark_scores.csv    # Scores pondérés DL
│       └── figures/                # Graphiques DL
│
└── requirements.txt                # Dépendances Python
```

---

## Lancer les Benchmarks

### Prérequis

- Python 3.9+
- pip

### Installation des dépendances

```bash
cd PER
pip install -r requirements.txt
```

### 1. Générer les Données (optionnel)

Si vous souhaitez régénérer les datasets synthétiques :

```bash
python scripts/generate_dataset.py --size 5000 --output data/ci_logs_dataset.csv
python scripts/generate_dataset.py --size 10000 --output data/ci_logs_dataset_large.csv
```

### 2. Benchmark ML (Boosting)

```bash
python scripts/benchmark_ml.py --data data/ci_logs_dataset.csv
```

**Sortie** : 
- `results/ml/benchmark_results.csv` - Métriques détaillées
- `results/ml/benchmark_scores.csv` - Scores pondérés
- `results/ml/figures/` - Graphiques de comparaison

### 3. Benchmark Deep Learning (Transformers)

```bash
python scripts/benchmark_transformers.py
```

**Note** : Ce benchmark est plus long (~10-30 min selon GPU/CPU). Il entraîne BERT, DistilBERT et RoBERTa sur les deux datasets.

**Sortie** :
- `results/dl/benchmark_results.csv` - Métriques détaillées
- `results/dl/benchmark_scores.csv` - Scores pondérés
- `results/dl/figures/` - Graphiques de comparaison

---

## Résultats Obtenus

### Machine Learning (Boosting)

#### Performances globales

| Modèle | Train Acc | Test Acc | Test F1 | Precision | Recall | Overfit Gap |
|--------|-----------|----------|---------|-----------|--------|-------------|
| XGBoost | 0.8643 | 0.7810 | **0.7872** | 0.7759 | 0.7988 | 0.0832 |
| Gradient Boosting | 0.7873 | 0.7750 | 0.7796 | 0.7743 | 0.7850 | 0.0122 |
| AdaBoost | 0.7020 | 0.6990 | 0.6760 | 0.7441 | 0.6193 | 0.0030 |

#### Validation croisée

| Modèle | CV F1 Mean | CV F1 Std | Interprétation |
|--------|------------|-----------|----------------|
| XGBoost | 0.7830 | 0.0153 | Performant et stable |
| Gradient Boosting | 0.7752 | 0.0196 | Stable |
| AdaBoost | 0.6504 | 0.0620 | Instable |

### Deep Learning (Transformers)

| Modèle | Accuracy | F1-Score | Temps Train | Paramètres |
|--------|----------|----------|-------------|------------|
| **BERT** | 80.5% | **80.5%** | 217s | 109M |
| RoBERTa | 79.7% | 79.7% | 227s | 125M |
| DistilBERT | 79.5% | 79.5% | 114s | 67M |

### Conclusion

- **DL (Transformers)** : +2% de F1-Score, mais 1000x plus lent
- **ML (XGBoost)** : Excellent rapport performance/coût
- **Recommandation** : XGBoost pour la production, BERT pour la précision maximale

---

## Métriques Évaluées

| Métrique | Description |
|----------|-------------|
| **Accuracy** | Taux de classification correcte |
| **F1-Score** | Moyenne harmonique Precision/Recall |
| **Precision** | Vrais positifs / Prédictions positives |
| **Recall** | Vrais positifs / Réels positifs |
| **Overfit Gap** | Écart train/test (généralisation) |
| **CV F1** | F1 en cross-validation (stabilité) |
| **Temps d'entraînement** | Durée du training |
| **Mémoire** | Consommation RAM/VRAM |

---

## Configuration

Le fichier `config/config.yaml` centralise tous les hyperparamètres :

```yaml
# Extrait de la configuration
ml:
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1

transformers:
  common:
    max_length: 256
    batch_size: 16
    epochs: 3
    learning_rate: 2e-5
```

