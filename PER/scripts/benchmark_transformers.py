"""Benchmark Deep Learning - BERT vs DistilBERT vs RoBERTa pour classification de logs CI/CD"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import psutil

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configuration
SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

np.random.seed(SEED)
torch.manual_seed(SEED)

class LogDataset(Dataset):
    """Dataset pour les logs CI/CD"""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding=True, 
                                   max_length=max_length, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

def compute_metrics(eval_pred):
    """Calcul des métriques d'évaluation"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted'),
        'precision': precision_score(labels, preds, average='weighted'),
        'recall': recall_score(labels, preds, average='weighted')
    }

def calculate_token_loss(texts, tokenizer, max_length=512):
    """Calcule le pourcentage de tokens perdus par troncation"""
    total_tokens = 0
    truncated_tokens = 0
    samples_truncated = 0
    
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        original_len = len(tokens)
        total_tokens += original_len
        
        if original_len > max_length:
            samples_truncated += 1
            truncated_tokens += (original_len - max_length)
    
    pct_samples_truncated = 100 * samples_truncated / len(texts)
    pct_tokens_lost = 100 * truncated_tokens / total_tokens if total_tokens > 0 else 0
    
    return {
        'samples_truncated': samples_truncated,
        'pct_samples_truncated': pct_samples_truncated,
        'total_tokens': total_tokens,
        'tokens_lost': truncated_tokens,
        'pct_tokens_lost': pct_tokens_lost
    }

def load_and_prepare_data(data_path):
    """Charge et prépare les données textuelles"""
    df = pd.read_csv(data_path)
    
    # Combine les features textuelles
    df['text'] = (df['log_message'].fillna('') + ' [SEP] ' + 
                  df['error_type'].fillna('') + ' [SEP] ' + 
                  df['stack_trace'].fillna(''))
    
    texts = df['text'].tolist()
    labels = df['is_flaky'].astype(int).tolist()
    
    # Split train/val/test
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=SEED, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=SEED, stratify=temp_labels
    )
    
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

def benchmark_model(model_name, model_id, train_texts, val_texts, test_texts, 
                    train_labels, val_labels, test_labels, output_dir):
    """Benchmark un modèle Transformer"""
    print(f"\n   [MODEL] {model_name}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mem_before = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Tokenizer et modèle
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    model.to(device)
    
    # Datasets
    train_dataset = LogDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = LogDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    test_dataset = LogDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'{output_dir}/{model_name}',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        logging_steps=50,
        seed=SEED,
        report_to='none',
        disable_tqdm=True
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Training
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    
    # Évaluation train
    train_results = trainer.evaluate(train_dataset)
    
    # Évaluation test
    start_inference = time.time()
    test_results = trainer.evaluate(test_dataset)
    inference_time = (time.time() - start_inference) / len(test_labels) * 1000
    
    mem_after = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Nettoyage mémoire
    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    results = {
        'model_name': model_name,
        'train_accuracy': train_results['eval_accuracy'],
        'train_f1': train_results['eval_f1'],
        'test_accuracy': test_results['eval_accuracy'],
        'test_f1': test_results['eval_f1'],
        'test_precision': test_results['eval_precision'],
        'test_recall': test_results['eval_recall'],
        'overfit_gap': train_results['eval_accuracy'] - test_results['eval_accuracy'],
        'train_time_sec': train_time,
        'inference_time_ms': inference_time,
        'memory_mb': mem_after - mem_before,
        'num_parameters': sum(p.numel() for p in AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2).parameters()) / 1e6
    }
    
    print(f"      Train Acc: {results['train_accuracy']:.4f} | Test Acc: {results['test_accuracy']:.4f}")
    print(f"      Test F1: {results['test_f1']:.4f} | Train Time: {train_time:.1f}s")
    print(f"      Overfit Gap: {results['overfit_gap']:.4f}")
    
    return results

def generate_plots(results_df, output_dir):
    """Génère les graphiques de comparaison"""
    fig_dir = f'{output_dir}/figures'
    os.makedirs(fig_dir, exist_ok=True)
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    models = results_df['model_name'].tolist()
    
    # 1. Comparaison F1
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, results_df['train_f1'], width, label='Train F1', color=colors[0], alpha=0.8)
    ax.bar(x + width/2, results_df['test_f1'], width, label='Test F1', color=colors[1], alpha=0.8)
    ax.set_ylabel('F1 Score')
    ax.set_title('Comparaison F1 - Modèles Transformers')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/dl_f1_comparison.png', dpi=150)
    plt.close()
    
    # 2. Temps d'entraînement
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(models, results_df['train_time_sec'], color=colors)
    ax.set_ylabel('Temps (secondes)')
    ax.set_title('Temps d\'entraînement - Modèles Transformers')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/dl_train_time.png', dpi=150)
    plt.close()
    
    # 3. Nombre de paramètres vs F1
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(results_df['num_parameters'], results_df['test_f1'], s=200, c=colors)
    for i, model in enumerate(models):
        ax.annotate(model, (results_df['num_parameters'].iloc[i], results_df['test_f1'].iloc[i]),
                   textcoords="offset points", xytext=(0,10), ha='center')
    ax.set_xlabel('Paramètres (millions)')
    ax.set_ylabel('Test F1')
    ax.set_title('Efficacité: Paramètres vs Performance')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/dl_params_vs_f1.png', dpi=150)
    plt.close()
    
    # 4. Overfitting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(models, results_df['overfit_gap'], color=colors)
    ax.set_ylabel('Overfit Gap (Train - Test)')
    ax.set_title('Écart d\'overfitting - Modèles Transformers')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/dl_overfitting.png', dpi=150)
    plt.close()
    
    print(f"[GRAPHS] Graphiques generes dans {fig_dir}")

def run_benchmark(data_path=None, dataset_name="default"):
    """Exécute le benchmark complet"""
    print("=" * 70)
    print(f"BENCHMARK DL - BERT vs DistilBERT vs RoBERTa ({dataset_name})")
    print("=" * 70)
    
    # Chemins
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_path is None:
        data_path = os.path.join(base_dir, 'data', 'ci_logs_dataset.csv')
    output_dir = os.path.join(base_dir, 'results', 'dl', dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[DEVICE] Device: {device}")
    
    # Chargement données
    print("[DATA] Chargement du dataset...")
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = load_and_prepare_data(data_path)
    print(f"   Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")
    
    # Calcul perte de tokens (utiliser BERT tokenizer comme référence)
    from transformers import AutoTokenizer
    ref_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    all_texts = train_texts + val_texts + test_texts
    token_loss_info = calculate_token_loss(all_texts, ref_tokenizer, max_length=512)
    
    print(f"\n[TRUNCATION] ANALYSE TRONCATION (limite 512 tokens):")
    print(f"   Echantillons tronques: {token_loss_info['samples_truncated']}/{len(all_texts)} ({token_loss_info['pct_samples_truncated']:.1f}%)")
    print(f"   Tokens perdus: {token_loss_info['tokens_lost']:,}/{token_loss_info['total_tokens']:,} ({token_loss_info['pct_tokens_lost']:.1f}%)")
    
    # Modèles à tester
    models = {
        'BERT': 'bert-base-uncased',
        'DistilBERT': 'distilbert-base-uncased',
        'RoBERTa': 'roberta-base'
    }
    
    print("\n[RUN] Execution du benchmark...")
    results = []
    
    for model_name, model_id in models.items():
        result = benchmark_model(
            model_name, model_id,
            train_texts, val_texts, test_texts,
            train_labels, val_labels, test_labels,
            output_dir
        )
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Affichage résultats
    print("\n" + "=" * 70)
    print("RÉSULTATS DÉTAILLÉS")
    print("=" * 70)
    
    print("\nPERFORMANCE:")
    perf_cols = ['model_name', 'train_accuracy', 'test_accuracy', 'test_f1', 'test_precision', 'test_recall']
    print(results_df[perf_cols].to_string(index=False))
    
    print("\nOVERFITTING:")
    overfit_cols = ['model_name', 'train_accuracy', 'test_accuracy', 'overfit_gap']
    print(results_df[overfit_cols].to_string(index=False))
    
    print("\nEFFICACITE:")
    eff_cols = ['model_name', 'train_time_sec', 'inference_time_ms', 'memory_mb', 'num_parameters']
    print(results_df[eff_cols].to_string(index=False))
    
    print("\n[TRUNCATION] TRONCATION (limitation 512 tokens):")
    print(f"   Dataset: {dataset_name}")
    print(f"   Echantillons tronques: {token_loss_info['pct_samples_truncated']:.1f}%")
    print(f"   Tokens perdus: {token_loss_info['pct_tokens_lost']:.1f}%")
    if token_loss_info['pct_tokens_lost'] > 0:
        print(f"   [WARNING] IMPACT: Les modeles ne voient que {100 - token_loss_info['pct_tokens_lost']:.1f}% de l'information!")
    else:
        print(f"   [OK] Aucune perte: tous les tokens sont traites")
    
    # Analyse
    print("\n" + "=" * 70)
    print("ANALYSE")
    print("=" * 70)
    
    best_f1 = results_df.loc[results_df['test_f1'].idxmax()]
    best_speed = results_df.loc[results_df['train_time_sec'].idxmin()]
    best_overfit = results_df.loc[results_df['overfit_gap'].abs().idxmin()]
    smallest = results_df.loc[results_df['num_parameters'].idxmin()]
    
    print(f"\n[BEST] Meilleur F1 Test: {best_f1['model_name']} ({best_f1['test_f1']:.4f})")
    print(f"[SPEED] Plus rapide: {best_speed['model_name']} ({best_speed['train_time_sec']:.1f}s)")
    print(f"[FIT] Moins d'overfitting: {best_overfit['model_name']} (gap: {best_overfit['overfit_gap']:.4f})")
    print(f"[SIZE] Plus leger: {smallest['model_name']} ({smallest['num_parameters']:.1f}M params)")
    
    # Score global
    print("\n" + "=" * 70)
    print("SCORE GLOBAL PONDÉRÉ")
    print("=" * 70)
    
    def normalize(series, higher_better=True):
        if series.max() == series.min():
            return pd.Series([1.0] * len(series))
        if higher_better:
            return (series - series.min()) / (series.max() - series.min())
        return (series.max() - series) / (series.max() - series.min())
    
    # Ajouter perte de tokens aux résultats
    results_df['pct_tokens_lost'] = token_loss_info['pct_tokens_lost']
    results_df['pct_samples_truncated'] = token_loss_info['pct_samples_truncated']
    
    scores_df = pd.DataFrame({
        'model_name': results_df['model_name'],
        'score_f1_test': normalize(results_df['test_f1']),
        'score_overfit': normalize(results_df['overfit_gap'].abs(), False),
        'score_speed': normalize(results_df['train_time_sec'], False),
        'score_memory': normalize(results_df['memory_mb'].abs(), False),
    })
    
    # Pénalité pour perte de tokens (même pour tous les modèles, c'est une limitation commune)
    token_loss_penalty = token_loss_info['pct_tokens_lost'] / 100  # entre 0 et 1
    scores_df['score_token_loss'] = 1.0 - token_loss_penalty  # 1.0 = pas de perte, 0 = 100% perte
    
    # Pondération avec perte de tokens
    weights = {'score_f1_test': 0.40, 'score_memory': 0.20, 'score_speed': 0.10, 'score_overfit': 0.10, 'score_token_loss': 0.20}
    
    scores_df['total_score'] = sum(scores_df[col] * w for col, w in weights.items())
    scores_df = scores_df.sort_values('total_score', ascending=False)
    
    print(f"\n[WARNING] PERTE DE TOKENS: {token_loss_info['pct_tokens_lost']:.1f}% (penalite appliquee a tous les modeles)")
    print("\nPondération: F1 (40%), Perte tokens (20%), Mémoire (20%), Vitesse (10%), Overfitting (10%)")
    print(scores_df.to_string(index=False))
    
    winner = scores_df.iloc[0]['model_name']
    print(f"\n[WINNER] GAGNANT: {winner} (score: {scores_df.iloc[0]['total_score']:.3f})")
    
    # Génération graphiques
    generate_plots(results_df, output_dir)
    
    # Sauvegarde
    results_df.to_csv(f'{output_dir}/benchmark_results.csv', index=False)
    scores_df.to_csv(f'{output_dir}/benchmark_scores.csv', index=False)
    print(f"\n[SAVED] Resultats sauvegardes dans {output_dir}")
    
    return results_df, scores_df, winner

if __name__ == '__main__':
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    datasets = {
        'small_5k': os.path.join(base_dir, 'data', 'ci_logs_dataset.csv'),
        'large_10k': os.path.join(base_dir, 'data', 'ci_logs_dataset_large.csv')
    }
    
    all_winners = {}
    
    for name, path in datasets.items():
        print("\n" + "=" * 70)
        print(f"[DATASET] DATASET: {name.upper()}")
        print("=" * 70 + "\n")
        _, _, winner = run_benchmark(data_path=path, dataset_name=name)
        all_winners[name] = winner
    
    print("\n" + "=" * 70)
    print("RÉSUMÉ FINAL - TOUS LES DATASETS")
    print("=" * 70)
    for name, winner in all_winners.items():
        print(f"   {name}: [WINNER] {winner}")
    print("=" * 70)
