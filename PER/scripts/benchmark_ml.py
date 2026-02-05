"""
Benchmark ML - XGBoost vs AdaBoost vs GradientBoosting
"""

import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import pandas as pd
import numpy as np
import time
import psutil
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def load_and_prepare_data(data_path, test_size=0.2, seed=42):
    """Charge et prepare le dataset."""
    print("[DATA] Chargement du dataset...")
    df = pd.read_csv(data_path)
    print(f"   Taille: {len(df)} echantillons, {len(df.columns)} colonnes")
    
    text_cols = ['log_message', 'stack_trace', 'test_name', 'error_type']
    bool_cols = ['is_timeout', 'is_network_error', 'has_retry', 'is_first_run', 
                 'parallel_execution', 'has_external_dependency']
    cat_cols = ['component', 'environment', 'severity', 'build_tool', 'os', 
                'language', 'ci_platform', 'trigger_type', 'branch_type', 'day_of_week']
    num_cols = ['duration_ms', 'retry_count', 'memory_mb', 'cpu_percent', 'line_number',
                'file_count', 'test_count', 'failure_rate_history', 'time_since_last_success',
                'network_latency_ms', 'db_response_time_ms', 'queue_size']
    
    print("[PREP] Preparation des features...")
    label_encoders = {}
    df_encoded = df.copy()
    
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    feature_cols = num_cols + cat_cols + bool_cols
    X = df_encoded[feature_cols].values
    y = df_encoded['is_flaky'].values
    
    print(f"   Features: {len(feature_cols)} (num: {len(num_cols)}, cat: {len(cat_cols)}, bool: {len(bool_cols)})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    scaler = StandardScaler()
    X_train[:, :len(num_cols)] = scaler.fit_transform(X_train[:, :len(num_cols)])
    X_test[:, :len(num_cols)] = scaler.transform(X_test[:, :len(num_cols)])
    
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, feature_cols


def get_memory_usage():
    """Retourne l'utilisation mémoire en MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_model(model, model_name, X_train, X_test, y_train, y_test, cv_folds=5):
    """Benchmark complet d'un modèle."""
    results = {'model_name': model_name}
    mem_before = get_memory_usage()
    
    print(f"\n   [MODEL] {model_name}...")
    
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    model.fit(X_train, y_train)
    results['train_cpu_time'] = time.process_time() - start_cpu
    results['train_wall_time'] = time.perf_counter() - start_wall
    results['memory_delta_mb'] = get_memory_usage() - mem_before
    
    y_train_pred = model.predict(X_train)
    results['train_accuracy'] = accuracy_score(y_train, y_train_pred)
    results['train_f1'] = f1_score(y_train, y_train_pred)
    
    start_inference = time.perf_counter()
    y_test_pred = model.predict(X_test)
    inference_time = time.perf_counter() - start_inference
    
    results['test_accuracy'] = accuracy_score(y_test, y_test_pred)
    results['test_f1'] = f1_score(y_test, y_test_pred)
    results['test_precision'] = precision_score(y_test, y_test_pred)
    results['test_recall'] = recall_score(y_test, y_test_pred)
    results['inference_time'] = inference_time
    results['inference_time_per_sample_ms'] = (inference_time / len(y_test)) * 1000
    results['overfit_gap'] = results['train_accuracy'] - results['test_accuracy']
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='f1')
    results['cv_f1_mean'] = cv_scores.mean()
    results['cv_f1_std'] = cv_scores.std()
    
    print(f"      Train Acc: {results['train_accuracy']:.4f} | Test Acc: {results['test_accuracy']:.4f}")
    print(f"      Test F1: {results['test_f1']:.4f} | CV F1: {results['cv_f1_mean']:.4f} ± {results['cv_f1_std']:.4f}")
    print(f"      Overfit Gap: {results['overfit_gap']:.4f}")
    
    return results


def run_benchmark(data_path, seed=42):
    """Exécute le benchmark complet."""
    
    print("=" * 70)
    print("BENCHMARK ML - XGBoost vs AdaBoost vs GradientBoosting")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test, feature_cols = load_and_prepare_data(
        data_path, test_size=0.2, seed=seed
    )
    
    models = {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.05, reg_lambda=1.0, min_child_weight=1, gamma=0.01,
            random_state=seed, eval_metric='logloss', use_label_encoder=False, n_jobs=-1
        ),
        'AdaBoost': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=25, learning_rate=2.0,
            random_state=seed, algorithm='SAMME'
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=20, max_depth=1, learning_rate=0.8,
            subsample=0.3, min_samples_split=50, min_samples_leaf=30,
            random_state=seed
        )
    }
    
    print("\n[RUN] Execution du benchmark...")
    all_results = []
    
    for name, model in models.items():
        results = benchmark_model(model, name, X_train, X_test, y_train, y_test)
        all_results.append(results)
    
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 70)
    print("RÉSULTATS DÉTAILLÉS")
    print("=" * 70)
    
    print("\nPERFORMANCE:")
    print(results_df[['model_name', 'train_accuracy', 'test_accuracy', 'test_f1', 'test_precision', 'test_recall']].to_string(index=False))
    
    print("\nOVERFITTING:")
    print(results_df[['model_name', 'train_accuracy', 'test_accuracy', 'overfit_gap']].to_string(index=False))
    
    print("\nSTABILITE (CV):")
    print(results_df[['model_name', 'cv_f1_mean', 'cv_f1_std']].to_string(index=False))
    
    print("\nEFFICACITE:")
    print(results_df[['model_name', 'train_cpu_time', 'inference_time_per_sample_ms', 'memory_delta_mb']].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("ANALYSE")
    print("=" * 70)
    
    best_f1 = results_df.loc[results_df['test_f1'].idxmax()]
    best_cv = results_df.loc[results_df['cv_f1_mean'].idxmax()]
    best_stability = results_df.loc[results_df['cv_f1_std'].idxmin()]
    best_overfit = results_df.loc[results_df['overfit_gap'].abs().idxmin()]
    best_speed = results_df.loc[results_df['train_cpu_time'].idxmin()]
    
    print(f"\n[BEST] Meilleur F1 Test: {best_f1['model_name']} ({best_f1['test_f1']:.4f})")
    print(f"[CV] Meilleur CV F1: {best_cv['model_name']} ({best_cv['cv_f1_mean']:.4f})")
    print(f"[STABLE] Plus stable: {best_stability['model_name']} (std: {best_stability['cv_f1_std']:.4f})")
    print(f"[FIT] Moins d'overfitting: {best_overfit['model_name']} (gap: {best_overfit['overfit_gap']:.4f})")
    print(f"[SPEED] Plus rapide: {best_speed['model_name']} ({best_speed['train_cpu_time']:.3f}s)")
    
    print("\n" + "=" * 70)
    print("SCORE GLOBAL PONDÉRÉ")
    print("=" * 70)
    
    def normalize(series, higher_is_better=True):
        min_val, max_val = series.min(), series.max()
        if max_val == min_val:
            return pd.Series([1.0] * len(series))
        normalized = (series - min_val) / (max_val - min_val)
        return normalized if higher_is_better else (1 - normalized)
    
    scores_df = pd.DataFrame({
        'model_name': results_df['model_name'],
        'score_f1_test': normalize(results_df['test_f1'], True),
        'score_cv_f1': normalize(results_df['cv_f1_mean'], True),
        'score_stability': normalize(results_df['cv_f1_std'], False),
        'score_overfit': normalize(results_df['overfit_gap'].abs(), False),
        'score_speed': normalize(results_df['train_cpu_time'], False),
    })
    
    weights = {'score_f1_test': 0.45, 'score_cv_f1': 0.40, 'score_stability': 0.15, 
               'score_overfit': 0.00, 'score_speed': 0.00}
    
    scores_df['total_score'] = sum(scores_df[col] * w for col, w in weights.items())
    scores_df = scores_df.sort_values('total_score', ascending=False)
    
    print("\nPondération: F1 (45%), CV F1 (40%), Stabilité (15%)")
    print(scores_df.to_string(index=False))
    
    winner = scores_df.iloc[0]['model_name']
    print(f"\n[WINNER] GAGNANT: {winner} (score: {scores_df.iloc[0]['total_score']:.3f})")
    
    output_dir = root_dir / 'results' / 'ml'
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / 'benchmark_results.csv', index=False)
    scores_df.to_csv(output_dir / 'benchmark_scores.csv', index=False)
    
    if HAS_PLOTTING:
        generate_plots(results_df, scores_df, output_dir)
    
    print(f"\n[SAVED] Resultats sauvegardes dans {output_dir}")
    
    return results_df, scores_df


def generate_plots(results_df, scores_df, output_dir):
    """Génère les graphiques."""
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results_df))
    width = 0.35
    
    ax.bar(x - width/2, results_df['test_f1'], width, label='F1 Test', color=colors[0])
    ax.bar(x + width/2, results_df['cv_f1_mean'], width, label='CV F1 Mean', color=colors[2])
    ax.errorbar(x + width/2, results_df['cv_f1_mean'], yerr=results_df['cv_f1_std'], 
                fmt='none', color='black', capsize=5)
    
    ax.set_xlabel('Modèle')
    ax.set_ylabel('F1 Score')
    ax.set_title('Comparaison F1 - ML Models')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['model_name'])
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(fig_dir / 'f1_comparison.png', dpi=150)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(scores_df['model_name'], scores_df['total_score'], color=colors)
    ax.set_xlabel('Score Total')
    ax.set_title('Score Global - ML Models')
    ax.set_xlim(0, 1)
    for bar, score in zip(bars, scores_df['total_score']):
        ax.text(score + 0.02, bar.get_y() + bar.get_height()/2, f'{score:.3f}', 
                va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'total_score.png', dpi=150)
    plt.close()
    
    print(f"[GRAPHS] Graphiques generes dans {fig_dir}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark ML')
    parser.add_argument('--data', type=str, default='data/ci_logs_dataset.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    data_path = root_dir / args.data
    if not data_path.exists():
        print(f"[ERROR] Dataset non trouve: {data_path}")
        return
    
    run_benchmark(data_path, seed=args.seed)


if __name__ == '__main__':
    main()
