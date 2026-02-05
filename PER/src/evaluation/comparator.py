"""
Model Comparator - Comparaison des modèles ML vs Transformers
=============================================================

Ce module fournit des outils pour comparer les performances
des différents modèles et générer des rapports comparatifs.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Comparateur de modèles pour le benchmark PER.
    
    Compare les modèles ML (Boosting) et Transformers selon:
    - Performance (accuracy, F1, etc.)
    - Temps d'entraînement et d'inférence
    - Utilisation mémoire
    - Scalabilité
    """
    
    def __init__(self):
        """Initialise le comparateur."""
        self.ml_results = {}
        self.transformer_results = {}
        self.comparison_results = {}
    
    def add_ml_result(
        self,
        model_name: str,
        metrics: Dict[str, float],
        training_stats: Optional[Dict] = None,
        dataset_size: str = 'small'
    ) -> None:
        """
        Ajoute les résultats d'un modèle ML.
        
        Args:
            model_name: Nom du modèle
            metrics: Métriques de classification
            training_stats: Statistiques d'entraînement
            dataset_size: 'small' ou 'large'
        """
        result = {
            'model_name': model_name,
            'model_type': 'ML',
            'metrics': metrics,
            'training_stats': training_stats or {},
            'dataset_size': dataset_size,
            'timestamp': datetime.now().isoformat(),
        }
        
        if model_name not in self.ml_results:
            self.ml_results[model_name] = []
        
        self.ml_results[model_name].append(result)
        logger.info(f"Résultat ajouté pour {model_name} (ML)")
    
    def add_transformer_result(
        self,
        model_name: str,
        metrics: Dict[str, float],
        training_stats: Optional[Dict] = None,
        dataset_size: str = 'large'
    ) -> None:
        """
        Ajoute les résultats d'un modèle Transformer.
        
        Args:
            model_name: Nom du modèle
            metrics: Métriques de classification
            training_stats: Statistiques d'entraînement
            dataset_size: 'small' ou 'large'
        """
        result = {
            'model_name': model_name,
            'model_type': 'Transformer',
            'metrics': metrics,
            'training_stats': training_stats or {},
            'dataset_size': dataset_size,
            'timestamp': datetime.now().isoformat(),
        }
        
        if model_name not in self.transformer_results:
            self.transformer_results[model_name] = []
        
        self.transformer_results[model_name].append(result)
        logger.info(f"Résultat ajouté pour {model_name} (Transformer)")
    
    def create_ml_comparison_table(self) -> pd.DataFrame:
        """
        Crée un tableau comparatif des modèles ML.
        
        Returns:
            DataFrame comparatif
        """
        rows = []
        
        for model_name, results in self.ml_results.items():
            if not results:
                continue
            
            # Prendre le dernier résultat ou faire la moyenne
            result = results[-1]
            
            row = {
                'Modèle': model_name,
                'Accuracy': result['metrics'].get('accuracy', 0),
                'Precision': result['metrics'].get('precision', 0),
                'Recall': result['metrics'].get('recall', 0),
                'F1-Score': result['metrics'].get('f1', 0),
                'Temps entraînement (s)': result['training_stats'].get('training_time_seconds', 0),
                'Temps inférence (ms/sample)': result['metrics'].get('inference_time_per_sample_ms', 0),
                'Mémoire (MB)': result['training_stats'].get('memory_usage_mb', 0),
                'Dataset': result.get('dataset_size', 'unknown'),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if not df.empty:
            # Trier par F1-Score décroissant
            df = df.sort_values('F1-Score', ascending=False)
        
        return df
    
    def create_transformer_comparison_table(self) -> pd.DataFrame:
        """
        Crée un tableau comparatif des modèles Transformers.
        
        Returns:
            DataFrame comparatif
        """
        rows = []
        
        for model_name, results in self.transformer_results.items():
            if not results:
                continue
            
            result = results[-1]
            
            row = {
                'Modèle': model_name,
                'Accuracy': result['metrics'].get('accuracy', 0),
                'Precision': result['metrics'].get('precision', 0),
                'Recall': result['metrics'].get('recall', 0),
                'F1-Score': result['metrics'].get('f1', 0),
                'Temps entraînement (s)': result['training_stats'].get('training_time_seconds', 0),
                'Temps inférence (ms/sample)': result['metrics'].get('inference_time_per_sample_ms', 0),
                'Mémoire (MB)': result['training_stats'].get('memory_usage_mb', 0),
                'Epochs': result['training_stats'].get('epochs', 0),
                'Dataset': result.get('dataset_size', 'unknown'),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if not df.empty:
            df = df.sort_values('F1-Score', ascending=False)
        
        return df
    
    def create_global_comparison_table(self) -> pd.DataFrame:
        """
        Crée un tableau comparatif global ML vs Transformers.
        
        Returns:
            DataFrame comparatif global
        """
        ml_df = self.create_ml_comparison_table()
        transformer_df = self.create_transformer_comparison_table()
        
        if not ml_df.empty:
            ml_df['Type'] = 'ML (Boosting)'
        if not transformer_df.empty:
            transformer_df['Type'] = 'Transformers'
        
        # Colonnes communes
        common_cols = ['Modèle', 'Type', 'Accuracy', 'Precision', 'Recall', 'F1-Score',
                      'Temps entraînement (s)', 'Temps inférence (ms/sample)', 'Mémoire (MB)']
        
        dfs = []
        if not ml_df.empty:
            dfs.append(ml_df[[c for c in common_cols if c in ml_df.columns]])
        if not transformer_df.empty:
            dfs.append(transformer_df[[c for c in common_cols if c in transformer_df.columns]])
        
        if dfs:
            global_df = pd.concat(dfs, ignore_index=True)
            global_df = global_df.sort_values('F1-Score', ascending=False)
            return global_df
        
        return pd.DataFrame()
    
    def analyze_ml_family(self) -> Dict[str, Any]:
        """
        Analyse intra-famille des modèles ML.
        
        Returns:
            Dictionnaire d'analyse
        """
        df = self.create_ml_comparison_table()
        
        if df.empty:
            return {'error': 'Pas de résultats ML disponibles'}
        
        analysis = {
            'best_model': df.iloc[0]['Modèle'],
            'best_f1': df.iloc[0]['F1-Score'],
            'worst_model': df.iloc[-1]['Modèle'],
            'worst_f1': df.iloc[-1]['F1-Score'],
            'average_metrics': {
                'accuracy': df['Accuracy'].mean(),
                'precision': df['Precision'].mean(),
                'recall': df['Recall'].mean(),
                'f1': df['F1-Score'].mean(),
            },
            'fastest_training': df.loc[df['Temps entraînement (s)'].idxmin(), 'Modèle'],
            'fastest_inference': df.loc[df['Temps inférence (ms/sample)'].idxmin(), 'Modèle'],
            'lowest_memory': df.loc[df['Mémoire (MB)'].idxmin(), 'Modèle'] if df['Mémoire (MB)'].sum() > 0 else 'N/A',
            'recommendations': [],
        }
        
        # Générer des recommandations
        if analysis['best_f1'] > 0.9:
            analysis['recommendations'].append("Excellentes performances ML, modèles adaptés au small dataset")
        elif analysis['best_f1'] > 0.8:
            analysis['recommendations'].append("Bonnes performances ML, considérer feature engineering")
        else:
            analysis['recommendations'].append("Performances ML modérées, vérifier la qualité des features")
        
        return analysis
    
    def analyze_transformer_family(self) -> Dict[str, Any]:
        """
        Analyse intra-famille des modèles Transformers.
        
        Returns:
            Dictionnaire d'analyse
        """
        df = self.create_transformer_comparison_table()
        
        if df.empty:
            return {'error': 'Pas de résultats Transformer disponibles'}
        
        analysis = {
            'best_model': df.iloc[0]['Modèle'],
            'best_f1': df.iloc[0]['F1-Score'],
            'worst_model': df.iloc[-1]['Modèle'],
            'worst_f1': df.iloc[-1]['F1-Score'],
            'average_metrics': {
                'accuracy': df['Accuracy'].mean(),
                'precision': df['Precision'].mean(),
                'recall': df['Recall'].mean(),
                'f1': df['F1-Score'].mean(),
            },
            'fastest_training': df.loc[df['Temps entraînement (s)'].idxmin(), 'Modèle'],
            'fastest_inference': df.loc[df['Temps inférence (ms/sample)'].idxmin(), 'Modèle'],
            'efficiency_winner': None,
            'recommendations': [],
        }
        
        # Trouver le meilleur compromis performance/temps
        df['efficiency_score'] = df['F1-Score'] / (df['Temps entraînement (s)'] + 1)
        analysis['efficiency_winner'] = df.loc[df['efficiency_score'].idxmax(), 'Modèle']
        
        # Recommandations
        if analysis['best_f1'] > 0.9:
            analysis['recommendations'].append("Excellentes performances Transformer sur large dataset")
        
        if 'DistilBERT' in df['Modèle'].values:
            distilbert_row = df[df['Modèle'] == 'DistilBERT'].iloc[0]
            bert_rows = df[df['Modèle'].str.contains('BERT', case=False) & ~df['Modèle'].str.contains('Distil')]
            if not bert_rows.empty:
                bert_row = bert_rows.iloc[0]
                if distilbert_row['F1-Score'] >= bert_row['F1-Score'] * 0.97:
                    analysis['recommendations'].append("DistilBERT offre un bon compromis performance/efficacité")
        
        return analysis
    
    def analyze_cross_family(self) -> Dict[str, Any]:
        """
        Analyse transversale ML vs Transformers.
        
        Returns:
            Dictionnaire d'analyse comparative
        """
        ml_df = self.create_ml_comparison_table()
        tf_df = self.create_transformer_comparison_table()
        
        if ml_df.empty and tf_df.empty:
            return {'error': 'Pas de résultats disponibles'}
        
        analysis = {
            'comparison_criteria': {},
            'recommendations': {},
            'summary': '',
        }
        
        # Performance
        if not ml_df.empty and not tf_df.empty:
            ml_best_f1 = ml_df['F1-Score'].max()
            tf_best_f1 = tf_df['F1-Score'].max()
            
            analysis['comparison_criteria']['performance'] = {
                'ml_best_f1': ml_best_f1,
                'transformer_best_f1': tf_best_f1,
                'winner': 'Transformers' if tf_best_f1 > ml_best_f1 else 'ML',
                'difference': abs(tf_best_f1 - ml_best_f1),
            }
            
            # Temps d'entraînement
            ml_avg_train = ml_df['Temps entraînement (s)'].mean()
            tf_avg_train = tf_df['Temps entraînement (s)'].mean()
            
            analysis['comparison_criteria']['training_time'] = {
                'ml_average': ml_avg_train,
                'transformer_average': tf_avg_train,
                'winner': 'ML' if ml_avg_train < tf_avg_train else 'Transformers',
                'ratio': tf_avg_train / max(ml_avg_train, 0.001),
            }
            
            # Temps d'inférence
            ml_avg_inf = ml_df['Temps inférence (ms/sample)'].mean()
            tf_avg_inf = tf_df['Temps inférence (ms/sample)'].mean()
            
            analysis['comparison_criteria']['inference_time'] = {
                'ml_average': ml_avg_inf,
                'transformer_average': tf_avg_inf,
                'winner': 'ML' if ml_avg_inf < tf_avg_inf else 'Transformers',
                'ratio': tf_avg_inf / max(ml_avg_inf, 0.001),
            }
        
        # Recommandations
        analysis['recommendations'] = {
            'small_dataset': "ML (Boosting) recommandé pour les petits datasets - entraînement rapide, bonnes performances",
            'large_dataset': "Transformers recommandés pour les grands datasets - meilleures capacités de généralisation",
            'production_constraints': "DistilBERT ou XGBoost selon les contraintes de latence",
            'hybrid_approach': "Possibilité d'approche hybride: ML pour le triage rapide, Transformers pour les cas complexes",
        }
        
        # Résumé
        analysis['summary'] = self._generate_summary(analysis)
        
        self.comparison_results = analysis
        return analysis
    
    def _generate_summary(self, analysis: Dict) -> str:
        """Génère un résumé textuel de l'analyse."""
        lines = [
            "=== RÉSUMÉ DE L'ANALYSE COMPARATIVE ===",
            "",
        ]
        
        if 'comparison_criteria' in analysis:
            criteria = analysis['comparison_criteria']
            
            if 'performance' in criteria:
                perf = criteria['performance']
                lines.append(f"PERFORMANCE:")
                lines.append(f"  - Meilleur F1 ML: {perf['ml_best_f1']:.4f}")
                lines.append(f"  - Meilleur F1 Transformer: {perf['transformer_best_f1']:.4f}")
                lines.append(f"  - Gagnant: {perf['winner']} (+{perf['difference']:.4f})")
                lines.append("")
            
            if 'training_time' in criteria:
                train = criteria['training_time']
                lines.append(f"TEMPS D'ENTRAÎNEMENT:")
                lines.append(f"  - ML moyen: {train['ml_average']:.2f}s")
                lines.append(f"  - Transformer moyen: {train['transformer_average']:.2f}s")
                lines.append(f"  - Ratio: Transformers {train['ratio']:.1f}x plus lent")
                lines.append("")
            
            if 'inference_time' in criteria:
                inf = criteria['inference_time']
                lines.append(f"TEMPS D'INFÉRENCE:")
                lines.append(f"  - ML moyen: {inf['ml_average']:.4f}ms/sample")
                lines.append(f"  - Transformer moyen: {inf['transformer_average']:.4f}ms/sample")
                lines.append("")
        
        lines.append("RECOMMANDATIONS:")
        for key, rec in analysis.get('recommendations', {}).items():
            lines.append(f"  - {key}: {rec}")
        
        return "\n".join(lines)
    
    def export_to_latex(self, output_dir: str) -> Dict[str, str]:
        """
        Exporte les tableaux en format LaTeX.
        
        Args:
            output_dir: Répertoire de sortie
            
        Returns:
            Dictionnaire des chemins de fichiers
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # Tableau ML
        ml_df = self.create_ml_comparison_table()
        if not ml_df.empty:
            latex_ml = ml_df.to_latex(index=False, float_format="%.4f")
            ml_path = output_dir / 'ml_comparison.tex'
            with open(ml_path, 'w', encoding='utf-8') as f:
                f.write(latex_ml)
            files['ml'] = str(ml_path)
        
        # Tableau Transformers
        tf_df = self.create_transformer_comparison_table()
        if not tf_df.empty:
            latex_tf = tf_df.to_latex(index=False, float_format="%.4f")
            tf_path = output_dir / 'transformer_comparison.tex'
            with open(tf_path, 'w', encoding='utf-8') as f:
                f.write(latex_tf)
            files['transformer'] = str(tf_path)
        
        # Tableau global
        global_df = self.create_global_comparison_table()
        if not global_df.empty:
            latex_global = global_df.to_latex(index=False, float_format="%.4f")
            global_path = output_dir / 'global_comparison.tex'
            with open(global_path, 'w', encoding='utf-8') as f:
                f.write(latex_global)
            files['global'] = str(global_path)
        
        logger.info(f"Tableaux LaTeX exportés dans {output_dir}")
        return files
    
    def export_to_csv(self, output_dir: str) -> Dict[str, str]:
        """
        Exporte les tableaux en format CSV.
        
        Args:
            output_dir: Répertoire de sortie
            
        Returns:
            Dictionnaire des chemins de fichiers
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # Tableau ML
        ml_df = self.create_ml_comparison_table()
        if not ml_df.empty:
            ml_path = output_dir / 'ml_comparison.csv'
            ml_df.to_csv(ml_path, index=False)
            files['ml'] = str(ml_path)
        
        # Tableau Transformers
        tf_df = self.create_transformer_comparison_table()
        if not tf_df.empty:
            tf_path = output_dir / 'transformer_comparison.csv'
            tf_df.to_csv(tf_path, index=False)
            files['transformer'] = str(tf_path)
        
        # Tableau global
        global_df = self.create_global_comparison_table()
        if not global_df.empty:
            global_path = output_dir / 'global_comparison.csv'
            global_df.to_csv(global_path, index=False)
            files['global'] = str(global_path)
        
        logger.info(f"Tableaux CSV exportés dans {output_dir}")
        return files
    
    def save_analysis(self, filepath: str) -> None:
        """
        Sauvegarde l'analyse complète en JSON.
        
        Args:
            filepath: Chemin du fichier
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'ml_results': self.ml_results,
            'transformer_results': self.transformer_results,
            'comparison_results': self.comparison_results,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Analyse sauvegardée: {filepath}")


def main():
    """Fonction de test du comparateur."""
    print("Test du ModelComparator")
    print("=" * 50)
    
    comparator = ModelComparator()
    
    # Ajouter des résultats ML simulés
    comparator.add_ml_result(
        model_name="XGBoost",
        metrics={'accuracy': 0.92, 'precision': 0.91, 'recall': 0.90, 'f1': 0.905, 'inference_time_per_sample_ms': 0.05},
        training_stats={'training_time_seconds': 5.2, 'memory_usage_mb': 120},
        dataset_size='small'
    )
    comparator.add_ml_result(
        model_name="AdaBoost",
        metrics={'accuracy': 0.88, 'precision': 0.87, 'recall': 0.86, 'f1': 0.865, 'inference_time_per_sample_ms': 0.08},
        training_stats={'training_time_seconds': 8.1, 'memory_usage_mb': 95},
        dataset_size='small'
    )
    comparator.add_ml_result(
        model_name="Gradient Boosting",
        metrics={'accuracy': 0.90, 'precision': 0.89, 'recall': 0.88, 'f1': 0.885, 'inference_time_per_sample_ms': 0.06},
        training_stats={'training_time_seconds': 12.5, 'memory_usage_mb': 110},
        dataset_size='small'
    )
    
    # Ajouter des résultats Transformer simulés
    comparator.add_transformer_result(
        model_name="BERT",
        metrics={'accuracy': 0.94, 'precision': 0.93, 'recall': 0.92, 'f1': 0.925, 'inference_time_per_sample_ms': 15.2},
        training_stats={'training_time_seconds': 180, 'memory_usage_mb': 2048, 'epochs': 3},
        dataset_size='large'
    )
    comparator.add_transformer_result(
        model_name="DistilBERT",
        metrics={'accuracy': 0.93, 'precision': 0.92, 'recall': 0.91, 'f1': 0.915, 'inference_time_per_sample_ms': 8.5},
        training_stats={'training_time_seconds': 95, 'memory_usage_mb': 1024, 'epochs': 3},
        dataset_size='large'
    )
    comparator.add_transformer_result(
        model_name="RoBERTa",
        metrics={'accuracy': 0.95, 'precision': 0.94, 'recall': 0.93, 'f1': 0.935, 'inference_time_per_sample_ms': 16.8},
        training_stats={'training_time_seconds': 200, 'memory_usage_mb': 2200, 'epochs': 3},
        dataset_size='large'
    )
    
    # Afficher les tableaux
    print("\n=== TABLEAU ML ===")
    print(comparator.create_ml_comparison_table().to_string())
    
    print("\n=== TABLEAU TRANSFORMERS ===")
    print(comparator.create_transformer_comparison_table().to_string())
    
    print("\n=== TABLEAU GLOBAL ===")
    print(comparator.create_global_comparison_table().to_string())
    
    # Analyses
    print("\n=== ANALYSE ML ===")
    ml_analysis = comparator.analyze_ml_family()
    print(f"Meilleur modèle: {ml_analysis['best_model']} (F1={ml_analysis['best_f1']:.4f})")
    
    print("\n=== ANALYSE TRANSFORMERS ===")
    tf_analysis = comparator.analyze_transformer_family()
    print(f"Meilleur modèle: {tf_analysis['best_model']} (F1={tf_analysis['best_f1']:.4f})")
    print(f"Plus efficace: {tf_analysis['efficiency_winner']}")
    
    print("\n=== ANALYSE TRANSVERSALE ===")
    cross_analysis = comparator.analyze_cross_family()
    print(cross_analysis['summary'])


if __name__ == "__main__":
    main()
