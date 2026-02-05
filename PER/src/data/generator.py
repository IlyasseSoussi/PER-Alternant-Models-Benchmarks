"""
Generator - Génération de données synthétiques de logs CI/CD
============================================================

Ce module génère des logs synthétiques réalistes pour le benchmark.
Les données sont utilisées comme corpus expérimental anonymisé.
"""

import random
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import yaml
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticLogGenerator:
    """
    Générateur de logs CI/CD synthétiques pour le benchmark.
    
    Génère deux types de logs :
    - Flaky / False Positive (classe 0)
    - Non Flaky / Possible Regression (classe 1)
    """
    
    # Templates de logs FLAKY (classe 0)
    FLAKY_TEMPLATES = [
        # Timeout et network
        "Connection to {service} timed out after {timeout}ms, retrying...",
        "Network error: connection refused to {host}:{port}, attempt {attempt}/3",
        "Timeout waiting for {resource}, test marked as flaky",
        "Intermittent failure in {test_name}: connection reset by peer",
        "Resource temporarily unavailable: {resource}",
        "Socket timeout during {operation}, will retry",
        
        # Test instabilité
        "FLAKY: Test {test_name} failed intermittently, known issue",
        "Test {test_name} is unstable, skipping for now",
        "Random failure detected in {test_class}.{test_method}",
        "Non-deterministic test result in {test_name}",
        "Test flapping: {test_name} passed after {retries} retries",
        "Spurious failure in parallel test execution: {test_name}",
        
        # Race conditions
        "Possible race condition in {component}, test unreliable",
        "Concurrent access issue detected, marking as flaky",
        "Thread timing issue in {test_name}, intermittent failure",
        "Deadlock detected but recovered in {component}",
        
        # Ressources
        "Temporary resource exhaustion in {pool}",
        "Service {service} temporarily unavailable",
        "Cache miss caused timeout, retrying operation",
        "Database connection pool exhausted, waiting for free connection",
        
        # Infra
        "Jenkins agent disconnected temporarily during {stage}",
        "Build node {node} experiencing high load, test delayed",
        "CI infrastructure hiccup, restarting job",
        "Container startup timeout, extending deadline",
    ]
    
    # Templates de logs REGRESSION (classe 1)
    REGRESSION_TEMPLATES = [
        # Null/Type errors
        "FATAL: NullPointerException at {class_name}.{method}:{line}",
        "TypeError: Cannot read property '{prop}' of undefined",
        "ValueError: Expected {expected_type} but got {actual_type}",
        "NullReferenceException in {namespace}.{class_name}",
        "CRITICAL: Unhandled null reference in {component}",
        
        # Assertions
        "AssertionError: Expected {expected} but got {actual}",
        "Assertion failed: {condition} at {file}:{line}",
        "Test assertion mismatch: expected={expected}, actual={actual}",
        "REGRESSION: Output differs from expected in {test_name}",
        "Verification failed: {assertion_message}",
        
        # Crashes
        "FATAL ERROR: Segmentation fault in {component}",
        "Process crashed with signal SIGSEGV",
        "Stack overflow in recursive call at {method}",
        "Heap corruption detected in {allocator}",
        "Out of memory error during {operation}",
        
        # Compilation/Build
        "Compilation error: undefined reference to '{symbol}'",
        "Build failed: syntax error in {file}:{line}",
        "Linker error: unresolved symbol '{symbol}'",
        "Missing dependency: {dependency} required by {module}",
        "Type mismatch in {file}: cannot convert {from_type} to {to_type}",
        
        # Logic errors
        "BUG: Incorrect calculation in {function}: expected {expected}, got {actual}",
        "Logic error: division by zero in {method}",
        "Array index out of bounds at {array}[{index}]",
        "REGRESSION: Feature {feature} broken after commit {commit}",
        "Critical bug in {component}: data corruption detected",
        
        # Security/Auth
        "CRITICAL: Authentication bypass detected in {endpoint}",
        "Permission denied: user {user} lacks access to {resource}",
        "Authorization failure for {operation} on {resource}",
        "Security violation: invalid token in {service}",
        
        # API/Integration
        "API contract violation: missing required field '{field}'",
        "Integration test failed: service {service} returned unexpected status {status}",
        "Schema validation error: {error_message}",
        "Response mismatch in {endpoint}: expected HTTP {expected}, got {actual}",
    ]
    
    # Variables de substitution
    SUBSTITUTIONS = {
        'service': ['UserService', 'PaymentGateway', 'AuthService', 'DatabaseService', 
                   'CacheService', 'MessageQueue', 'APIGateway', 'NotificationService'],
        'host': ['localhost', 'db-server', 'api-host', 'cache-node', 'worker-1'],
        'port': ['8080', '3306', '5432', '6379', '9200', '27017'],
        'timeout': ['1000', '5000', '10000', '30000'],
        'resource': ['database', 'cache', 'file lock', 'connection pool', 'memory'],
        'test_name': ['TestUserLogin', 'TestPaymentFlow', 'TestDataSync', 'TestAPIEndpoint',
                     'TestConcurrentAccess', 'TestCacheInvalidation', 'TestAuthFlow'],
        'test_class': ['UserServiceTest', 'PaymentTest', 'IntegrationTest', 'E2ETest'],
        'test_method': ['testCreate', 'testUpdate', 'testDelete', 'testQuery', 'testValidation'],
        'component': ['AuthModule', 'DataLayer', 'CacheManager', 'EventBus', 'Scheduler'],
        'class_name': ['UserController', 'PaymentService', 'DataRepository', 'CacheHandler'],
        'method': ['process', 'validate', 'execute', 'compute', 'transform', 'handle'],
        'namespace': ['com.app.services', 'org.project.core', 'app.modules'],
        'line': ['42', '128', '256', '512', '1024'],
        'file': ['Main.java', 'Service.py', 'Handler.ts', 'Controller.cs', 'Module.go'],
        'expected': ['true', '200', '42', '"success"', '[1,2,3]'],
        'actual': ['false', '500', '0', '"error"', 'null', '[]'],
        'expected_type': ['string', 'number', 'object', 'array'],
        'actual_type': ['null', 'undefined', 'NaN', 'object'],
        'prop': ['id', 'name', 'value', 'data', 'result', 'items'],
        'condition': ['x > 0', 'result != null', 'count == expected'],
        'assertion_message': ['values should match', 'result should not be null', 'list should not be empty'],
        'symbol': ['main', 'init', 'process_data', 'cleanup'],
        'dependency': ['libssl', 'libpq', 'node_modules', 'vendor'],
        'module': ['core', 'utils', 'services', 'handlers'],
        'from_type': ['int', 'string', 'float'],
        'to_type': ['string', 'int', 'boolean'],
        'function': ['calculateTotal', 'processPayment', 'validateInput'],
        'array': ['items', 'results', 'data'],
        'index': ['10', '99', '-1', '1000'],
        'feature': ['user-login', 'payment-flow', 'data-export', 'report-generation'],
        'commit': ['abc1234', 'def5678', 'ghi9012'],
        'endpoint': ['/api/users', '/api/payments', '/api/auth', '/api/data'],
        'user': ['admin', 'user123', 'test_user'],
        'operation': ['read', 'write', 'delete', 'execute'],
        'field': ['user_id', 'amount', 'timestamp', 'status'],
        'status': ['500', '404', '401', '503'],
        'error_message': ['invalid format', 'missing required field', 'constraint violation'],
        'pool': ['connection_pool', 'thread_pool', 'memory_pool'],
        'node': ['node-1', 'node-2', 'worker-3'],
        'stage': ['build', 'test', 'deploy', 'validate'],
        'attempt': ['1', '2', '3'],
        'retries': ['2', '3', '5'],
        'allocator': ['default_allocator', 'custom_allocator'],
    }
    
    def __init__(self, config_path: Optional[str] = None, seed: int = 42):
        """
        Initialise le générateur.
        
        Args:
            config_path: Chemin vers le fichier de configuration
            seed: Graine pour la reproductibilité
        """
        random.seed(seed)
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Charge la configuration."""
        default_config = {
            'small_dataset_size': 1000,
            'large_dataset_size': 10000,
            'class_balance': 0.5,
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                if 'data' in yaml_config:
                    default_config.update(yaml_config['data'])
        
        return default_config
    
    def _substitute_template(self, template: str) -> str:
        """Remplace les variables dans un template."""
        result = template
        for var, options in self.SUBSTITUTIONS.items():
            placeholder = '{' + var + '}'
            while placeholder in result:
                result = result.replace(placeholder, random.choice(options), 1)
        return result
    
    def generate_flaky_log(self) -> str:
        """Génère un log flaky (classe 0)."""
        template = random.choice(self.FLAKY_TEMPLATES)
        return self._substitute_template(template)
    
    def generate_regression_log(self) -> str:
        """Génère un log de regression (classe 1)."""
        template = random.choice(self.REGRESSION_TEMPLATES)
        return self._substitute_template(template)
    
    def generate_log(self, label: int) -> str:
        """
        Génère un log pour une classe donnée.
        
        Args:
            label: 0 pour flaky, 1 pour regression
            
        Returns:
            Texte du log
        """
        if label == 0:
            return self.generate_flaky_log()
        else:
            return self.generate_regression_log()
    
    def generate_dataset(
        self,
        size: int,
        class_balance: float = 0.5,
        add_noise: bool = True
    ) -> pd.DataFrame:
        """
        Génère un dataset complet.
        
        Args:
            size: Nombre de logs à générer
            class_balance: Proportion de la classe 1 (regression)
            add_noise: Ajouter des variations dans le texte
            
        Returns:
            DataFrame avec colonnes ['log_text', 'label']
        """
        logger.info(f"Génération de {size} logs synthétiques...")
        
        logs = []
        labels = []
        
        # Calculer le nombre de chaque classe
        n_regression = int(size * class_balance)
        n_flaky = size - n_regression
        
        # Générer les logs flaky
        for _ in range(n_flaky):
            log = self.generate_flaky_log()
            if add_noise:
                log = self._add_noise(log)
            logs.append(log)
            labels.append(0)
        
        # Générer les logs regression
        for _ in range(n_regression):
            log = self.generate_regression_log()
            if add_noise:
                log = self._add_noise(log)
            logs.append(log)
            labels.append(1)
        
        # Créer le DataFrame
        df = pd.DataFrame({
            'log_text': logs,
            'label': labels
        })
        
        # Mélanger
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Dataset généré: {n_flaky} flaky, {n_regression} regression")
        
        return df
    
    def _add_noise(self, text: str) -> str:
        """Ajoute du bruit réaliste au texte."""
        modifications = []
        
        # Parfois ajouter un préfixe de niveau de log
        if random.random() < 0.3:
            prefix = random.choice(['[ERROR] ', '[WARN] ', '[INFO] ', '[DEBUG] ', 'ERROR: ', 'WARN: '])
            modifications.append(('prefix', prefix))
        
        # Parfois ajouter un suffixe
        if random.random() < 0.2:
            suffix = random.choice([' (see logs for details)', ' [retryable]', ' !', '...'])
            modifications.append(('suffix', suffix))
        
        # Parfois modifier la casse
        if random.random() < 0.1:
            modifications.append(('upper', None))
        
        # Appliquer les modifications
        result = text
        for mod_type, value in modifications:
            if mod_type == 'prefix':
                result = value + result
            elif mod_type == 'suffix':
                result = result + value
            elif mod_type == 'upper':
                result = result.upper()
        
        return result
    
    def generate_small_dataset(self) -> pd.DataFrame:
        """Génère le small dataset (simulation QA Rates)."""
        size = self.config.get('small_dataset_size', 1000)
        return self.generate_dataset(size, class_balance=0.5)
    
    def generate_large_dataset(self) -> pd.DataFrame:
        """Génère le large dataset (simulation QA Shopping)."""
        size = self.config.get('large_dataset_size', 10000)
        return self.generate_dataset(size, class_balance=0.5)
    
    def save_dataset(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Sauvegarde un dataset.
        
        Args:
            df: DataFrame à sauvegarder
            filepath: Chemin de destination
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Dataset sauvegardé: {filepath} ({len(df)} lignes)")
    
    def generate_and_save_all(self, base_path: str = 'data') -> Dict[str, str]:
        """
        Génère et sauvegarde tous les datasets.
        
        Args:
            base_path: Chemin de base pour la sauvegarde
            
        Returns:
            Dictionnaire des chemins des fichiers créés
        """
        base_path = Path(base_path)
        
        # Small dataset
        small_df = self.generate_small_dataset()
        small_path = base_path / 'small_dataset' / 'logs.csv'
        self.save_dataset(small_df, str(small_path))
        
        # Large dataset
        large_df = self.generate_large_dataset()
        large_path = base_path / 'large_dataset' / 'logs.csv'
        self.save_dataset(large_df, str(large_path))
        
        # Splits pour chaque dataset
        for name, df, path in [('small', small_df, base_path / 'small_dataset'),
                               ('large', large_df, base_path / 'large_dataset')]:
            train, val, test = self._split_dataset(df)
            self.save_dataset(train, str(path / 'train.csv'))
            self.save_dataset(val, str(path / 'val.csv'))
            self.save_dataset(test, str(path / 'test.csv'))
        
        return {
            'small_dataset': str(small_path),
            'large_dataset': str(large_path),
        }
    
    def _split_dataset(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Divise un dataset en train/val/test."""
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train = df[:train_end]
        val = df[train_end:val_end]
        test = df[val_end:]
        
        return train, val, test


def main():
    """Fonction principale pour générer les datasets."""
    import sys
    
    # Chercher le fichier de config
    config_path = None
    for path in ['config/config.yaml', '../config/config.yaml', '../../config/config.yaml']:
        if Path(path).exists():
            config_path = path
            break
    
    generator = SyntheticLogGenerator(config_path=config_path)
    
    print("=" * 60)
    print("Génération des datasets synthétiques")
    print("=" * 60)
    
    # Exemples de logs
    print("\nExemples de logs FLAKY (classe 0):")
    for _ in range(3):
        print(f"  - {generator.generate_flaky_log()}")
    
    print("\nExemples de logs REGRESSION (classe 1):")
    for _ in range(3):
        print(f"  - {generator.generate_regression_log()}")
    
    # Générer et sauvegarder
    print("\n" + "=" * 60)
    print("Génération et sauvegarde des datasets...")
    print("=" * 60)
    
    paths = generator.generate_and_save_all('data')
    
    print("\nDatasets créés:")
    for name, path in paths.items():
        print(f"  - {name}: {path}")
    
    print("\nGénération terminée!")


if __name__ == "__main__":
    main()
