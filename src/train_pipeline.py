"""
Student Success Prediction - Main Training Pipeline
===================================================

This module provides the complete ML pipeline from raw data to 
trained ensemble model. It orchestrates all components:
1. Data validation
2. Feature engineering
3. Train/test split
4. Model training
5. Evaluation
6. Model selection

Enhanced with:
- Feature engineering integration (Phase 1)
- SMOTE oversampling for class imbalance (Phase 2)
- Hyperparameter tuning with RandomizedSearchCV (Phase 3)
- Ensemble optimization (Phase 4)
- Per-class threshold optimization (Phase 5)
- Results tracking and reporting (Phase 6)

Author: ML Engineering Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, 
    confusion_matrix, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import uniform, randint
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
import warnings
import os

# Suppress all warnings globally
warnings.filterwarnings('ignore')

# Set environment variable to suppress warnings in child processes (for n_jobs=-1)
os.environ['PYTHONWARNINGS'] = 'ignore'

# Try to import imbalanced-learn for SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️ imbalanced-learn not installed. SMOTE will be disabled. Install with: pip install imbalanced-learn")


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

class PipelineConfig:
    """
    Configuration for the ML pipeline.
    
    WHY CONFIGURATION MATTERS:
    -------------------------
    Centralizing configuration makes experiments reproducible and 
    prevents "magic numbers" scattered throughout code.
    
    All important decisions are documented here.
    """
    
    # Data split ratios
    TEST_SIZE = 0.15          # 15% for final test
    VAL_SIZE = 0.15           # 15% for validation (from remaining 85%)
    
    # Cross-validation
    CV_FOLDS = 5
    
    # Random seed for reproducibility
    RANDOM_STATE = 42
    
    # Target column
    TARGET_COL = 'Target'
    
    # Class mapping
    CLASS_NAMES = ['Dropout', 'Enrolled', 'Graduate']
    
    # Minimum acceptable performance
    MIN_MACRO_F1 = 0.70
    TARGET_MACRO_F1 = 0.80
    
    # Paths
    MODEL_SAVE_PATH = Path('models/saved_models')
    RESULTS_PATH = Path('reports/results')
    
    # Feature Engineering
    USE_FEATURE_ENGINEERING = True
    
    # Class Imbalance Handling
    USE_SMOTE = True
    SMOTE_SAMPLING_STRATEGY = 'auto'  # Or dict like {1: 1000} for specific counts
    
    # Hyperparameter Tuning
    TUNE_HYPERPARAMETERS = True
    TUNING_N_ITER = 50  # Number of random search iterations
    TUNING_CV_FOLDS = 3  # CV folds for tuning (less than main CV for speed)
    
    # Threshold Optimization
    OPTIMIZE_THRESHOLDS = True


# =============================================================================
# RESULTS TRACKER (Phase 6)
# =============================================================================

class ResultsTracker:
    """
    Track and log results from each phase of the pipeline.
    
    This enables:
    - Comparison of results before/after each phase
    - Ablation studies showing contribution of each improvement
    - Automatic REPORT.md generation
    """
    
    def __init__(self):
        self.results = {
            'baseline': None,
            'phases': [],
            'best_params': {},
            'feature_importance': None,
            'confusion_matrices': {},
            'timestamps': {}
        }
        self.start_time = datetime.now()
        
    def log_baseline(self, metrics: Dict[str, Any]):
        """Log baseline metrics before any improvements."""
        self.results['baseline'] = {
            **metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.results['timestamps']['baseline'] = datetime.now().isoformat()
        print(f"📝 Logged baseline: Macro F1 = {metrics.get('macro_f1', 'N/A'):.4f}")
        
    def log_phase(self, phase_name: str, metrics: Dict[str, Any], 
                  description: str = "", params: Dict = None):
        """Log results after a phase."""
        phase_result = {
            'phase_name': phase_name,
            'description': description,
            'metrics': metrics,
            'params': params or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate delta from baseline if available
        if self.results['baseline']:
            baseline_f1 = self.results['baseline'].get('macro_f1', 0)
            current_f1 = metrics.get('macro_f1', 0)
            phase_result['delta_from_baseline'] = current_f1 - baseline_f1
            phase_result['delta_pct'] = ((current_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
            
        self.results['phases'].append(phase_result)
        print(f"📝 Logged {phase_name}: Macro F1 = {metrics.get('macro_f1', 'N/A'):.4f}")
        
    def log_best_params(self, model_name: str, params: Dict):
        """Log best hyperparameters found."""
        self.results['best_params'][model_name] = params
        
    def log_feature_importance(self, importance_df: pd.DataFrame):
        """Log feature importance rankings."""
        self.results['feature_importance'] = importance_df.to_dict()
        
    def log_confusion_matrix(self, phase_name: str, cm: np.ndarray, class_names: List[str]):
        """Log confusion matrix for a phase."""
        self.results['confusion_matrices'][phase_name] = {
            'matrix': cm.tolist(),
            'class_names': class_names
        }
        
    def generate_report(self, output_path: str = "REPORT.md") -> str:
        """Generate REPORT.md with all results."""
        report_lines = []
        
        # Header
        report_lines.append("# Student Success Prediction - Experiment Report")
        report_lines.append("")
        report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Duration**: {(datetime.now() - self.start_time).total_seconds():.1f} seconds")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("## Executive Summary")
        report_lines.append("")
        
        if self.results['baseline'] and self.results['phases']:
            baseline_f1 = self.results['baseline'].get('macro_f1', 0)
            final_phase = self.results['phases'][-1]
            final_f1 = final_phase['metrics'].get('macro_f1', 0)
            improvement = final_f1 - baseline_f1
            
            report_lines.append(f"| Metric | Baseline | Final | Improvement |")
            report_lines.append(f"|--------|----------|-------|-------------|")
            report_lines.append(f"| Macro F1 | {baseline_f1:.4f} | {final_f1:.4f} | **+{improvement:.4f}** ({improvement/baseline_f1*100:.1f}%) |")
            
            # Per-class comparison if available
            if 'per_class_f1' in self.results['baseline'] and 'per_class_f1' in final_phase['metrics']:
                baseline_per_class = self.results['baseline']['per_class_f1']
                final_per_class = final_phase['metrics']['per_class_f1']
                class_names = ['Dropout', 'Enrolled', 'Graduate']
                
                for i, name in enumerate(class_names):
                    if i < len(baseline_per_class) and i < len(final_per_class):
                        delta = final_per_class[i] - baseline_per_class[i]
                        report_lines.append(f"| {name} F1 | {baseline_per_class[i]:.4f} | {final_per_class[i]:.4f} | +{delta:.4f} |")
        
        report_lines.append("")
        
        # Baseline Results
        report_lines.append("## Baseline Results")
        report_lines.append("")
        if self.results['baseline']:
            report_lines.append(f"- **Macro F1**: {self.results['baseline'].get('macro_f1', 'N/A'):.4f}")
            report_lines.append(f"- **Accuracy**: {self.results['baseline'].get('accuracy', 'N/A'):.4f}")
            if 'per_class_f1' in self.results['baseline']:
                report_lines.append(f"- **Per-class F1**: Dropout={self.results['baseline']['per_class_f1'][0]:.4f}, " +
                                   f"Enrolled={self.results['baseline']['per_class_f1'][1]:.4f}, " +
                                   f"Graduate={self.results['baseline']['per_class_f1'][2]:.4f}")
        report_lines.append("")
        
        # Phase Results
        report_lines.append("## Phase Results")
        report_lines.append("")
        
        for phase in self.results['phases']:
            report_lines.append(f"### {phase['phase_name']}")
            report_lines.append("")
            if phase.get('description'):
                report_lines.append(f"*{phase['description']}*")
                report_lines.append("")
            
            metrics = phase['metrics']
            report_lines.append(f"- **Macro F1**: {metrics.get('macro_f1', 'N/A'):.4f}")
            report_lines.append(f"- **Accuracy**: {metrics.get('accuracy', 'N/A'):.4f}")
            
            if 'delta_from_baseline' in phase:
                delta = phase['delta_from_baseline']
                delta_pct = phase['delta_pct']
                sign = '+' if delta >= 0 else ''
                report_lines.append(f"- **Delta from baseline**: {sign}{delta:.4f} ({sign}{delta_pct:.1f}%)")
            
            if 'per_class_f1' in metrics:
                report_lines.append(f"- **Per-class F1**: Dropout={metrics['per_class_f1'][0]:.4f}, " +
                                   f"Enrolled={metrics['per_class_f1'][1]:.4f}, " +
                                   f"Graduate={metrics['per_class_f1'][2]:.4f}")
            report_lines.append("")
        
        # Best Hyperparameters
        if self.results['best_params']:
            report_lines.append("## Best Hyperparameters")
            report_lines.append("")
            for model_name, params in self.results['best_params'].items():
                report_lines.append(f"### {model_name}")
                report_lines.append("```")
                for k, v in params.items():
                    report_lines.append(f"{k}: {v}")
                report_lines.append("```")
                report_lines.append("")
        
        # Feature Importance
        if self.results['feature_importance']:
            report_lines.append("## Top Features")
            report_lines.append("")
            report_lines.append("| Rank | Feature | Importance |")
            report_lines.append("|------|---------|------------|")
            
            importance_data = self.results['feature_importance']
            if 'Feature' in importance_data and 'Importance_Pct' in importance_data:
                features = list(importance_data['Feature'].values())
                importances = list(importance_data['Importance_Pct'].values())
                for i, (feat, imp) in enumerate(zip(features[:15], importances[:15]), 1):
                    report_lines.append(f"| {i} | {feat} | {imp:.2f}% |")
            report_lines.append("")
        
        # Confusion Matrices
        if self.results['confusion_matrices']:
            report_lines.append("## Confusion Matrices")
            report_lines.append("")
            for phase_name, cm_data in self.results['confusion_matrices'].items():
                report_lines.append(f"### {phase_name}")
                report_lines.append("```")
                cm = np.array(cm_data['matrix'])
                class_names = cm_data['class_names']
                header = "           " + "  ".join(f"{name:>10}" for name in class_names)
                report_lines.append(header)
                for i, name in enumerate(class_names):
                    row = f"{name:>10} " + "  ".join(f"{cm[i,j]:>10}" for j in range(len(class_names)))
                    report_lines.append(row)
                report_lines.append("```")
                report_lines.append("")
        
        # Ablation Study
        if len(self.results['phases']) > 1:
            report_lines.append("## Ablation Study")
            report_lines.append("")
            report_lines.append("Contribution of each phase to overall improvement:")
            report_lines.append("")
            report_lines.append("| Phase | Delta F1 | Cumulative F1 |")
            report_lines.append("|-------|----------|---------------|")
            
            prev_f1 = self.results['baseline'].get('macro_f1', 0) if self.results['baseline'] else 0
            for phase in self.results['phases']:
                current_f1 = phase['metrics'].get('macro_f1', 0)
                phase_delta = current_f1 - prev_f1
                sign = '+' if phase_delta >= 0 else ''
                report_lines.append(f"| {phase['phase_name']} | {sign}{phase_delta:.4f} | {current_f1:.4f} |")
                prev_f1 = current_f1
            report_lines.append("")
        
        # Write report
        report_content = "\n".join(report_lines)
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        print(f"📄 Report saved to: {output_path}")
        return report_content


# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================

class DataLoader:
    """
    Load and validate the dataset.
    
    VALIDATION PHILOSOPHY:
    ---------------------
    "Garbage in, garbage out" - validating data early prevents 
    mysterious model failures later.
    
    We check for:
    - Missing values
    - Unexpected values in categorical columns
    - Class distribution
    - Data types
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.label_encoder = LabelEncoder()
        self.feature_engineer = None
        
    def load_data(self, filepath: str, apply_feature_engineering: bool = True) -> pd.DataFrame:
        """
        Load dataset from file.
        
        Supports: CSV, Excel, Parquet
        
        Phase 1: Now integrates StudentFeatureEngineer for engineered features.
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath)
        elif filepath.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        elif filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Apply cleaning pipeline
        from src.data.cleaning import clean_data
        df = clean_data(df)
        
        print(f"📂 Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Phase 1: Apply feature engineering
        if apply_feature_engineering and self.config.USE_FEATURE_ENGINEERING:
            df = self._apply_feature_engineering(df)
        
        return df
    
    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply StudentFeatureEngineer to create new features.
        
        This integrates the feature engineering module that was previously unused.
        """
        from src.data.feature_engineering import StudentFeatureEngineer
        
        original_cols = df.shape[1]
        
        # Initialize and apply feature engineering
        self.feature_engineer = StudentFeatureEngineer(
            apply_academic=True,
            apply_engagement=True,
            apply_financial=True,
            apply_demographic=True,
            apply_interactions=True
        )
        
        # Separate target if present
        target_col = self.config.TARGET_COL
        if target_col in df.columns:
            target = df[target_col]
            features_df = df.drop(columns=[target_col])
            features_df = self.feature_engineer.fit_transform(features_df)
            features_df[target_col] = target
            df = features_df
        else:
            df = self.feature_engineer.fit_transform(df)
        
        new_cols = df.shape[1]
        print(f"🔧 Feature engineering: {original_cols} → {new_cols} columns (+{new_cols - original_cols} engineered features)")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run validation checks on the data.
        
        Returns a report of any issues found.
        """
        report = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check for target column
        if self.config.TARGET_COL not in df.columns:
            report['is_valid'] = False
            report['issues'].append(f"Missing target column: {self.config.TARGET_COL}")
            return report
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            report['warnings'].append(f"Missing values found: {missing[missing > 0].to_dict()}")
        
        # Class distribution
        class_dist = df[self.config.TARGET_COL].value_counts(normalize=True)
        report['stats']['class_distribution'] = class_dist.to_dict()
        
        # Check for severe imbalance
        if class_dist.min() < 0.05:
            report['warnings'].append(
                f"Severe class imbalance detected: minority class is {class_dist.min():.1%}"
            )
        
        # Data types
        report['stats']['dtypes'] = df.dtypes.value_counts().to_dict()
        
        # Basic stats
        report['stats']['n_samples'] = len(df)
        report['stats']['n_features'] = len(df.columns) - 1
        
        return report
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Separate features and target, encode target.
        
        Returns:
        --------
        X : feature matrix
        y : encoded target vector
        feature_names : list of feature column names
        """
        # Separate features and target
        feature_cols = [col for col in df.columns if col != self.config.TARGET_COL]
        X = df[feature_cols].copy()
        y = df[self.config.TARGET_COL].copy()
        
        # Handle string target (if needed)
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        else:
            y = y.values
        
        # Convert features to numeric where possible
        for col in X.columns:
            if X[col].dtype == 'object':
                # Try to convert, create dummies for categorical
                try:
                    X[col] = pd.to_numeric(X[col])
                except ValueError:
                    # One-hot encode categorical columns
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
        
        # Fill any remaining NaN with 0 (from engineered features with division by zero)
        X = X.fillna(0)
        
        # Replace inf values
        X = X.replace([np.inf, -np.inf], 0)
        
        return X.values, y, list(X.columns)


# =============================================================================
# THRESHOLD OPTIMIZER (Phase 5)
# =============================================================================

class ThresholdOptimizer:
    """
    Optimize classification thresholds per class to maximize Macro F1.
    
    Instead of using argmax(probabilities), find optimal thresholds
    that maximize the target metric (Macro F1).
    """
    
    def __init__(self, n_classes: int = 3, n_thresholds: int = 20):
        self.n_classes = n_classes
        self.n_thresholds = n_thresholds
        self.optimal_thresholds = None
        
    def fit(self, y_proba: np.ndarray, y_true: np.ndarray) -> 'ThresholdOptimizer':
        """
        Find optimal thresholds using grid search on validation set.
        
        Parameters:
        -----------
        y_proba : np.ndarray of shape (n_samples, n_classes)
            Predicted probabilities
        y_true : np.ndarray of shape (n_samples,)
            True labels
        """
        # Generate threshold candidates
        thresholds = np.linspace(0.1, 0.9, self.n_thresholds)
        
        best_f1 = 0
        best_thresholds = np.ones(self.n_classes) / self.n_classes
        
        # For 3 classes, we can do a more targeted search
        # We focus on adjusting the threshold for the minority class (Enrolled = class 1)
        print("🎯 Optimizing classification thresholds...")
        
        for t1 in thresholds:  # Threshold adjustment for class 0 (Dropout)
            for t2 in thresholds:  # Threshold adjustment for class 1 (Enrolled - minority)
                for t3 in thresholds:  # Threshold adjustment for class 2 (Graduate)
                    adjusted_thresholds = np.array([t1, t2, t3])
                    
                    # Apply thresholds - multiply probabilities by threshold factors
                    adjusted_proba = y_proba * adjusted_thresholds
                    y_pred = adjusted_proba.argmax(axis=1)
                    
                    f1 = f1_score(y_true, y_pred, average='macro')
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresholds = adjusted_thresholds.copy()
        
        self.optimal_thresholds = best_thresholds
        print(f"   Optimal thresholds: {best_thresholds}")
        print(f"   Best Macro F1 with thresholds: {best_f1:.4f}")
        
        return self
    
    def predict(self, y_proba: np.ndarray) -> np.ndarray:
        """Apply optimal thresholds to get predictions."""
        if self.optimal_thresholds is None:
            raise ValueError("ThresholdOptimizer not fitted. Call fit() first.")
        
        adjusted_proba = y_proba * self.optimal_thresholds
        return adjusted_proba.argmax(axis=1)


# =============================================================================
# HYPERPARAMETER TUNER (Phase 3)
# =============================================================================

class HyperparameterTuner:
    """
    Tune hyperparameters using RandomizedSearchCV.
    
    Provides search spaces for XGBoost, LightGBM, and Random Forest.
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.best_params = {}
        
    def get_xgboost_search_space(self) -> Dict:
        """Search space for XGBoost."""
        return {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 12),
            'learning_rate': uniform(0.01, 0.29),  # 0.01 to 0.30
            'subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
            'colsample_bytree': uniform(0.6, 0.4),  # 0.6 to 1.0
            'min_child_weight': randint(1, 10),
            'gamma': uniform(0, 0.5),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 2)
        }
    
    def get_lightgbm_search_space(self) -> Dict:
        """Search space for LightGBM."""
        return {
            'n_estimators': randint(100, 500),
            'num_leaves': randint(20, 100),
            'max_depth': randint(3, 15),
            'learning_rate': uniform(0.01, 0.29),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 2),
            'min_child_samples': randint(5, 50)
        }
    
    def get_rf_search_space(self) -> Dict:
        """Search space for Random Forest."""
        return {
            'n_estimators': randint(100, 500),
            'max_depth': randint(5, 30),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None]
        }
    
    def tune_model(self, model, param_space: Dict, X: np.ndarray, y: np.ndarray,
                   model_name: str, n_iter: int = None, n_jobs: int = 2) -> Tuple[Any, Dict]:
        """
        Tune a model using RandomizedSearchCV.
        
        Returns:
        --------
        best_model : fitted model with best parameters
        best_params : dictionary of best parameters
        
        Note: n_jobs defaults to 2 to prevent memory issues on most hardware.
        """
        n_iter = n_iter or self.config.TUNING_N_ITER
        
        print(f"\n🔍 Tuning {model_name} ({n_iter} iterations, {n_jobs} parallel jobs)...")
        
        cv = StratifiedKFold(n_splits=self.config.TUNING_CV_FOLDS, shuffle=True, 
                             random_state=self.config.RANDOM_STATE)
        
        search = RandomizedSearchCV(
            model,
            param_distributions=param_space,
            n_iter=n_iter,
            cv=cv,
            scoring='f1_macro',
            n_jobs=n_jobs,  # Reduced parallelism for memory safety
            random_state=self.config.RANDOM_STATE,
            verbose=0
        )
        
        search.fit(X, y)
        
        print(f"   Best CV Macro F1: {search.best_score_:.4f}")
        print(f"   Best params: {search.best_params_}")
        
        self.best_params[model_name] = search.best_params_
        
        return search.best_estimator_, search.best_params_


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

class TrainingPipeline:
    """
    Complete training pipeline orchestrator.
    
    PIPELINE STAGES:
    ---------------
    1. Data Loading & Validation
    2. Feature Engineering
    3. Data Splitting (Train/Val/Test)
    4. SMOTE Oversampling (Phase 2)
    5. Baseline Evaluation
    6. Model Training with Cross-Validation
    7. Hyperparameter Tuning (Phase 3)
    8. Model Selection
    9. Threshold Optimization (Phase 5)
    10. Final Evaluation on Test Set
    11. Model Saving
    12. Report Generation (Phase 6)
    
    This class ties everything together into a reproducible workflow.
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.data_loader = DataLoader(self.config)
        self.results_tracker = ResultsTracker()
        self.threshold_optimizer = None
        self.tuner = HyperparameterTuner(self.config)
        
        # Storage for results
        self.X_train = None
        self.X_train_resampled = None  # After SMOTE
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_train_resampled = None  # After SMOTE
        self.y_val = None
        self.y_test = None
        self.feature_names = None
        
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_and_prepare(self, filepath: str) -> 'TrainingPipeline':
        """Stage 1: Load and prepare data with feature engineering."""
        print("\n" + "="*60)
        print("📂 STAGE 1: DATA LOADING AND VALIDATION")
        print("="*60)
        
        df = self.data_loader.load_data(filepath, apply_feature_engineering=self.config.USE_FEATURE_ENGINEERING)
        
        # Validate
        report = self.data_loader.validate_data(df)
        
        if not report['is_valid']:
            raise ValueError(f"Data validation failed: {report['issues']}")
        
        for warning in report['warnings']:
            print(f"⚠️ Warning: {warning}")
        
        print(f"\n📊 Dataset stats:")
        print(f"   Samples: {report['stats']['n_samples']}")
        print(f"   Features: {report['stats']['n_features']}")
        print(f"   Class distribution: {report['stats']['class_distribution']}")
        
        # Prepare
        X, y, feature_names = self.data_loader.prepare_data(df)
        self.feature_names = feature_names
        
        # First split: separate test set (final evaluation)
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, 
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
        
        # Second split: separate validation set (hyperparameter tuning)
        val_size_adjusted = self.config.VAL_SIZE / (1 - self.config.TEST_SIZE)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.RANDOM_STATE,
            stratify=y_temp
        )
        
        print(f"\n📊 Data splits:")
        print(f"   Training: {len(self.y_train)} samples")
        print(f"   Validation: {len(self.y_val)} samples")
        print(f"   Test: {len(self.y_test)} samples")
        
        return self
    
    def apply_smote(self) -> 'TrainingPipeline':
        """Stage 2: Apply SMOTE oversampling for class imbalance."""
        print("\n" + "="*60)
        print("⚖️ STAGE 2: CLASS IMBALANCE HANDLING (SMOTE)")
        print("="*60)
        
        if not SMOTE_AVAILABLE:
            print("⚠️ SMOTE not available. Skipping resampling.")
            self.X_train_resampled = self.X_train
            self.y_train_resampled = self.y_train
            return self
        
        if not self.config.USE_SMOTE:
            print("⚠️ SMOTE disabled in config. Skipping resampling.")
            self.X_train_resampled = self.X_train
            self.y_train_resampled = self.y_train
            return self
        
        # Original class distribution
        unique, counts = np.unique(self.y_train, return_counts=True)
        print("\n📊 Original class distribution:")
        for cls, count in zip(unique, counts):
            print(f"   Class {cls}: {count} ({count/len(self.y_train)*100:.1f}%)")
        
        # Apply SMOTE
        smote = SMOTE(
            sampling_strategy=self.config.SMOTE_SAMPLING_STRATEGY,
            random_state=self.config.RANDOM_STATE,
            k_neighbors=5
        )
        
        self.X_train_resampled, self.y_train_resampled = smote.fit_resample(
            self.X_train, self.y_train
        )
        
        # New class distribution
        unique, counts = np.unique(self.y_train_resampled, return_counts=True)
        print("\n📊 After SMOTE:")
        for cls, count in zip(unique, counts):
            print(f"   Class {cls}: {count} ({count/len(self.y_train_resampled)*100:.1f}%)")
        
        print(f"\n   Total samples: {len(self.y_train)} → {len(self.y_train_resampled)} (+{len(self.y_train_resampled) - len(self.y_train)})")
        
        return self
    
    def train_baselines(self) -> 'TrainingPipeline':
        """Stage 3: Train and evaluate baseline models."""
        print("\n" + "="*60)
        print("🏁 STAGE 3: BASELINE EVALUATION")
        print("="*60)
        
        from src.models.baselines import (
            MostFrequentBaseline, 
            StratifiedRandomBaseline,
            LogisticRegressionBaseline
        )
        
        baselines = [
            ('Most Frequent', MostFrequentBaseline()),
            ('Stratified Random', StratifiedRandomBaseline()),
            ('Logistic Regression', LogisticRegressionBaseline())
        ]
        
        # Use resampled data if available
        X_train = self.X_train_resampled if self.X_train_resampled is not None else self.X_train
        y_train = self.y_train_resampled if self.y_train_resampled is not None else self.y_train
        
        for name, model in baselines:
            print(f"\n🔧 Training: {name}")
            
            if hasattr(model, 'feature_names'):
                model.fit(X_train, y_train, feature_names=self.feature_names)
            else:
                model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = model.predict(self.X_val)
            macro_f1 = f1_score(self.y_val, y_pred, average='macro')
            accuracy = accuracy_score(self.y_val, y_pred)
            per_class_f1 = f1_score(self.y_val, y_pred, average=None)
            
            self.model_results[name] = {
                'model': model,
                'macro_f1': macro_f1,
                'accuracy': accuracy,
                'per_class_f1': per_class_f1.tolist(),
                'is_baseline': True
            }
            
            print(f"   Macro F1: {macro_f1:.4f}")
            print(f"   Accuracy: {accuracy:.4f}")
        
        return self
    
    def train_tree_models(self, tune: bool = None) -> 'TrainingPipeline':
        """Stage 4: Train tree-based models with optional hyperparameter tuning."""
        print("\n" + "="*60)
        print("🌳 STAGE 4: TREE-BASED MODELS")
        print("="*60)
        
        tune = tune if tune is not None else self.config.TUNE_HYPERPARAMETERS
        
        from src.models.tree_models import RandomForestModel
        
        # Use resampled data if available
        X_train = self.X_train_resampled if self.X_train_resampled is not None else self.X_train
        y_train = self.y_train_resampled if self.y_train_resampled is not None else self.y_train
        
        # Check for optional models
        models = []
        
        # Random Forest
        if tune:
            from sklearn.ensemble import RandomForestClassifier
            rf_base = RandomForestClassifier(
                class_weight='balanced',
                random_state=self.config.RANDOM_STATE,
                n_jobs=2  # Reduced for memory safety
            )
            rf_tuned, rf_params = self.tuner.tune_model(
                rf_base, 
                self.tuner.get_rf_search_space(),
                X_train, y_train,
                'Random Forest',
                n_iter=min(30, self.config.TUNING_N_ITER),
                n_jobs=2  # Reduced parallelism
            )
            # Wrap in our model class
            rf_model = RandomForestModel()
            rf_model.model = rf_tuned
            models.append(('Random Forest (Tuned)', rf_model))
            self.results_tracker.log_best_params('Random Forest', rf_params)
        else:
            models.append(('Random Forest', RandomForestModel()))
        
        # XGBoost
        try:
            from src.models.tree_models import XGBoostModel
            if tune:
                from xgboost import XGBClassifier
                from src.models.tree_models import is_gpu_available
                
                xgb_base_params = {
                    'random_state': self.config.RANDOM_STATE,
                    'n_jobs': 2,  # Reduced for memory safety
                    'eval_metric': 'mlogloss',
                    'verbosity': 0
                }
                if is_gpu_available():
                    xgb_base_params['tree_method'] = 'hist'
                    xgb_base_params['device'] = 'cuda'
                    
                xgb_base = XGBClassifier(**xgb_base_params)
                xgb_tuned, xgb_params = self.tuner.tune_model(
                    xgb_base,
                    self.tuner.get_xgboost_search_space(),
                    X_train, y_train,
                    'XGBoost',
                    n_iter=self.config.TUNING_N_ITER,
                    n_jobs=2  # Reduced parallelism
                )
                xgb_model = XGBoostModel()
                xgb_model.model = xgb_tuned
                models.append(('XGBoost (Tuned)', xgb_model))
                self.results_tracker.log_best_params('XGBoost', xgb_params)
            else:
                models.append(('XGBoost', XGBoostModel()))
        except ImportError:
            print("⚠️ XGBoost not available")
        
        # LightGBM
        try:
            from src.models.tree_models import LightGBMModel
            if tune:
                from lightgbm import LGBMClassifier
                lgbm_base = LGBMClassifier(
                    class_weight='balanced',
                    random_state=self.config.RANDOM_STATE,
                    n_jobs=2,  # Reduced for memory safety
                    verbosity=-1
                )
                lgbm_tuned, lgbm_params = self.tuner.tune_model(
                    lgbm_base,
                    self.tuner.get_lightgbm_search_space(),
                    X_train, y_train,
                    'LightGBM',
                    n_iter=self.config.TUNING_N_ITER,
                    n_jobs=2  # Reduced parallelism
                )
                lgbm_model = LightGBMModel()
                lgbm_model.model = lgbm_tuned
                models.append(('LightGBM (Tuned)', lgbm_model))
                self.results_tracker.log_best_params('LightGBM', lgbm_params)
            else:
                models.append(('LightGBM', LightGBMModel()))
        except ImportError:
            print("⚠️ LightGBM not available")
        
        for name, model in models:
            print(f"\n🔧 {'Training' if not tune else 'Evaluating'}: {name}")
            
            # Train if not already fitted (tuned models are already fitted)
            if not hasattr(model.model, 'classes_'):
                if 'XGBoost' in name and hasattr(model, 'fit'):
                    model.fit(
                        X_train, y_train,
                        feature_names=self.feature_names,
                        eval_set=[(self.X_val, self.y_val)]
                    )
                else:
                    model.fit(X_train, y_train, feature_names=self.feature_names)
            
            # Cross-validation score (skip for tuned models as they already did CV)
            if not tune:
                cv_model = model.model
                if 'XGBoost' in name and hasattr(cv_model, 'get_params'):
                    from sklearn.base import clone
                    cv_model = clone(cv_model)
                    cv_model.set_params(early_stopping_rounds=None)
                
                cv_scores = cross_val_score(
                    cv_model, X_train, y_train,
                    cv=self.config.CV_FOLDS,
                    scoring='f1_macro',
                    n_jobs=-1
                )
                print(f"   CV Macro F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Validation set score
            y_pred = model.predict(self.X_val)
            val_f1 = f1_score(self.y_val, y_pred, average='macro')
            val_acc = accuracy_score(self.y_val, y_pred)
            per_class_f1 = f1_score(self.y_val, y_pred, average=None)
            
            self.model_results[name] = {
                'model': model,
                'macro_f1': val_f1,
                'accuracy': val_acc,
                'per_class_f1': per_class_f1.tolist(),
                'is_baseline': False
            }
            
            print(f"   Val Macro F1: {val_f1:.4f}")
            print(f"   Val Accuracy: {val_acc:.4f}")
            print(f"   Per-class F1: Dropout={per_class_f1[0]:.4f}, Enrolled={per_class_f1[1]:.4f}, Graduate={per_class_f1[2]:.4f}")
            
            # Feature importance
            if hasattr(model, 'get_feature_importance'):
                print(f"\n   🔍 Top 5 Features:")
                importance = model.get_feature_importance()
                for _, row in importance.head(5).iterrows():
                    print(f"      {row['Feature']}: {row['Importance_Pct']:.1f}%")
                
                # Log feature importance for best performing tree model
                if val_f1 >= max([r['macro_f1'] for r in self.model_results.values() if not r.get('is_baseline', True)], default=0):
                    self.results_tracker.log_feature_importance(importance)
        
        return self
    
    def train_ensembles(self, use_weighted_voting: bool = True) -> 'TrainingPipeline':
        """Stage 5: Train ensemble models with optional weighted voting."""
        print("\n" + "="*60)
        print("🎯 STAGE 5: ENSEMBLE MODELS")
        print("="*60)
        
        from src.models.ensembles import VotingEnsemble, StackingEnsemble
        
        # Use resampled data if available
        X_train = self.X_train_resampled if self.X_train_resampled is not None else self.X_train
        y_train = self.y_train_resampled if self.y_train_resampled is not None else self.y_train
        
        ensembles = [
            ('Voting Ensemble', VotingEnsemble(voting='soft', use_calibration=True)),
            ('Stacking Ensemble', StackingEnsemble(meta_model='logistic'))
        ]
        
        # Add Ridge meta-learner stacking for comparison
        ensembles.append(('Stacking (Ridge)', StackingEnsemble(meta_model='ridge')))
        
        for name, model in ensembles:
            print(f"\n🔧 Training: {name}")
            
            model.fit(X_train, y_train)
            
            # Validation set score
            y_pred = model.predict(self.X_val)
            val_f1 = f1_score(self.y_val, y_pred, average='macro')
            val_acc = accuracy_score(self.y_val, y_pred)
            per_class_f1 = f1_score(self.y_val, y_pred, average=None)
            
            self.model_results[name] = {
                'model': model,
                'macro_f1': val_f1,
                'accuracy': val_acc,
                'per_class_f1': per_class_f1.tolist(),
                'is_baseline': False,
                'is_ensemble': True
            }
            
            print(f"   Val Macro F1: {val_f1:.4f}")
            print(f"   Val Accuracy: {val_acc:.4f}")
            print(f"   Per-class F1: Dropout={per_class_f1[0]:.4f}, Enrolled={per_class_f1[1]:.4f}, Graduate={per_class_f1[2]:.4f}")
        
        return self
    
    def select_best_model(self) -> 'TrainingPipeline':
        """Stage 6: Select the best model."""
        print("\n" + "="*60)
        print("🏆 STAGE 6: MODEL SELECTION")
        print("="*60)
        
        # Rank by macro F1
        ranked = sorted(
            self.model_results.items(),
            key=lambda x: x[1]['macro_f1'],
            reverse=True
        )
        
        print("\n📊 Model Rankings (by Macro F1):")
        print("-" * 50)
        
        for i, (name, results) in enumerate(ranked, 1):
            baseline = " (baseline)" if results.get('is_baseline', False) else ""
            ensemble = " (ensemble)" if results.get('is_ensemble', False) else ""
            print(f"{i}. {name}{baseline}{ensemble}")
            print(f"   Macro F1: {results['macro_f1']:.4f}")
            print(f"   Accuracy: {results['accuracy']:.4f}")
        
        # Select best non-baseline model
        for name, results in ranked:
            if not results.get('is_baseline', False):
                self.best_model = results['model']
                self.best_model_name = name
                break
        
        print(f"\n🏆 Selected model: {self.best_model_name}")
        
        # Log baseline metrics (using Logistic Regression as baseline reference)
        if 'Logistic Regression' in self.model_results:
            baseline_results = self.model_results['Logistic Regression']
            self.results_tracker.log_baseline({
                'macro_f1': baseline_results['macro_f1'],
                'accuracy': baseline_results['accuracy'],
                'per_class_f1': baseline_results['per_class_f1'],
                'model_name': 'Logistic Regression'
            })
        
        # Log current best model as a phase result
        best_results = self.model_results[self.best_model_name]
        self.results_tracker.log_phase(
            "Model Training",
            {
                'macro_f1': best_results['macro_f1'],
                'accuracy': best_results['accuracy'],
                'per_class_f1': best_results['per_class_f1']
            },
            f"Best model: {self.best_model_name}"
        )
        
        # Check against thresholds
        best_f1 = self.model_results[self.best_model_name]['macro_f1']
        
        if best_f1 >= self.config.TARGET_MACRO_F1:
            print(f"✅ Exceeds target performance ({self.config.TARGET_MACRO_F1})")
        elif best_f1 >= self.config.MIN_MACRO_F1:
            print(f"✓ Meets minimum requirements ({self.config.MIN_MACRO_F1})")
        else:
            print(f"⚠️ Below minimum threshold ({self.config.MIN_MACRO_F1})")
        
        return self
    
    def optimize_thresholds(self) -> 'TrainingPipeline':
        """Stage 7: Optimize classification thresholds."""
        print("\n" + "="*60)
        print("🎯 STAGE 7: THRESHOLD OPTIMIZATION")
        print("="*60)
        
        if not self.config.OPTIMIZE_THRESHOLDS:
            print("⚠️ Threshold optimization disabled. Skipping.")
            return self
        
        if self.best_model is None:
            print("⚠️ No model selected. Run select_best_model() first.")
            return self
        
        # Check if model supports predict_proba
        if not hasattr(self.best_model, 'predict_proba'):
            print("⚠️ Best model doesn't support predict_proba. Skipping threshold optimization.")
            return self
        
        # Get probabilities on validation set
        try:
            y_proba = self.best_model.predict_proba(self.X_val)
        except Exception as e:
            print(f"⚠️ Could not get probabilities: {e}. Skipping threshold optimization.")
            return self
        
        # Before threshold optimization
        y_pred_before = self.best_model.predict(self.X_val)
        f1_before = f1_score(self.y_val, y_pred_before, average='macro')
        per_class_before = f1_score(self.y_val, y_pred_before, average=None)
        
        print(f"\n📊 Before threshold optimization:")
        print(f"   Macro F1: {f1_before:.4f}")
        print(f"   Per-class F1: Dropout={per_class_before[0]:.4f}, Enrolled={per_class_before[1]:.4f}, Graduate={per_class_before[2]:.4f}")
        
        # Optimize thresholds
        self.threshold_optimizer = ThresholdOptimizer(n_classes=3, n_thresholds=15)
        self.threshold_optimizer.fit(y_proba, self.y_val)
        
        # After threshold optimization
        y_pred_after = self.threshold_optimizer.predict(y_proba)
        f1_after = f1_score(self.y_val, y_pred_after, average='macro')
        per_class_after = f1_score(self.y_val, y_pred_after, average=None)
        
        print(f"\n📊 After threshold optimization:")
        print(f"   Macro F1: {f1_after:.4f}")
        print(f"   Per-class F1: Dropout={per_class_after[0]:.4f}, Enrolled={per_class_after[1]:.4f}, Graduate={per_class_after[2]:.4f}")
        print(f"   Improvement: +{f1_after - f1_before:.4f}")
        
        # Log threshold optimization phase
        self.results_tracker.log_phase(
            "Threshold Optimization",
            {
                'macro_f1': f1_after,
                'accuracy': accuracy_score(self.y_val, y_pred_after),
                'per_class_f1': per_class_after.tolist()
            },
            f"Optimized thresholds: {self.threshold_optimizer.optimal_thresholds}"
        )
        
        return self
    
    def final_evaluation(self) -> Dict[str, Any]:
        """Stage 8: Final evaluation on test set."""
        print("\n" + "="*60)
        print("📋 STAGE 8: FINAL TEST SET EVALUATION")
        print("="*60)
        
        if self.best_model is None:
            raise ValueError("No model selected. Run select_best_model() first.")
        
        # Predict on held-out test set
        if self.threshold_optimizer is not None and hasattr(self.best_model, 'predict_proba'):
            y_proba = self.best_model.predict_proba(self.X_test)
            y_pred = self.threshold_optimizer.predict(y_proba)
            print("   Using optimized thresholds for predictions")
        else:
            y_pred = self.best_model.predict(self.X_test)
        
        # Comprehensive metrics
        results = {
            'model_name': self.best_model_name,
            'test_macro_f1': f1_score(self.y_test, y_pred, average='macro'),
            'test_weighted_f1': f1_score(self.y_test, y_pred, average='weighted'),
            'test_accuracy': accuracy_score(self.y_test, y_pred),
            'test_per_class_f1': f1_score(self.y_test, y_pred, average=None).tolist(),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n🎯 Test Set Results for {self.best_model_name}:")
        print(f"   Macro F1: {results['test_macro_f1']:.4f}")
        print(f"   Weighted F1: {results['test_weighted_f1']:.4f}")
        print(f"   Accuracy: {results['test_accuracy']:.4f}")
        
        print(f"\n📊 Per-class F1 Scores:")
        for i, (name, f1) in enumerate(zip(self.config.CLASS_NAMES, results['test_per_class_f1'])):
            print(f"   {name}: {f1:.4f}")
        
        print(f"\n📊 Classification Report:")
        print(classification_report(
            self.y_test, y_pred, 
            target_names=self.config.CLASS_NAMES
        ))
        
        # Log final results
        self.results_tracker.log_phase(
            "Final Evaluation (Test Set)",
            {
                'macro_f1': results['test_macro_f1'],
                'accuracy': results['test_accuracy'],
                'per_class_f1': results['test_per_class_f1']
            },
            "Final evaluation on held-out test set"
        )
        
        # Log confusion matrix
        self.results_tracker.log_confusion_matrix(
            "Final",
            np.array(results['confusion_matrix']),
            self.config.CLASS_NAMES
        )
        
        return results
    
    def save_model(self, save_path: str = None) -> str:
        """Save the best model to disk."""
        if save_path is None:
            self.config.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
            save_path = self.config.MODEL_SAVE_PATH / f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        
        joblib.dump({
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'threshold_optimizer': self.threshold_optimizer,
            'config': {
                'class_names': self.config.CLASS_NAMES,
                'target_col': self.config.TARGET_COL
            }
        }, save_path)
        
        print(f"\n💾 Model saved to: {save_path}")
        return str(save_path)
    
    def generate_report(self, output_path: str = "REPORT.md") -> str:
        """Generate comprehensive report."""
        return self.results_tracker.generate_report(output_path)
    
    def run_full_pipeline(self, filepath: str, generate_report: bool = True) -> Dict[str, Any]:
        """
        Run the complete pipeline from data to trained model.
        
        This is the main entry point for training.
        """
        print("🚀 STUDENT SUCCESS PREDICTION - FULL TRAINING PIPELINE")
        print("=" * 60)
        print(f"📋 Configuration:")
        print(f"   Feature Engineering: {self.config.USE_FEATURE_ENGINEERING}")
        print(f"   SMOTE: {self.config.USE_SMOTE}")
        print(f"   Hyperparameter Tuning: {self.config.TUNE_HYPERPARAMETERS}")
        print(f"   Threshold Optimization: {self.config.OPTIMIZE_THRESHOLDS}")
        
        # Execute all stages
        self.load_and_prepare(filepath)
        self.apply_smote()
        self.train_baselines()
        self.train_tree_models(tune=self.config.TUNE_HYPERPARAMETERS)
        self.train_ensembles()
        self.select_best_model()
        self.optimize_thresholds()
        results = self.final_evaluation()
        
        # Generate report
        if generate_report:
            self.generate_report()
        
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETE")
        print("=" * 60)
        
        return results


# =============================================================================
# QUICK START FUNCTION
# =============================================================================

def train_student_success_model(data_path: str, 
                                save_model: bool = True,
                                tune_hyperparameters: bool = True,
                                use_smote: bool = True,
                                use_feature_engineering: bool = True,
                                optimize_thresholds: bool = True) -> Tuple[Any, Dict]:
    """
    Quick start function to train a student success prediction model.
    
    Usage:
    ------
    model, results = train_student_success_model('data/dataset.csv')
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset CSV file
        
    save_model : bool
        Whether to save the trained model to disk
        
    tune_hyperparameters : bool
        Whether to run hyperparameter tuning (adds ~30-40 min)
        
    use_smote : bool
        Whether to use SMOTE for class imbalance
        
    use_feature_engineering : bool
        Whether to apply feature engineering
        
    optimize_thresholds : bool
        Whether to optimize classification thresholds
        
    Returns:
    --------
    model : trained model object
    results : dictionary with evaluation metrics
    """
    config = PipelineConfig()
    config.TUNE_HYPERPARAMETERS = tune_hyperparameters
    config.USE_SMOTE = use_smote
    config.USE_FEATURE_ENGINEERING = use_feature_engineering
    config.OPTIMIZE_THRESHOLDS = optimize_thresholds
    
    pipeline = TrainingPipeline(config)
    results = pipeline.run_full_pipeline(data_path)
    
    if save_model:
        pipeline.save_model()
    
    return pipeline.best_model, results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of the training pipeline.
    
    In production, you would:
    1. Set the correct data path
    2. Run the pipeline
    3. Deploy the saved model
    """
    
    print("""
    🎓 Student Success Prediction - Training Pipeline
    
    To use this pipeline:
    
    1. Ensure your dataset is available:
       data/dataset.csv
       
    2. Run the pipeline:
       python train_pipeline.py
       
    3. Or in Python:
       from train_pipeline import train_student_success_model
       model, results = train_student_success_model('data/dataset.csv')
    
    The pipeline will:
    - Validate your data
    - Apply feature engineering (20+ new features)
    - Handle class imbalance with SMOTE
    - Train and tune multiple model types
    - Optimize classification thresholds
    - Select the best performer
    - Evaluate on held-out test set
    - Generate REPORT.md with all results
    - Save the model for deployment
    """)
