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

Author: ML Engineering Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, 
    confusion_matrix, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')


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
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load dataset from file.
        
        Supports: CSV, Excel, Parquet
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
        
        return X.values, y, list(X.columns)


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
    4. Baseline Evaluation
    5. Model Training with Cross-Validation
    6. Model Selection
    7. Final Evaluation on Test Set
    8. Model Saving
    
    This class ties everything together into a reproducible workflow.
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.data_loader = DataLoader(self.config)
        
        # Storage for results
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.feature_names = None
        
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_and_prepare(self, filepath: str) -> 'TrainingPipeline':
        """Stage 1: Load and prepare data."""
        print("\n" + "="*60)
        print("📂 STAGE 1: DATA LOADING AND VALIDATION")
        print("="*60)
        
        df = self.data_loader.load_data(filepath)
        
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
    
    def train_baselines(self) -> 'TrainingPipeline':
        """Stage 2: Train and evaluate baseline models."""
        print("\n" + "="*60)
        print("🏁 STAGE 2: BASELINE EVALUATION")
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
        
        for name, model in baselines:
            print(f"\n🔧 Training: {name}")
            
            if hasattr(model, 'feature_names'):
                model.fit(self.X_train, self.y_train, feature_names=self.feature_names)
            else:
                model.fit(self.X_train, self.y_train)
            
            # Evaluate on validation set
            y_pred = model.predict(self.X_val)
            macro_f1 = f1_score(self.y_val, y_pred, average='macro')
            accuracy = accuracy_score(self.y_val, y_pred)
            
            self.model_results[name] = {
                'model': model,
                'macro_f1': macro_f1,
                'accuracy': accuracy,
                'is_baseline': True
            }
            
            print(f"   Macro F1: {macro_f1:.4f}")
            print(f"   Accuracy: {accuracy:.4f}")
        
        return self
    
    def train_tree_models(self) -> 'TrainingPipeline':
        """Stage 3: Train tree-based models."""
        print("\n" + "="*60)
        print("🌳 STAGE 3: TREE-BASED MODELS")
        print("="*60)
        
        from src.models.tree_models import RandomForestModel
        
        # Check for optional models
        models = [
            ('Random Forest', RandomForestModel())
        ]
        
        try:
            from src.models.tree_models import XGBoostModel
            models.append(('XGBoost', XGBoostModel()))
        except ImportError:
            print("⚠️ XGBoost not available")
        
        try:
            from src.models.tree_models import LightGBMModel
            models.append(('LightGBM', LightGBMModel()))
        except ImportError:
            print("⚠️ LightGBM not available")
        
        for name, model in models:
            print(f"\n🔧 Training: {name}")
            
            # For XGBoost, use early stopping
            if name == 'XGBoost' and hasattr(model, 'fit'):
                model.fit(
                    self.X_train, self.y_train,
                    feature_names=self.feature_names,
                    eval_set=[(self.X_val, self.y_val)],
                    early_stopping_rounds=50
                )
            else:
                model.fit(self.X_train, self.y_train, feature_names=self.feature_names)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                model.model, self.X_train, self.y_train,
                cv=self.config.CV_FOLDS,
                scoring='f1_macro',
                n_jobs=-1
            )
            
            # Validation set score
            y_pred = model.predict(self.X_val)
            val_f1 = f1_score(self.y_val, y_pred, average='macro')
            val_acc = accuracy_score(self.y_val, y_pred)
            
            self.model_results[name] = {
                'model': model,
                'macro_f1': val_f1,
                'accuracy': val_acc,
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'is_baseline': False
            }
            
            print(f"   CV Macro F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"   Val Macro F1: {val_f1:.4f}")
            print(f"   Val Accuracy: {val_acc:.4f}")
            
            # Feature importance
            print(f"\n   🔍 Top 5 Features:")
            importance = model.get_feature_importance()
            for _, row in importance.head(5).iterrows():
                print(f"      {row['Feature']}: {row['Importance_Pct']:.1f}%")
        
        return self
    
    def train_ensembles(self) -> 'TrainingPipeline':
        """Stage 4: Train ensemble models."""
        print("\n" + "="*60)
        print("🎯 STAGE 4: ENSEMBLE MODELS")
        print("="*60)
        
        from src.models.ensembles import VotingEnsemble, StackingEnsemble
        
        ensembles = [
            ('Voting Ensemble', VotingEnsemble(voting='soft')),
            ('Stacking Ensemble', StackingEnsemble(meta_model='logistic'))
        ]
        
        for name, model in ensembles:
            print(f"\n🔧 Training: {name}")
            
            model.fit(self.X_train, self.y_train)
            
            # Validation set score
            y_pred = model.predict(self.X_val)
            val_f1 = f1_score(self.y_val, y_pred, average='macro')
            val_acc = accuracy_score(self.y_val, y_pred)
            
            self.model_results[name] = {
                'model': model,
                'macro_f1': val_f1,
                'accuracy': val_acc,
                'is_baseline': False,
                'is_ensemble': True
            }
            
            print(f"   Val Macro F1: {val_f1:.4f}")
            print(f"   Val Accuracy: {val_acc:.4f}")
        
        return self
    
    def select_best_model(self) -> 'TrainingPipeline':
        """Stage 5: Select the best model."""
        print("\n" + "="*60)
        print("🏆 STAGE 5: MODEL SELECTION")
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
        
        # Check against thresholds
        best_f1 = self.model_results[self.best_model_name]['macro_f1']
        
        if best_f1 >= self.config.TARGET_MACRO_F1:
            print(f"✅ Exceeds target performance ({self.config.TARGET_MACRO_F1})")
        elif best_f1 >= self.config.MIN_MACRO_F1:
            print(f"✓ Meets minimum requirements ({self.config.MIN_MACRO_F1})")
        else:
            print(f"⚠️ Below minimum threshold ({self.config.MIN_MACRO_F1})")
        
        return self
    
    def final_evaluation(self) -> Dict[str, Any]:
        """Stage 6: Final evaluation on test set."""
        print("\n" + "="*60)
        print("📋 STAGE 6: FINAL TEST SET EVALUATION")
        print("="*60)
        
        if self.best_model is None:
            raise ValueError("No model selected. Run select_best_model() first.")
        
        # Predict on held-out test set
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
            'config': {
                'class_names': self.config.CLASS_NAMES,
                'target_col': self.config.TARGET_COL
            }
        }, save_path)
        
        print(f"\n💾 Model saved to: {save_path}")
        return str(save_path)
    
    def run_full_pipeline(self, filepath: str) -> Dict[str, Any]:
        """
        Run the complete pipeline from data to trained model.
        
        This is the main entry point for training.
        """
        print("🚀 STUDENT SUCCESS PREDICTION - FULL TRAINING PIPELINE")
        print("=" * 60)
        
        # Execute all stages
        self.load_and_prepare(filepath)
        self.train_baselines()
        self.train_tree_models()
        self.train_ensembles()
        self.select_best_model()
        results = self.final_evaluation()
        
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETE")
        print("=" * 60)
        
        return results


# =============================================================================
# QUICK START FUNCTION
# =============================================================================

def train_student_success_model(data_path: str, 
                                save_model: bool = True) -> Tuple[Any, Dict]:
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
        
    Returns:
    --------
    model : trained model object
    results : dictionary with evaluation metrics
    """
    pipeline = TrainingPipeline()
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
    - Train multiple model types
    - Select the best performer
    - Evaluate on held-out test set
    - Save the model for deployment
    """)
