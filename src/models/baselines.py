"""
Student Success Prediction - Baseline Models
=============================================

This module implements baseline models that serve as performance floors.
Think of baselines like the "control group" in an experiment - they help us
understand if our fancy models are actually learning anything useful.

Author: ML Engineering Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# BASELINE MODEL 1: Most Frequent Classifier
# =============================================================================

class MostFrequentBaseline:
    """
    The Simplest Possible Model - Always Predicts the Most Common Class
    
    WHY THIS MATTERS:
    -----------------
    Imagine a doctor who diagnoses every patient with "common cold" because 
    it's the most frequent diagnosis. They'd be right surprisingly often, 
    but they'd miss every serious illness.
    
    This baseline tells us: "What's the minimum performance we'd get 
    by just predicting the majority class?"
    
    If our model can't beat this, we have a problem.
    
    EXPECTED PERFORMANCE:
    --------------------
    - Accuracy = % of majority class (e.g., 60% if 60% are Graduates)
    - Macro F1 ≈ 0.2-0.3 (poor by design - it ignores minority classes)
    - Minority class recall = 0 (it NEVER predicts minority classes)
    """
    
    def __init__(self):
        self.model = DummyClassifier(strategy='most_frequent')
        self.name = "Most Frequent Baseline"
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MostFrequentBaseline':
        """Learn which class is most frequent."""
        self.model.fit(X, y)
        self.majority_class_ = self.model.class_prior_.argmax()
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Always predict the majority class."""
        return self.model.predict(X)
    
    def get_insight(self) -> str:
        """Explain what this baseline teaches us."""
        return (
            f"This model always predicts class {self.majority_class_}. "
            f"Any useful model must significantly outperform this."
        )


# =============================================================================
# BASELINE MODEL 2: Stratified Random Classifier
# =============================================================================

class StratifiedRandomBaseline:
    """
    Random Guessing That Respects Class Proportions
    
    WHY THIS MATTERS:
    -----------------
    This is like a student who hasn't studied but knows that 60% of 
    multiple choice answers are "C" - they guess randomly but weighted 
    toward common answers.
    
    This baseline tells us: "What performance would random guessing achieve 
    if we knew the class distribution?"
    
    EXPECTED PERFORMANCE:
    --------------------
    - For balanced 3-class problem: ~33% accuracy, ~0.33 Macro F1
    - For imbalanced: accuracy matches majority %, but all classes get some predictions
    
    KEY INSIGHT:
    -----------
    The gap between Most Frequent and Stratified Random baselines shows 
    how much class imbalance affects "naive" strategies.
    """
    
    def __init__(self, random_state: int = 42):
        self.model = DummyClassifier(strategy='stratified', random_state=random_state)
        self.name = "Stratified Random Baseline"
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StratifiedRandomBaseline':
        """Learn class proportions for weighted random guessing."""
        self.model.fit(X, y)
        self.class_proportions_ = self.model.class_prior_
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Randomly guess according to class proportions."""
        return self.model.predict(X)
    
    def get_insight(self) -> str:
        """Explain what this baseline teaches us."""
        props = [f"Class {i}: {p:.1%}" for i, p in enumerate(self.class_proportions_)]
        return (
            f"This model randomly guesses with probabilities: {', '.join(props)}. "
            f"It represents 'informed random guessing'."
        )


# =============================================================================
# BASELINE MODEL 3: Logistic Regression (The First "Real" Model)
# =============================================================================

class LogisticRegressionBaseline:
    """
    The First Interpretable Learning Model
    
    WHY THIS MATTERS:
    -----------------
    Logistic Regression is like a weighted checklist:
    - Good grades? +10 points toward graduation
    - Financial debt? -5 points toward graduation
    - Sum up all points, convert to probability
    
    It's simple but powerful because:
    1. Coefficients directly show feature importance
    2. Fast to train and predict
    3. Often surprisingly competitive with complex models
    4. Serves as the "linear" baseline - if LR does well, relationships are mostly linear
    
    HOW IT WORKS (Intuition):
    -------------------------
    For each class, LR learns a linear equation:
    
    score_graduate = β₀ + β₁×grades + β₂×attendance + β₃×debt + ...
    score_dropout  = γ₀ + γ₁×grades + γ₂×attendance + γ₃×debt + ...
    score_enrolled = δ₀ + δ₁×grades + δ₂×attendance + δ₃×debt + ...
    
    Then converts scores to probabilities using softmax:
    P(graduate) = exp(score_graduate) / sum(exp(all_scores))
    
    INTERPRETING COEFFICIENTS:
    -------------------------
    - Positive coefficient: Feature increases probability of that class
    - Negative coefficient: Feature decreases probability of that class
    - Larger magnitude: Stronger effect
    
    EXPECTED PERFORMANCE:
    --------------------
    - For well-preprocessed tabular data: 0.65-0.75 Macro F1
    - If it achieves >0.80: Congratulations, you might not need fancier models!
    """
    
    def __init__(self, 
                 max_iter: int = 1000,
                 class_weight: str = 'balanced',
                 random_state: int = 42,
                 C: float = 1.0):
        """
        Parameters:
        -----------
        max_iter : int
            Maximum iterations for solver convergence
            
        class_weight : str or dict
            'balanced' automatically adjusts weights inversely proportional 
            to class frequencies. This is crucial for imbalanced datasets.
            
        C : float
            Inverse of regularization strength. Smaller = more regularization.
            Think of it as "how much to trust the data vs. keep coefficients small"
        """
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),  # LR needs scaled features!
            ('classifier', LogisticRegression(
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=random_state,
                C=C,
                multi_class='multinomial',  # Native multi-class handling
                solver='lbfgs'
            ))
        ])
        self.name = "Logistic Regression Baseline"
        self.feature_names = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> 'LogisticRegressionBaseline':
        """
        Train the logistic regression model.
        
        The scaler learns mean and std from training data.
        The classifier learns coefficients that best separate classes.
        """
        self.pipeline.fit(X, y)
        self.feature_names = feature_names
        self.classes_ = self.pipeline.named_steps['classifier'].classes_
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities - useful for understanding confidence."""
        return self.pipeline.predict_proba(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extract and rank feature importances.
        
        For multi-class LR, we get coefficients for each class.
        Larger absolute values = more important features.
        """
        clf = self.pipeline.named_steps['classifier']
        
        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(clf.coef_.shape[1])]
        else:
            feature_names = self.feature_names
            
        # For multi-class, coefficients are shape (n_classes, n_features)
        importance_df = pd.DataFrame(
            clf.coef_.T,
            index=feature_names,
            columns=[f'Class_{c}' for c in self.classes_]
        )
        
        # Add overall importance (mean absolute coefficient across classes)
        importance_df['Overall_Importance'] = np.abs(clf.coef_).mean(axis=0)
        importance_df = importance_df.sort_values('Overall_Importance', ascending=False)
        
        return importance_df
    
    def get_insight(self) -> str:
        """Explain what this model teaches us."""
        return (
            "Logistic Regression finds linear decision boundaries. "
            "If it performs well, complex non-linear models may offer diminishing returns. "
            "Check feature importance to understand what drives predictions."
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def evaluate_baseline(model, X_test: np.ndarray, y_test: np.ndarray,
                     class_names: List[str] = None) -> Dict:
    """
    Comprehensive evaluation of a baseline model.
    
    Returns a dictionary with all relevant metrics for comparison.
    """
    y_pred = model.predict(X_test)
    
    # Handle class names
    if class_names is None:
        class_names = ['Dropout', 'Enrolled', 'Graduate']
    
    # Calculate metrics
    results = {
        'model_name': model.name,
        'accuracy': (y_pred == y_test).mean(),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'weighted_f1': f1_score(y_test, y_pred, average='weighted'),
        'per_class_f1': f1_score(y_test, y_pred, average=None),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True
        )
    }
    
    return results


def compare_baselines(X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     feature_names: List[str] = None) -> pd.DataFrame:
    """
    Train and compare all baseline models.
    
    This function runs all baselines and creates a comparison table.
    Use this to establish the performance floor before trying complex models.
    
    Returns:
    --------
    DataFrame with columns: Model, Accuracy, Macro_F1, Weighted_F1, 
                           Dropout_F1, Enrolled_F1, Graduate_F1
    """
    # Initialize baselines
    baselines = [
        MostFrequentBaseline(),
        StratifiedRandomBaseline(),
        LogisticRegressionBaseline()
    ]
    
    results = []
    
    for baseline in baselines:
        print(f"\n{'='*50}")
        print(f"Training: {baseline.name}")
        print('='*50)
        
        # Fit the model
        if isinstance(baseline, LogisticRegressionBaseline):
            baseline.fit(X_train, y_train, feature_names=feature_names)
        else:
            baseline.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_baseline(baseline, X_test, y_test)
        
        # Store results
        row = {
            'Model': baseline.name,
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'Macro_F1': f"{metrics['macro_f1']:.3f}",
            'Weighted_F1': f"{metrics['weighted_f1']:.3f}",
            'Dropout_F1': f"{metrics['per_class_f1'][0]:.3f}",
            'Enrolled_F1': f"{metrics['per_class_f1'][1]:.3f}",
            'Graduate_F1': f"{metrics['per_class_f1'][2]:.3f}"
        }
        results.append(row)
        
        # Print insight
        print(f"\n📊 {baseline.get_insight()}")
        
        # For LR, show top features
        if isinstance(baseline, LogisticRegressionBaseline):
            print("\n🔍 Top 5 Most Important Features:")
            importance = baseline.get_feature_importance()
            print(importance.head())
    
    return pd.DataFrame(results)


# =============================================================================
# DEMONSTRATION / USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    """
    Example usage demonstrating how to use baseline models.
    
    In a real scenario, you would:
    1. Load your actual dataset
    2. Split into train/test
    3. Run this comparison
    4. Use the results to set expectations for complex models
    """
    
    # Create synthetic data for demonstration
    # Replace this with actual data loading in production
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Simulate features (in reality, these come from your dataset)
    X = np.random.randn(n_samples, n_features)
    
    # Simulate imbalanced classes: 50% Graduate, 30% Dropout, 20% Enrolled
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.30, 0.20, 0.50])
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Generate feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    print("🎓 STUDENT SUCCESS PREDICTION - BASELINE COMPARISON")
    print("=" * 60)
    print(f"\nDataset: {n_samples} samples, {n_features} features")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Train/Test split: {len(X_train)}/{len(X_test)}")
    
    # Run comparison
    comparison_df = compare_baselines(
        X_train, y_train, X_test, y_test, feature_names
    )
    
    print("\n" + "=" * 60)
    print("📈 BASELINE COMPARISON SUMMARY")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("🎯 KEY INSIGHTS")
    print("=" * 60)
    print("""
    1. Most Frequent Baseline: Sets the absolute floor. If your model 
       can't beat this, it's not learning anything.
       
    2. Stratified Random: Shows what "informed guessing" achieves.
       The gap between this and Most Frequent shows class imbalance impact.
       
    3. Logistic Regression: The first "real" model. If it achieves 
       strong performance, you may not need complex models.
       
    NEXT STEPS:
    - If LR Macro F1 < 0.60: Feature engineering needed
    - If LR Macro F1 0.60-0.75: Try Random Forest / XGBoost
    - If LR Macro F1 > 0.75: LR might be your production model!
    """)
