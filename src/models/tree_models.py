"""
Student Success Prediction - Tree-Based Models
===============================================

This module implements tree-based models, which are typically the best performers
for medium-sized tabular datasets like ours. Think of decision trees as a series
of yes/no questions that eventually lead to a prediction.

Why Tree-Based Models Excel Here:
--------------------------------
1. Handle mixed data types (numerical + categorical) naturally
2. Capture non-linear relationships without explicit feature engineering
3. Robust to outliers (trees split on rank, not raw values)
4. Provide interpretable feature importance
5. Work well with limited data (unlike deep learning)

Author: ML Engineering Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Try to import XGBoost and LightGBM (they're optional but recommended)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not installed. Install with: pip install lightgbm")

warnings.filterwarnings('ignore')

def is_gpu_available() -> bool:
    """Detect if a GPU is available for training."""
    # Method 1: Check via PyTorch (most reliable if installed)
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except ImportError:
        pass
        
    # Method 2: Check via nvidia-smi command
    import subprocess
    try:
        subprocess.check_output('nvidia-smi')
        return True
    except Exception:
        pass
        
    return False


# =============================================================================
# MODEL 1: RANDOM FOREST CLASSIFIER
# =============================================================================

class RandomForestModel:
    """
    Random Forest - The "Wisdom of Crowds" Approach
    
    THE CORE IDEA (An Analogy):
    ---------------------------
    Imagine you're trying to predict if a student will succeed. You could ask:
    - One expert advisor who sees all data → Might overfit to specific cases
    - 100 different advisors, each with partial data → Wisdom of crowds!
    
    Random Forest builds many decision trees, each trained on:
    - A random subset of students (bootstrap sampling)
    - A random subset of features at each split
    
    Final prediction = majority vote of all trees.
    
    WHY IT WORKS:
    -------------
    1. Each tree has "blind spots" (doesn't see all data/features)
    2. But different trees have DIFFERENT blind spots
    3. Errors of individual trees cancel out in the ensemble
    4. Result: Lower variance, better generalization
    
    THE "RANDOM" PARTS:
    ------------------
    1. Bootstrap Sampling: Each tree trains on ~63% of unique samples
       (some samples appear multiple times, some not at all)
    
    2. Feature Subsampling: At each split, only consider √n_features
       (forces trees to find diverse splitting patterns)
    
    WHEN TO USE:
    -----------
    ✅ Medium-sized tabular data (100 - 100,000 samples)
    ✅ When you need feature importance rankings
    ✅ When you want a reliable, low-maintenance model
    ✅ When interpretability matters
    
    WHEN TO BE CAUTIOUS:
    -------------------
    ⚠️ Very high-dimensional sparse data (text, images)
    ⚠️ When you need probability calibration
    ⚠️ Real-time prediction with strict latency requirements
    """
    
    def __init__(self,
                 n_estimators: int = 200,
                 max_depth: Optional[int] = None,
                 min_samples_leaf: int = 5,
                 class_weight: str = 'balanced',
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Initialize Random Forest with sensible defaults.
        
        Parameters Explained:
        --------------------
        n_estimators : int (default=200)
            Number of trees in the forest.
            - More trees = more stable predictions, but slower
            - 100-500 is usually the sweet spot
            - Beyond 500, diminishing returns
            
        max_depth : int or None (default=None)
            Maximum depth of each tree.
            - None = trees grow until leaves are pure
            - Limiting depth adds regularization (prevents overfitting)
            - Start with None, reduce if overfitting
            
        min_samples_leaf : int (default=5)
            Minimum samples required in a leaf node.
            - Higher value = more regularization
            - Prevents creating leaves for single outliers
            - Good range: 1-20
            
        class_weight : str (default='balanced')
            How to weight classes.
            - 'balanced' = inversely proportional to class frequency
            - Critical for imbalanced datasets!
            
        n_jobs : int (default=-1)
            Number of CPU cores to use.
            - -1 = use all available cores
            - Trees can be trained in parallel (unlike boosting)
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            oob_score=True  # Out-of-bag score for free validation
        )
        self.name = "Random Forest"
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> 'RandomForestModel':
        """
        Train the Random Forest.
        
        What happens during training:
        1. For each tree:
           a. Sample N students with replacement (bootstrap)
           b. At each node, sample √features to consider
           c. Find best split among those features
           d. Repeat until stopping criteria met
        2. Store all trees for voting
        """
        self.model.fit(X, y)
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        self.is_fitted = True
        
        # OOB score is like free cross-validation!
        print(f"📊 Out-of-Bag Score: {self.model.oob_score_:.3f}")
        print("   (This approximates test set performance without a holdout)")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Each tree votes, majority wins.
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Probabilities = proportion of trees voting for each class.
        """
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, method: str = 'impurity') -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Two methods available:
        
        1. 'impurity' (default): Based on how much each feature reduces 
           impurity (Gini/entropy) across all trees.
           - Fast
           - Can be biased toward high-cardinality features
           
        2. 'permutation': Measure performance drop when feature is shuffled.
           - More reliable
           - Slower (requires additional computation)
        """
        if method == 'impurity':
            importance = self.model.feature_importances_
        else:
            raise NotImplementedError("Permutation importance requires separate computation")
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Normalize to percentages
        importance_df['Importance_Pct'] = (
            importance_df['Importance'] / importance_df['Importance'].sum() * 100
        )
        
        return importance_df.reset_index(drop=True)
    
    def get_hyperparameter_grid(self) -> Dict[str, List]:
        """
        Return recommended hyperparameter grid for tuning.
        
        These ranges are based on practical experience:
        - Start in the middle, expand if needed
        - Don't go too granular initially
        """
        return {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_leaf': [1, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2', 0.5]
        }


# =============================================================================
# MODEL 2: XGBOOST CLASSIFIER
# =============================================================================

if XGBOOST_AVAILABLE:
    class XGBoostModel:
        """
        XGBoost - The "Learn From Mistakes" Approach
        
        THE CORE IDEA (An Analogy):
        ---------------------------
        Imagine a student taking practice exams:
        - Exam 1: Gets geometry wrong
        - Exam 2: Studies geometry, now gets algebra wrong
        - Exam 3: Studies algebra, now gets trigonometry wrong
        - Each exam focuses on previous weak areas
        
        XGBoost builds trees SEQUENTIALLY, where each new tree focuses on 
        correcting the mistakes of all previous trees combined.
        
        GRADIENT BOOSTING INTUITION:
        ---------------------------
        1. Start with a simple prediction (e.g., class probabilities)
        2. Calculate residuals (how wrong we are for each sample)
        3. Train a tree to predict these residuals
        4. Add tree's predictions to running total (with learning rate)
        5. Repeat until convergence
        
        KEY DIFFERENCE FROM RANDOM FOREST:
        ---------------------------------
        | Aspect          | Random Forest      | XGBoost            |
        |-----------------|--------------------|--------------------|
        | Tree Building   | Parallel           | Sequential         |
        | Tree Purpose    | Independent vote   | Fix previous errors|
        | Overfitting     | Resistant          | Prone (needs tuning)|
        | Training Speed  | Faster             | Slower             |
        | Final Model     | Average of trees   | Sum of trees       |
        
        WHY XGBOOST SPECIFICALLY:
        ------------------------
        XGBoost adds several improvements over basic gradient boosting:
        1. Regularization (L1/L2) built into objective function
        2. Column subsampling (like Random Forest)
        3. Efficient handling of missing values
        4. Parallel tree building within each boosting round
        
        WHEN TO USE:
        -----------
        ✅ When you need maximum predictive performance
        ✅ Kaggle competitions (XGBoost wins ~70% of tabular competitions)
        ✅ When you have time for hyperparameter tuning
        ✅ When you need to handle missing values elegantly
        
        WHEN TO BE CAUTIOUS:
        -------------------
        ⚠️ When interpretability is paramount
        ⚠️ When training time is severely constrained
        ⚠️ When you can't do proper hyperparameter tuning
        """
        
        def __init__(self,
                     n_estimators: int = 200,
                     max_depth: int = 6,
                     learning_rate: float = 0.1,
                     subsample: float = 0.8,
                     colsample_bytree: float = 0.8,
                     random_state: int = 42,
                     n_jobs: int = -1):
            """
            Initialize XGBoost with sensible defaults.
            
            Parameters Explained:
            --------------------
            n_estimators : int (default=200)
                Number of boosting rounds.
                - More rounds = more complex model
                - Use early stopping to find optimal value
                
            max_depth : int (default=6)
                Maximum tree depth.
                - XGBoost trees are typically shallower than RF trees
                - 3-10 is common range
                - Deeper = more complex interactions captured
                
            learning_rate : float (default=0.1)
                How much each tree contributes to the ensemble.
                - Smaller = more trees needed, but often better generalization
                - 0.01-0.3 is typical range
                - Trade-off: lower rate + more trees = better but slower
                
            subsample : float (default=0.8)
                Fraction of samples used per tree.
                - Similar to bootstrap in RF
                - Reduces overfitting
                
            colsample_bytree : float (default=0.8)
                Fraction of features used per tree.
                - Similar to max_features in RF
                - Forces diversity among trees
            """
            # Calculate scale_pos_weight for imbalanced classes
            # This will be updated during fit() based on actual class distribution
            
            self.use_gpu = True # Default to auto-detect
            
            # Base parameters (XGBoost 2.0+ removed use_label_encoder)
            xgb_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'random_state': random_state,
                'n_jobs': n_jobs,
                'eval_metric': 'mlogloss',
                'objective': 'multi:softprob',
                'verbosity': 0  # Suppress XGBoost warnings
            }
            
            # GPU Configuration (XGBoost 2.0+ uses device='cuda' instead of tree_method='gpu_hist')
            if self.use_gpu and is_gpu_available():
                print("🚀 GPU detected! Configuring XGBoost to use CUDA.")
                xgb_params['tree_method'] = 'hist'
                xgb_params['device'] = 'cuda'
            else:
                print("🖥️ Using CPU for XGBoost.")
            
            self.model = XGBClassifier(**xgb_params)
            self.name = "XGBoost"
            self.feature_names = None
            
        def fit(self, X: np.ndarray, y: np.ndarray,
                feature_names: Optional[List[str]] = None,
                eval_set: Optional[List[Tuple]] = None) -> 'XGBoostModel':
            """
            Train XGBoost with optional early stopping.
            
            Early Stopping Explained:
            ------------------------
            Rather than guessing number of trees, we:
            1. Train on training data
            2. Monitor performance on validation set
            3. Stop when validation performance stops improving
            
            This automatically finds the optimal number of trees!
            """
            self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
            
            fit_params = {}
            if eval_set is not None:
                fit_params['eval_set'] = eval_set
                fit_params['verbose'] = False
                # Set early_stopping_rounds only when eval_set is provided (XGBoost 2.0+)
                # This prevents CV from failing when no eval_set is available
                self.model.set_params(early_stopping_rounds=50)
            else:
                # Disable early stopping for CV (no eval_set available)
                self.model.set_params(early_stopping_rounds=None)
            
            self.model.fit(X, y, **fit_params)
            
            if eval_set is not None and hasattr(self.model, 'best_iteration'):
                print(f"📊 Best iteration: {self.model.best_iteration}")
            
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict class labels."""
            return self.model.predict(X)
        
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """Predict class probabilities."""
            return self.model.predict_proba(X)
        
        def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
            """
            Get feature importance rankings.
            
            importance_type options:
            - 'gain': Total gain brought by the feature (default, most useful)
            - 'weight': Number of times feature is used for splitting
            - 'cover': Number of samples affected by splits on this feature
            """
            importance = self.model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            importance_df['Importance_Pct'] = (
                importance_df['Importance'] / importance_df['Importance'].sum() * 100
            )
            
            return importance_df.reset_index(drop=True)
        
        def get_hyperparameter_grid(self) -> Dict[str, List]:
            """
            Return recommended hyperparameter grid for tuning.
            
            XGBoost has many hyperparameters. These are the most impactful:
            """
            return {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2]  # Minimum loss reduction for split
            }


# =============================================================================
# MODEL 3: LIGHTGBM CLASSIFIER
# =============================================================================

if LIGHTGBM_AVAILABLE:
    class LightGBMModel:
        """
        LightGBM - The "Fast and Memory-Efficient" Gradient Boosting
        
        WHY LIGHTGBM:
        -------------
        LightGBM is like XGBoost's faster cousin:
        
        1. Leaf-wise growth (vs. level-wise): Grows the leaf that reduces 
           error most, regardless of level. More efficient but higher 
           overfitting risk.
           
        2. Histogram-based splitting: Bins continuous features into discrete 
           buckets, dramatically speeding up split finding.
           
        3. Native categorical support: No need for one-hot encoding!
           LightGBM can use categorical features directly.
        
        XGBOOST vs LIGHTGBM:
        -------------------
        | Aspect            | XGBoost          | LightGBM          |
        |-------------------|------------------|-------------------|
        | Growth Strategy   | Level-wise       | Leaf-wise         |
        | Speed             | Slower           | Faster            |
        | Memory            | Higher           | Lower             |
        | Categorical       | Needs encoding   | Native support    |
        | Overfitting Risk  | Lower            | Higher            |
        | Accuracy          | Similar          | Similar           |
        
        WHEN TO USE:
        -----------
        ✅ Large datasets where training time matters
        ✅ When you have many categorical features
        ✅ When memory is constrained
        ✅ Quick experimentation during development
        
        WHEN TO PREFER XGBOOST:
        ----------------------
        ⚠️ When dataset is small (higher overfitting risk with leaf-wise)
        ⚠️ When you need most stable out-of-the-box performance
        ⚠️ When deploying to systems that already have XGBoost
        """
        
        def __init__(self,
                     n_estimators: int = 200,
                     max_depth: int = -1,  # -1 means no limit
                     num_leaves: int = 31,
                     learning_rate: float = 0.1,
                     subsample: float = 0.8,
                     colsample_bytree: float = 0.8,
                     random_state: int = 42,
                     n_jobs: int = -1,
                     class_weight: str = 'balanced'):
            """
            Initialize LightGBM with sensible defaults.
            
            Key Parameter: num_leaves
            ------------------------
            This is the main parameter controlling model complexity in LightGBM.
            
            Rule of thumb: num_leaves < 2^max_depth
            
            Example: max_depth=5 → max leaves = 32
            So num_leaves=31 with max_depth=-1 (unlimited) is reasonable.
            """
            self.use_gpu = True # Default to auto-detect
            
            lgbm_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'num_leaves': num_leaves,
                'learning_rate': learning_rate,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'random_state': random_state,
                'n_jobs': n_jobs,
                'class_weight': class_weight,
                'verbose': -1
            }
            
            # GPU Configuration
            # Note: LightGBM GPU support requires specific installation build
            if self.use_gpu and is_gpu_available():
                print("🚀 GPU detected! Attempting to use LightGBM with CUDA.")
                lgbm_params['device'] = 'gpu'
            else:
                print("🖥️ Using CPU for LightGBM.")

            self.model = LGBMClassifier(**lgbm_params)
            self.name = "LightGBM"
            self.feature_names = None
            
        def fit(self, X: np.ndarray, y: np.ndarray,
                feature_names: Optional[List[str]] = None,
                categorical_features: Optional[List[int]] = None) -> 'LightGBMModel':
            """
            Train LightGBM.
            
            categorical_features: List of column indices that are categorical.
            LightGBM can use these directly without one-hot encoding!
            """
            self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
            
            fit_params = {}
            if categorical_features:
                fit_params['categorical_feature'] = categorical_features
            
            self.model.fit(X, y, **fit_params)
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict class labels."""
            return self.model.predict(X)
        
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """Predict class probabilities."""
            return self.model.predict_proba(X)
        
        def get_feature_importance(self) -> pd.DataFrame:
            """Get feature importance rankings."""
            importance = self.model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            importance_df['Importance_Pct'] = (
                importance_df['Importance'] / importance_df['Importance'].sum() * 100
            )
            
            return importance_df.reset_index(drop=True)
        
        def get_hyperparameter_grid(self) -> Dict[str, List]:
            """Return recommended hyperparameter grid for tuning."""
            return {
                'n_estimators': [100, 200, 300, 500],
                'num_leaves': [15, 31, 63, 127],
                'max_depth': [-1, 5, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_samples': [5, 10, 20]
            }


# =============================================================================
# HYPERPARAMETER TUNING UTILITY
# =============================================================================

class TreeModelTuner:
    """
    Hyperparameter tuning utilities for tree-based models.
    
    HYPERPARAMETER TUNING EXPLAINED:
    --------------------------------
    Model performance depends heavily on hyperparameters. Finding the best 
    combination is like finding the best recipe - you need to try variations.
    
    TWO MAIN APPROACHES:
    
    1. Grid Search: Try ALL combinations
       - Exhaustive but slow
       - Good for small grids (< 100 combinations)
       - Use when you know roughly where optimal values are
       
    2. Randomized Search: Try RANDOM combinations
       - Sample from distributions
       - Good for large spaces (> 100 combinations)
       - Often finds good solutions faster
       
    IMPORTANT: Always use cross-validation during tuning!
    Otherwise you're optimizing for one specific train/val split.
    """
    
    @staticmethod
    def tune_model(model: Any,
                   X: np.ndarray,
                   y: np.ndarray,
                   param_grid: Dict[str, List],
                   method: str = 'random',
                   n_iter: int = 50,
                   cv: int = 5,
                   scoring: str = 'f1_macro',
                   n_jobs: int = -1) -> Tuple[Any, Dict]:
        """
        Tune model hyperparameters.
        
        Parameters:
        -----------
        model : sklearn-compatible model
            The model to tune
            
        param_grid : dict
            Hyperparameter search space
            
        method : str ('grid' or 'random')
            Search strategy
            
        n_iter : int
            Number of random combinations to try (only for random search)
            
        cv : int
            Number of cross-validation folds
            
        scoring : str
            Metric to optimize ('f1_macro' recommended for imbalanced data)
            
        Returns:
        --------
        best_model : fitted model with best parameters
        best_params : dict of best hyperparameters
        """
        print(f"🔧 Starting hyperparameter tuning ({method} search)")
        print(f"   Combinations to try: {n_iter if method == 'random' else 'all'}")
        print(f"   Cross-validation folds: {cv}")
        print(f"   Scoring metric: {scoring}")
        
        if method == 'grid':
            search = GridSearchCV(
                model.model if hasattr(model, 'model') else model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1,
                return_train_score=True
            )
        else:
            search = RandomizedSearchCV(
                model.model if hasattr(model, 'model') else model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1,
                random_state=42,
                return_train_score=True
            )
        
        search.fit(X, y)
        
        print(f"\n✅ Best {scoring}: {search.best_score_:.4f}")
        print(f"📊 Best parameters:")
        for param, value in search.best_params_.items():
            print(f"   {param}: {value}")
        
        # Check for overfitting
        results = pd.DataFrame(search.cv_results_)
        best_idx = search.best_index_
        train_score = results.loc[best_idx, 'mean_train_score']
        test_score = results.loc[best_idx, 'mean_test_score']
        
        if train_score - test_score > 0.05:
            print(f"\n⚠️ Warning: Possible overfitting detected!")
            print(f"   Train score: {train_score:.4f}")
            print(f"   CV test score: {test_score:.4f}")
            print(f"   Gap: {train_score - test_score:.4f}")
        
        return search.best_estimator_, search.best_params_


# =============================================================================
# MODEL COMPARISON UTILITY
# =============================================================================

def compare_tree_models(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        feature_names: List[str] = None,
                        tune: bool = False) -> pd.DataFrame:
    """
    Train and compare all available tree-based models.
    
    This is your main function for exploring which tree model works best.
    
    Parameters:
    -----------
    tune : bool
        Whether to perform hyperparameter tuning. 
        Set to True for final model selection, False for quick comparison.
    """
    models = [
        RandomForestModel()
    ]
    
    if XGBOOST_AVAILABLE:
        models.append(XGBoostModel())
    
    if LIGHTGBM_AVAILABLE:
        models.append(LightGBMModel())
    
    results = []
    feature_importances = {}
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"🌳 Training: {model.name}")
        print('='*60)
        
        # Train
        if isinstance(model, XGBoostModel) and XGBOOST_AVAILABLE:
            model.fit(X_train, y_train, feature_names=feature_names,
                     eval_set=[(X_test, y_test)], early_stopping_rounds=50)
        else:
            model.fit(X_train, y_train, feature_names=feature_names)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        per_class_f1 = f1_score(y_test, y_pred, average=None)
        accuracy = (y_pred == y_test).mean()
        
        print(f"\n📊 Results:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Macro F1: {macro_f1:.3f}")
        print(f"   Per-class F1: {per_class_f1}")
        
        # Store results
        results.append({
            'Model': model.name,
            'Accuracy': f"{accuracy:.3f}",
            'Macro_F1': f"{macro_f1:.3f}",
            'Weighted_F1': f"{weighted_f1:.3f}",
            'Dropout_F1': f"{per_class_f1[0]:.3f}",
            'Enrolled_F1': f"{per_class_f1[1]:.3f}",
            'Graduate_F1': f"{per_class_f1[2]:.3f}"
        })
        
        # Store feature importance
        feature_importances[model.name] = model.get_feature_importance()
        
        print(f"\n🔍 Top 10 Important Features:")
        print(model.get_feature_importance().head(10).to_string(index=False))
    
    return pd.DataFrame(results), feature_importances


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    """Example usage of tree-based models."""
    
    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.30, 0.20, 0.50])
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    print("🌳 TREE-BASED MODEL COMPARISON")
    print("=" * 60)
    
    results_df, importances = compare_tree_models(
        X_train, y_train, X_test, y_test, feature_names
    )
    
    print("\n" + "=" * 60)
    print("📊 FINAL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))
