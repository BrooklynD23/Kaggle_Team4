"""
Student Success Prediction - Ensemble Methods
=============================================

Ensemble methods combine multiple models to achieve better performance than 
any single model. Think of it as "wisdom of crowds" applied to machine learning.

THE ENSEMBLE PHILOSOPHY:
-----------------------
Imagine you're trying to predict stock prices. You could ask:
1. A technical analyst (looks at charts)
2. A fundamental analyst (looks at financials)  
3. An economist (looks at macro trends)

Each has blind spots, but together they cover more ground. That's ensembling!

Enhanced with Phase 4 improvements:
- Weighted voting based on validation performance
- Improved probability calibration
- Multiple meta-learner options for stacking
- Gradient Boosting meta-learner support

Author: ML Engineering Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional, Any
import warnings

from src.models.tree_models import is_gpu_available

# Import our tree models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

warnings.filterwarnings('ignore')


# =============================================================================
# ENSEMBLE METHOD 1: VOTING CLASSIFIER
# =============================================================================

class VotingEnsemble:
    """
    Voting Classifier - Democratic Decision Making
    
    THE CORE IDEA (An Analogy):
    ---------------------------
    Imagine a hiring committee with three members:
    - The CEO focuses on leadership potential
    - The CTO focuses on technical skills  
    - HR focuses on cultural fit
    
    Final decision = combined judgment of all three.
    
    TWO VOTING STRATEGIES:
    ---------------------
    
    1. HARD VOTING (Majority Rules):
       - Each model votes for a class
       - Class with most votes wins
       - Simple but ignores confidence levels
       
       Example:
       Model A: Graduate (100% confident)
       Model B: Dropout (51% confident)
       Model C: Dropout (51% confident)
       Result: Dropout wins (2 vs 1)
       
       Problem: Model A was very confident, B and C barely sure!
    
    2. SOFT VOTING (Weighted Probability):
       - Each model gives probability for each class
       - Average probabilities across models
       - Class with highest average probability wins
       
       Example:
       Model A: [0.0, 0.0, 1.0] (100% Graduate)
       Model B: [0.51, 0.0, 0.49] (51% Dropout)
       Model C: [0.51, 0.0, 0.49] (51% Dropout)
       Average: [0.34, 0.0, 0.66] → Graduate wins!
       
       Better: Respects confidence levels.
    
    WHY VOTING WORKS:
    ----------------
    1. Different models make different errors
    2. Errors are (hopefully) uncorrelated
    3. Majority vote filters out individual mistakes
    4. Like having multiple expert opinions
    
    WHEN TO USE:
    -----------
    ✅ When you have 3-7 diverse, strong base models
    ✅ When you want a simple, interpretable ensemble
    ✅ When base models have similar performance levels
    ✅ As a first ensemble attempt before trying stacking
    
    WHEN TO BE CAUTIOUS:
    -------------------
    ⚠️ With only 2 models (ties are problematic)
    ⚠️ When one model is much better than others (it should dominate)
    ⚠️ When models make correlated errors
    
    PHASE 4 ENHANCEMENTS:
    --------------------
    - Weighted voting based on validation performance
    - Improved probability calibration with isotonic regression option
    - Performance-based weight computation
    """
    
    def __init__(self, 
                 voting: str = 'soft',
                 use_calibration: bool = True,
                 use_weighted_voting: bool = True,
                 calibration_method: str = 'sigmoid',
                 random_state: int = 42):
        """
        Initialize Voting Ensemble.
        
        Parameters:
        -----------
        voting : str ('soft' or 'hard')
            Voting strategy. 'soft' is almost always better.
            
        use_calibration : bool
            Whether to calibrate probabilities. Important for soft voting!
            Calibration ensures that "70% confident" actually means 70% accuracy.
            
        use_weighted_voting : bool (Phase 4 enhancement)
            If True, compute weights based on cross-validation performance.
            Better-performing models get higher weights.
            
        calibration_method : str ('sigmoid' or 'isotonic')
            Method for probability calibration.
            - 'sigmoid': Platt scaling, works well with small datasets
            - 'isotonic': Non-parametric, needs more data but more flexible
        """
        self.voting = voting
        self.use_calibration = use_calibration
        self.use_weighted_voting = use_weighted_voting
        self.calibration_method = calibration_method
        self.random_state = random_state
        self.name = f"Voting Ensemble ({voting})"
        self.model = None
        self.base_models = None
        self.model_weights = None
        
    def _create_base_models(self) -> List[Tuple[str, Any]]:
        """
        Create a diverse set of base models.
        
        DIVERSITY IS KEY:
        -----------------
        We want models that:
        1. Are individually strong (Macro F1 > 0.65)
        2. Make DIFFERENT types of errors
        3. Use DIFFERENT learning algorithms
        
        Good combination:
        - Logistic Regression (linear, probabilistic)
        - Random Forest (tree-based, bagging)
        - XGBoost (tree-based, boosting)
        - KNN (distance-based, non-parametric)
        
        Bad combination:
        - Random Forest + ExtraTrees + Bagging (all tree-bagging, too similar!)
        """
        base_models = []
        
        # 1. Logistic Regression - The Linear Baseline
        # Scaled features are critical for LR
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_state
            ))
        ])
        base_models.append(('logistic_regression', lr_pipeline))
        
        # 2. Random Forest - The Bagging Tree Model
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        base_models.append(('random_forest', rf))
        
        # 3. XGBoost - The Boosting Tree Model
        if XGBOOST_AVAILABLE:
            xgb_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': -1,
                'eval_metric': 'mlogloss',
                'verbosity': 0
            }
            
            # XGBoost 2.0+ GPU configuration
            if is_gpu_available():
                xgb_params['tree_method'] = 'hist'
                xgb_params['device'] = 'cuda'
                
            xgb = XGBClassifier(**xgb_params)
            base_models.append(('xgboost', xgb))
        
        # 4. KNN - The Distance-Based Model
        # Needs scaling!
        knn_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(
                n_neighbors=15,  # sqrt(4425) ≈ 66, but we use fewer for speed
                weights='distance',
                n_jobs=-1
            ))
        ])
        base_models.append(('knn', knn_pipeline))
        
        return base_models
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'VotingEnsemble':
        """
        Train the voting ensemble.
        
        What happens:
        1. Each base model trains on the full training data
        2. Models are stored for later voting
        3. If calibration enabled, probabilities are adjusted
        4. If weighted voting enabled, compute weights from CV performance (Phase 4)
        """
        self.base_models = self._create_base_models()
        
        # Phase 4: Compute performance-based weights using cross-validation
        if self.use_weighted_voting and self.voting == 'soft':
            print("📊 Computing performance-based weights (CV)...")
            weights = []
            for name, model in self.base_models:
                try:
                    cv_scores = cross_val_score(model, X, y, cv=3, scoring='f1_macro', n_jobs=-1)
                    weight = cv_scores.mean()
                    weights.append(weight)
                    print(f"   {name}: CV F1 = {weight:.3f}")
                except Exception as e:
                    weights.append(0.5)  # Default weight if CV fails
                    print(f"   {name}: CV failed, using default weight")
            
            # Normalize weights to sum to len(weights) for interpretability
            weights = np.array(weights)
            weights = weights / weights.sum() * len(weights)
            self.model_weights = weights.tolist()
            print(f"   Computed weights: {[f'{w:.2f}' for w in self.model_weights]}")
        else:
            self.model_weights = None
        
        # Apply calibration if requested (important for soft voting!)
        if self.use_calibration and self.voting == 'soft':
            calibrated_models = []
            for name, model in self.base_models:
                # Platt scaling or isotonic calibration
                calibrated = CalibratedClassifierCV(
                    model, 
                    method=self.calibration_method, 
                    cv=3
                )
                calibrated_models.append((name, calibrated))
            self.base_models = calibrated_models
        
        self.model = VotingClassifier(
            estimators=self.base_models,
            voting=self.voting,
            weights=self.model_weights,
            n_jobs=-1
        )
        
        print(f"\n🗳️ Training Voting Ensemble with {len(self.base_models)} models:")
        for i, (name, _) in enumerate(self.base_models):
            weight_str = f" (weight={self.model_weights[i]:.2f})" if self.model_weights else ""
            print(f"   - {name}{weight_str}")
        
        self.model.fit(X, y)
        
        # Evaluate individual models
        print("\n📊 Individual Model Performance (on training data):")
        for name, estimator in self.model.named_estimators_.items():
            y_pred = estimator.predict(X)
            f1 = f1_score(y, y_pred, average='macro')
            print(f"   {name}: Macro F1 = {f1:.3f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using majority vote (hard) or averaged probabilities (soft)."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get averaged class probabilities (soft voting only)."""
        if self.voting == 'hard':
            raise ValueError("predict_proba not available for hard voting")
        return self.model.predict_proba(X)
    
    def get_model_contributions(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Analyze how much each base model contributes to ensemble performance.
        
        This helps identify:
        - Which models are pulling their weight
        - Which models might be redundant
        - Where to focus improvement efforts
        """
        contributions = []
        
        for name, estimator in self.model.named_estimators_.items():
            y_pred = estimator.predict(X)
            f1 = f1_score(y, y_pred, average='macro')
            accuracy = (y_pred == y).mean()
            
            contributions.append({
                'Model': name,
                'Macro_F1': f1,
                'Accuracy': accuracy
            })
        
        # Add ensemble performance
        y_pred_ensemble = self.model.predict(X)
        ensemble_f1 = f1_score(y, y_pred_ensemble, average='macro')
        ensemble_acc = (y_pred_ensemble == y).mean()
        
        contributions.append({
            'Model': 'ENSEMBLE',
            'Macro_F1': ensemble_f1,
            'Accuracy': ensemble_acc
        })
        
        return pd.DataFrame(contributions)


# =============================================================================
# ENSEMBLE METHOD 2: STACKING CLASSIFIER
# =============================================================================

class StackingEnsemble:
    """
    Stacking Classifier - Learning How to Combine Models
    
    THE CORE IDEA (An Analogy):
    ---------------------------
    Imagine you have three weather forecasters:
    - Forecaster A: Great at predicting sunny days, poor at rain
    - Forecaster B: Great at predicting rain, poor at snow
    - Forecaster C: Great at predicting snow, poor at sunny days
    
    A "meta-forecaster" learns WHEN to trust each one:
    - "When A says sunny AND B disagrees, trust A"
    - "When B says rain AND C says maybe, trust B"
    
    This is stacking: a meta-model learns the optimal combination!
    
    HOW IT WORKS:
    ------------
    Layer 0 (Base Models):
        RF → predictions_rf
        XGB → predictions_xgb  
        KNN → predictions_knn
        
    These predictions become FEATURES for the meta-model!
    
    Layer 1 (Meta-Model):
        Input: [predictions_rf, predictions_xgb, predictions_knn]
        Output: Final prediction
        
    The meta-model learns patterns like:
    - "When RF is confident but XGB disagrees, trust RF"
    - "When all three agree, that's probably correct"
    - "When KNN disagrees with RF+XGB, ignore KNN"
    
    KEY INSIGHT:
    -----------
    The meta-model doesn't just average—it learns WHEN to trust each model.
    
    PREVENTING LEAKAGE:
    ------------------
    CRITICAL: Base model predictions must come from cross-validation!
    
    Wrong approach (leaky):
    1. Train RF on all data
    2. Get RF predictions on same data
    3. Train meta-model on these predictions
    Problem: Meta-model sees RF's "cheating" predictions
    
    Correct approach:
    1. For each fold, train RF on other folds
    2. Get RF predictions on held-out fold
    3. Combine to get out-of-fold predictions for all data
    4. Train meta-model on these honest predictions
    
    sklearn's StackingClassifier does this automatically!
    
    WHY STACKING CAN OUTPERFORM VOTING:
    ----------------------------------
    1. Learns non-uniform weighting (some models get more weight)
    2. Can learn conditional trust (trust model A for class 1, model B for class 2)
    3. Captures model correlation patterns
    
    WHEN TO USE:
    -----------
    ✅ When you need maximum performance
    ✅ When base models have complementary strengths
    ✅ When you have enough data (need to split for CV)
    ✅ When computational cost is acceptable
    
    WHEN TO BE CAUTIOUS:
    -------------------
    ⚠️ With very small datasets (CV splits become tiny)
    ⚠️ When training time is critical
    ⚠️ When interpretability is paramount
    """
    
    def __init__(self,
                 meta_model: str = 'logistic',
                 cv: int = 5,
                 passthrough: bool = False,
                 random_state: int = 42):
        """
        Initialize Stacking Ensemble.
        
        Parameters:
        -----------
        meta_model : str ('logistic', 'ridge', 'rf', 'gb')
            Type of meta-learner.
            
            'logistic' (recommended): Simple, interpretable, less prone to overfit
            'ridge': L2-regularized linear model
            'rf': Can capture non-linear combinations (but may overfit)
            'gb': GradientBoosting - powerful but may overfit (Phase 4 addition)
            
        cv : int
            Cross-validation folds for generating base model predictions.
            More folds = more reliable predictions, but slower training.
            
        passthrough : bool
            Whether to include original features alongside base model predictions.
            - True: Meta-model sees original features + predictions
            - False: Meta-model only sees predictions
            
            Usually False is better (prevents meta-model from "bypassing" base models)
        """
        self.meta_model_type = meta_model
        self.cv = cv
        self.passthrough = passthrough
        self.random_state = random_state
        self.name = f"Stacking Ensemble (meta={meta_model})"
        self.model = None
        
    def _create_base_models(self) -> List[Tuple[str, Any]]:
        """
        Create diverse base models for stacking.
        
        Same principles as voting: diversity is key!
        """
        base_models = []
        
        # Logistic Regression (linear)
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_state
            ))
        ])
        base_models.append(('lr', lr_pipeline))
        
        # Random Forest (bagging)
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        base_models.append(('rf', rf))
        
        # XGBoost (boosting)
        if XGBOOST_AVAILABLE:
            xgb_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': self.random_state,
                'n_jobs': -1,
                'eval_metric': 'mlogloss',
                'verbosity': 0
            }
            
            # XGBoost 2.0+ GPU configuration
            if is_gpu_available():
                xgb_params['tree_method'] = 'hist'
                xgb_params['device'] = 'cuda'
                
            xgb = XGBClassifier(**xgb_params)
            base_models.append(('xgb', xgb))
        
        # KNN (distance-based)
        knn_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, weights='distance'))
        ])
        base_models.append(('knn', knn_pipeline))
        
        return base_models
    
    def _create_meta_model(self) -> Any:
        """
        Create the meta-learner.
        
        WHY LOGISTIC REGRESSION FOR META-MODEL?
        --------------------------------------
        1. Simple enough to not overfit to base model predictions
        2. Coefficients show how much each base model is trusted
        3. Fast to train
        4. Works well with probability inputs
        
        If LR meta-model coefficients are:
        [RF: 0.4, XGB: 0.5, KNN: 0.1, LR_base: 0.0]
        
        This tells us: Trust XGB most, RF second, KNN a little, ignore LR_base
        
        PHASE 4 ADDITIONS:
        -----------------
        - GradientBoosting meta-learner for capturing non-linear combinations
        """
        if self.meta_model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_state
            )
        elif self.meta_model_type == 'ridge':
            return RidgeClassifier(random_state=self.random_state)
        elif self.meta_model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                class_weight='balanced',
                random_state=self.random_state
            )
        elif self.meta_model_type == 'gb':
            # Phase 4: GradientBoosting meta-learner
            return GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown meta_model: {self.meta_model_type}. Choose from: logistic, ridge, rf, gb")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """
        Train the stacking ensemble.
        
        Training process:
        1. Create base models and meta-model
        2. For each CV fold:
           a. Train base models on training folds
           b. Generate predictions on held-out fold
        3. Combine all held-out predictions
        4. Train meta-model on these cross-validated predictions
        """
        base_models = self._create_base_models()
        meta_model = self._create_meta_model()
        
        self.model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=self.cv,
            passthrough=self.passthrough,
            n_jobs=-1
        )
        
        print(f"🏗️ Training Stacking Ensemble")
        print(f"   Base models: {[name for name, _ in base_models]}")
        print(f"   Meta-model: {self.meta_model_type}")
        print(f"   CV folds: {self.cv}")
        
        self.model.fit(X, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the stacked ensemble."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities."""
        return self.model.predict_proba(X)
    
    def get_meta_model_weights(self) -> Optional[pd.DataFrame]:
        """
        Extract weights from the meta-model (if interpretable).
        
        Only works when meta_model is 'logistic'.
        Shows how much each base model is trusted.
        """
        if self.meta_model_type != 'logistic':
            print("⚠️ Weights only available for logistic meta-model")
            return None
        
        meta = self.model.final_estimator_
        base_names = [name for name, _ in self.model.estimators]
        
        # For multi-class, we have coefficients for each class
        n_classes = meta.coef_.shape[0]
        n_base_models = len(base_names)
        
        # Each base model contributes n_classes probabilities
        weights_data = []
        for class_idx in range(n_classes):
            row = {'Class': class_idx}
            for model_idx, name in enumerate(base_names):
                # Get the coefficient for this base model's contribution to this class
                start_idx = model_idx * n_classes
                end_idx = start_idx + n_classes
                model_weights = meta.coef_[class_idx, start_idx:end_idx]
                row[name] = np.mean(np.abs(model_weights))
            weights_data.append(row)
        
        return pd.DataFrame(weights_data)


# =============================================================================
# CUSTOM CASCADING ENSEMBLE (ADVANCED)
# =============================================================================

class CascadingEnsemble:
    """
    Cascading Ensemble - Progressive Refinement
    
    THE CORE IDEA (An Analogy):
    ---------------------------
    Imagine a medical diagnosis system:
    
    Stage 1 (Fast Filter): 
        Simple model quickly identifies "easy" cases
        - Clear graduates → immediately classified
        - Obvious dropouts → immediately classified
        - Uncertain cases → pass to Stage 2
        
    Stage 2 (Specialist):
        Complex model handles "hard" cases
        - Only runs on uncertain cases
        - More accurate but slower
    
    BENEFITS:
    --------
    1. Computational efficiency: Most cases are "easy"
    2. Targeted complexity: Complex models only where needed
    3. Interpretable: Easy to explain "confidence thresholds"
    
    HOW IT WORKS:
    ------------
    1. Stage 1 model predicts with probabilities
    2. If max probability > threshold → accept prediction
    3. If max probability ≤ threshold → pass to Stage 2
    4. Stage 2 makes final decision
    
    WHEN TO USE:
    -----------
    ✅ When inference speed matters (production systems)
    ✅ When most cases are "easy" to classify
    ✅ When you have a clear confidence threshold
    ✅ When you want interpretable decision paths
    """
    
    def __init__(self,
                 confidence_threshold: float = 0.7,
                 random_state: int = 42):
        """
        Initialize Cascading Ensemble.
        
        Parameters:
        -----------
        confidence_threshold : float
            If Stage 1's max probability exceeds this, accept its prediction.
            - Higher threshold: More cases go to Stage 2 (more accurate, slower)
            - Lower threshold: Fewer cases go to Stage 2 (faster, might miss hard cases)
            
            Typical range: 0.6 - 0.85
        """
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state
        self.name = f"Cascading Ensemble (threshold={confidence_threshold})"
        self.stage1_model = None
        self.stage2_model = None
        
    def _create_stage1_model(self) -> Any:
        """
        Create Stage 1 model - should be FAST and reasonably accurate.
        
        Good choices:
        - Logistic Regression (fastest)
        - Small Random Forest (good balance)
        - LightGBM (fast and accurate)
        """
        return Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_state
            ))
        ])
    
    def _create_stage2_model(self) -> Any:
        """
        Create Stage 2 model - should be ACCURATE (speed less important).
        
        Good choices:
        - XGBoost with more trees
        - Stacking ensemble
        - Well-tuned Random Forest
        """
        if XGBOOST_AVAILABLE:
            xgb_params = {
                'n_estimators': 300,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': -1,
                'eval_metric': 'mlogloss',
                'verbosity': 0
            }
            
            # XGBoost 2.0+ GPU configuration
            if is_gpu_available():
                xgb_params['tree_method'] = 'hist'
                xgb_params['device'] = 'cuda'
                
            return XGBClassifier(**xgb_params)
        else:
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CascadingEnsemble':
        """Train both stages of the cascade."""
        print(f"🔄 Training Cascading Ensemble")
        print(f"   Confidence threshold: {self.confidence_threshold}")
        
        # Train Stage 1
        print("   Training Stage 1 (fast model)...")
        self.stage1_model = self._create_stage1_model()
        self.stage1_model.fit(X, y)
        
        # Train Stage 2 (could optionally train only on "hard" examples)
        print("   Training Stage 2 (accurate model)...")
        self.stage2_model = self._create_stage2_model()
        self.stage2_model.fit(X, y)
        
        # Analyze expected cascade behavior
        stage1_probs = self.stage1_model.predict_proba(X)
        max_probs = stage1_probs.max(axis=1)
        pct_to_stage2 = (max_probs <= self.confidence_threshold).mean() * 100
        
        print(f"   Expected cascade behavior:")
        print(f"   - {100 - pct_to_stage2:.1f}% handled by Stage 1 (fast)")
        print(f"   - {pct_to_stage2:.1f}% sent to Stage 2 (accurate)")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the cascade.
        
        1. Get Stage 1 predictions and confidence
        2. For high-confidence predictions, use Stage 1
        3. For low-confidence predictions, use Stage 2
        """
        # Stage 1 predictions
        stage1_probs = self.stage1_model.predict_proba(X)
        stage1_preds = stage1_probs.argmax(axis=1)
        max_probs = stage1_probs.max(axis=1)
        
        # Determine which samples need Stage 2
        needs_stage2 = max_probs <= self.confidence_threshold
        
        # Initialize final predictions with Stage 1
        final_preds = stage1_preds.copy()
        
        # For uncertain samples, use Stage 2
        if needs_stage2.any():
            X_uncertain = X[needs_stage2]
            stage2_preds = self.stage2_model.predict(X_uncertain)
            final_preds[needs_stage2] = stage2_preds
        
        return final_preds
    
    def predict_with_info(self, X: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Predict with additional information about the cascade.
        
        Returns predictions AND a DataFrame showing which stage handled each sample.
        Useful for analysis and debugging.
        """
        stage1_probs = self.stage1_model.predict_proba(X)
        stage1_preds = stage1_probs.argmax(axis=1)
        max_probs = stage1_probs.max(axis=1)
        needs_stage2 = max_probs <= self.confidence_threshold
        
        final_preds = stage1_preds.copy()
        stages_used = np.ones(len(X), dtype=int)  # Default: Stage 1
        
        if needs_stage2.any():
            X_uncertain = X[needs_stage2]
            stage2_preds = self.stage2_model.predict(X_uncertain)
            final_preds[needs_stage2] = stage2_preds
            stages_used[needs_stage2] = 2
        
        info_df = pd.DataFrame({
            'Stage_Used': stages_used,
            'Stage1_Confidence': max_probs,
            'Stage1_Prediction': stage1_preds,
            'Final_Prediction': final_preds
        })
        
        return final_preds, info_df


# =============================================================================
# ENSEMBLE COMPARISON UTILITY
# =============================================================================

def compare_ensembles(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Train and compare all ensemble methods.
    
    This helps identify which ensemble strategy works best for your data.
    """
    ensembles = [
        VotingEnsemble(voting='soft'),
        VotingEnsemble(voting='hard'),
        StackingEnsemble(meta_model='logistic'),
        CascadingEnsemble(confidence_threshold=0.7)
    ]
    
    results = []
    
    for ensemble in ensembles:
        print(f"\n{'='*60}")
        print(f"🔧 Training: {ensemble.name}")
        print('='*60)
        
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        per_class_f1 = f1_score(y_test, y_pred, average=None)
        accuracy = (y_pred == y_test).mean()
        
        print(f"\n📊 Results:")
        print(f"   Macro F1: {macro_f1:.3f}")
        print(f"   Accuracy: {accuracy:.3f}")
        
        results.append({
            'Ensemble': ensemble.name,
            'Accuracy': f"{accuracy:.3f}",
            'Macro_F1': f"{macro_f1:.3f}",
            'Weighted_F1': f"{weighted_f1:.3f}",
            'Dropout_F1': f"{per_class_f1[0]:.3f}",
            'Enrolled_F1': f"{per_class_f1[1]:.3f}",
            'Graduate_F1': f"{per_class_f1[2]:.3f}"
        })
    
    return pd.DataFrame(results)


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    """Example usage of ensemble methods."""
    
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.30, 0.20, 0.50])
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("🎯 ENSEMBLE METHODS COMPARISON")
    print("=" * 60)
    
    results_df = compare_ensembles(X_train, y_train, X_test, y_test)
    
    print("\n" + "=" * 60)
    print("📊 FINAL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("💡 ENSEMBLE INSIGHTS")
    print("=" * 60)
    print("""
    WHEN TO USE EACH ENSEMBLE:
    
    1. Voting (Soft): Default choice. Simple, effective, interpretable.
       Best when base models have similar performance.
       
    2. Voting (Hard): When you don't need probabilities.
       Slightly simpler but loses confidence information.
       
    3. Stacking: When you need maximum performance.
       Meta-model learns optimal combination.
       More complex, higher risk of overfitting.
       
    4. Cascading: When inference speed matters.
       Routes easy cases to fast model.
       Great for production systems.
    
    GENERAL ADVICE:
    - Start with Soft Voting
    - Try Stacking if you need more performance
    - Use Cascading for production deployment
    """)
