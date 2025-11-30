"""
Student Success Prediction - Model Interpretation
=================================================

In education, we can't just predict - we must EXPLAIN. A teacher won't 
trust a black box that says "this student will dropout" without knowing 
WHY the model thinks that.

Model interpretation serves three purposes:
1. BUILD TRUST: Stakeholders understand and trust the predictions
2. DEBUG MODELS: Find if the model learned spurious correlations
3. GENERATE INSIGHTS: Discover what actually drives student success

Author: ML Engineering Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

warnings.filterwarnings('ignore')

# Check for optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Install with: pip install shap")


# =============================================================================
# INTERPRETATION CONCEPTS EXPLAINED
# =============================================================================
"""
WHY INTERPRETATION MATTERS - THE LOAN ANALOGY:
---------------------------------------------
Imagine a bank uses ML to approve/deny loans. 

Scenario A (Black Box):
Bank: "Your loan is denied."
Customer: "Why?"
Bank: "The model said so."
Customer: "That's not fair! I want to know why!"

Scenario B (Interpretable):
Bank: "Your loan is denied."
Customer: "Why?"
Bank: "Your debt-to-income ratio is 45% (threshold is 35%), and 
      you had 2 late payments in the last year."
Customer: "I understand. I'll work on reducing my debt."

For student success, interpretation helps advisors INTERVENE effectively:
"This student is at risk because grades dropped 15% and they withdrew 
from 2 courses. Let's discuss their situation."

INTERPRETATION METHODS:
----------------------

1. FEATURE IMPORTANCE (Global)
   What features matter most OVERALL across all predictions?
   
   - Permutation Importance: Shuffle a feature, see how much accuracy drops
   - Tree-based: Count how often feature is used for splits
   - Coefficient-based: For linear models, coefficient magnitude
   
   Limitation: Shows average importance, not per-prediction.

2. SHAP VALUES (Local + Global)
   How much does each feature contribute to THIS SPECIFIC prediction?
   
   Based on game theory: If features are "players" and prediction is 
   the "payout", SHAP fairly distributes credit among players.
   
   Properties:
   - Local accuracy: SHAP values sum to (prediction - baseline)
   - Consistency: If a feature matters more, it gets higher SHAP
   - Missingness: Missing features contribute 0
   
   Best for: Understanding individual predictions AND global patterns.

3. PARTIAL DEPENDENCE PLOTS (PDP)
   How does the prediction change as ONE feature varies?
   
   Analogy: "All else being equal, how does age affect prediction?"
   
   Limitation: Assumes features are independent (often not true).

4. INDIVIDUAL CONDITIONAL EXPECTATION (ICE)
   Like PDP, but shows the curve for each individual sample.
   
   Useful for: Detecting if a feature affects different samples differently.
"""


# =============================================================================
# PERMUTATION FEATURE IMPORTANCE
# =============================================================================

class PermutationImportance:
    """
    Permutation Feature Importance - Model-Agnostic Interpretation
    
    THE INTUITION:
    -------------
    If a feature is important, shuffling its values should hurt predictions.
    If shuffling doesn't matter, the feature isn't helping.
    
    Analogy: If you're predicting exam scores and "hours studied" is 
    important, randomly assigning hours studied to different students 
    should make predictions much worse.
    
    ADVANTAGES:
    ----------
    - Works with ANY model (model-agnostic)
    - Easy to understand and explain
    - Accounts for feature interactions (unlike coefficient importance)
    
    DISADVANTAGES:
    -------------
    - Can be slow (requires multiple model evaluations)
    - Correlated features can share importance artificially
    """
    
    def __init__(self, 
                 model: Any,
                 scoring: str = 'f1_macro',
                 n_repeats: int = 10,
                 random_state: int = 42):
        """
        Initialize permutation importance calculator.
        
        Parameters:
        -----------
        model : fitted sklearn-compatible model
            Must have a predict method
            
        scoring : str
            Metric to evaluate ('f1_macro', 'accuracy', etc.)
            
        n_repeats : int
            Number of times to shuffle each feature.
            More repeats = more stable estimates, but slower.
        """
        self.model = model
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.importances_ = None
        
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            feature_names: List[str] = None) -> 'PermutationImportance':
        """
        Calculate permutation importance for all features.
        
        Process:
        1. Get baseline score on unshuffled data
        2. For each feature:
           a. Shuffle that feature's values
           b. Get new score
           c. Importance = baseline - shuffled score
        3. Repeat n_repeats times, average results
        """
        from sklearn.inspection import permutation_importance as sklearn_perm
        
        # Use sklearn's implementation
        result = sklearn_perm(
            self.model,
            X, y,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            scoring=self.scoring,
            n_jobs=-1
        )
        
        # Store results
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        self.importances_ = pd.DataFrame({
            'Feature': feature_names,
            'Importance_Mean': result.importances_mean,
            'Importance_Std': result.importances_std
        }).sort_values('Importance_Mean', ascending=False)
        
        return self
    
    def get_importance(self, top_n: int = None) -> pd.DataFrame:
        """Get feature importance ranking."""
        if self.importances_ is None:
            raise ValueError("Must call fit() first")
        
        df = self.importances_.copy()
        if top_n:
            df = df.head(top_n)
        return df.reset_index(drop=True)
    
    def explain(self) -> str:
        """Generate human-readable explanation of top features."""
        if self.importances_ is None:
            raise ValueError("Must call fit() first")
        
        top_5 = self.importances_.head(5)
        
        explanation = "📊 Feature Importance Analysis\n"
        explanation += "=" * 40 + "\n\n"
        explanation += "The most important features for prediction are:\n\n"
        
        for i, row in top_5.iterrows():
            importance_pct = row['Importance_Mean'] * 100
            explanation += f"{row.name + 1}. {row['Feature']}\n"
            explanation += f"   Impact: Shuffling this feature reduces {self.scoring} by {importance_pct:.2f}%\n\n"
        
        return explanation


# =============================================================================
# SHAP EXPLAINER
# =============================================================================

if SHAP_AVAILABLE:
    class SHAPExplainer:
        """
        SHAP (SHapley Additive exPlanations) - Game Theory for ML
        
        THE GAME THEORY INTUITION:
        -------------------------
        Imagine a team project where you need to fairly divide credit.
        
        - Player 1 alone: Grade = C
        - Player 2 alone: Grade = B
        - Player 1 + 2 together: Grade = A
        
        How much credit does each player deserve?
        Shapley values solve this by considering ALL possible coalitions.
        
        For ML:
        - "Players" = features
        - "Payout" = prediction
        - SHAP values = fair credit to each feature
        
        KEY PROPERTIES:
        --------------
        1. Additivity: sum(SHAP values) = prediction - expected_value
        2. Consistency: More important features get higher SHAP
        3. Local: Explains each prediction individually
        4. Global: Aggregate for overall feature importance
        
        INTERPRETING SHAP VALUES:
        ------------------------
        - Positive SHAP: Feature pushes prediction UP
        - Negative SHAP: Feature pushes prediction DOWN
        - Magnitude: How much it pushes
        
        Example for Student Success:
        - "Grade improvement" SHAP = +0.15 
          → "Improving grades increases graduation probability by 15%"
        - "Financial risk" SHAP = -0.20
          → "Financial risk decreases graduation probability by 20%"
        """
        
        def __init__(self, 
                     model: Any,
                     model_type: str = 'tree',
                     background_samples: int = 100):
            """
            Initialize SHAP explainer.
            
            Parameters:
            -----------
            model : fitted model
                The model to explain
                
            model_type : str
                'tree' for tree-based models (fast, exact)
                'kernel' for any model (slow, approximate)
                'linear' for linear models (fast, exact)
                
            background_samples : int
                Number of background samples for kernel SHAP.
                More = slower but more accurate.
            """
            self.model = model
            self.model_type = model_type
            self.background_samples = background_samples
            self.explainer = None
            self.shap_values = None
            self.expected_value = None
            
        def fit(self, 
                X: np.ndarray,
                feature_names: List[str] = None) -> 'SHAPExplainer':
            """
            Create SHAP explainer and compute SHAP values.
            
            Parameters:
            -----------
            X : array
                Data to explain (typically validation or test set)
                
            feature_names : list
                Names of features for readable output
            """
            self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
            
            # Create appropriate explainer
            if self.model_type == 'tree':
                # Fast, exact for tree-based models
                self.explainer = shap.TreeExplainer(self.model)
                
            elif self.model_type == 'linear':
                # Fast, exact for linear models
                self.explainer = shap.LinearExplainer(self.model, X)
                
            else:  # 'kernel'
                # Slow, approximate, but works for any model
                background = shap.sample(X, self.background_samples)
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') 
                    else self.model.predict,
                    background
                )
            
            # Compute SHAP values
            self.shap_values = self.explainer.shap_values(X)
            self.expected_value = self.explainer.expected_value
            self.X = X
            
            return self
        
        def explain_prediction(self, 
                              idx: int,
                              class_idx: int = 2) -> Dict[str, Any]:
            """
            Explain a single prediction.
            
            Parameters:
            -----------
            idx : int
                Index of the sample to explain
                
            class_idx : int
                Which class to explain (default 2 = Graduate)
                
            Returns:
            --------
            Dictionary with feature contributions
            """
            if self.shap_values is None:
                raise ValueError("Must call fit() first")
            
            # Handle different SHAP output formats
            if isinstance(self.shap_values, list):
                # Multi-class: list of arrays per class
                shap_for_class = self.shap_values[class_idx][idx]
            else:
                shap_for_class = self.shap_values[idx]
            
            # Create explanation DataFrame
            explanation = pd.DataFrame({
                'Feature': self.feature_names,
                'Value': self.X[idx],
                'SHAP': shap_for_class
            })
            
            # Sort by absolute SHAP value
            explanation['Abs_SHAP'] = np.abs(explanation['SHAP'])
            explanation = explanation.sort_values('Abs_SHAP', ascending=False)
            
            # Get expected value for this class
            if isinstance(self.expected_value, (list, np.ndarray)):
                base_value = self.expected_value[class_idx]
            else:
                base_value = self.expected_value
            
            return {
                'sample_idx': idx,
                'class_explained': class_idx,
                'base_value': base_value,
                'prediction_contribution': shap_for_class.sum(),
                'feature_contributions': explanation.drop('Abs_SHAP', axis=1)
            }
        
        def get_global_importance(self, class_idx: int = 2) -> pd.DataFrame:
            """
            Get global feature importance from SHAP values.
            
            Global importance = mean(|SHAP|) across all samples.
            """
            if self.shap_values is None:
                raise ValueError("Must call fit() first")
            
            if isinstance(self.shap_values, list):
                shap_matrix = self.shap_values[class_idx]
            else:
                shap_matrix = self.shap_values
            
            # Mean absolute SHAP value per feature
            mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
            
            importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Mean_Abs_SHAP': mean_abs_shap
            }).sort_values('Mean_Abs_SHAP', ascending=False)
            
            return importance.reset_index(drop=True)
        
        def generate_narrative(self, 
                              idx: int,
                              class_names: List[str] = None,
                              class_idx: int = 2) -> str:
            """
            Generate human-readable explanation for a prediction.
            
            This is what you'd show to an academic advisor!
            """
            explanation = self.explain_prediction(idx, class_idx)
            
            if class_names is None:
                class_names = ['Dropout', 'Enrolled', 'Graduate']
            
            narrative = f"📋 PREDICTION EXPLANATION - Sample {idx}\n"
            narrative += "=" * 50 + "\n\n"
            
            # Overall prediction
            class_name = class_names[class_idx]
            contribution = explanation['prediction_contribution']
            direction = "increases" if contribution > 0 else "decreases"
            
            narrative += f"The model predicts this student's probability of '{class_name}' "
            narrative += f"{direction} by {abs(contribution):.1%} from the baseline.\n\n"
            
            # Top contributing factors
            narrative += "🔍 Key Factors:\n\n"
            
            contributions = explanation['feature_contributions'].head(5)
            
            for _, row in contributions.iterrows():
                feature = row['Feature']
                value = row['Value']
                shap_val = row['SHAP']
                
                if shap_val > 0:
                    effect = f"increases probability by {shap_val:.1%}"
                else:
                    effect = f"decreases probability by {abs(shap_val):.1%}"
                
                narrative += f"  • {feature} = {value:.2f}\n"
                narrative += f"    Effect: {effect}\n\n"
            
            return narrative


# =============================================================================
# SIMPLE IMPORTANCE (No External Dependencies)
# =============================================================================

class SimpleFeatureImportance:
    """
    Simple feature importance methods that don't require external libraries.
    
    Useful when SHAP isn't available or for quick analysis.
    """
    
    @staticmethod
    def from_tree_model(model: Any, 
                        feature_names: List[str]) -> pd.DataFrame:
        """
        Extract feature importance from tree-based models.
        
        Works with: RandomForest, GradientBoosting, XGBoost, LightGBM
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            importance = model.model.feature_importances_
        else:
            raise ValueError("Model doesn't have feature_importances_ attribute")
        
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        df['Importance_Pct'] = df['Importance'] / df['Importance'].sum() * 100
        
        return df.reset_index(drop=True)
    
    @staticmethod
    def from_linear_model(model: Any,
                          feature_names: List[str],
                          class_idx: int = 2) -> pd.DataFrame:
        """
        Extract feature importance from linear models.
        
        For logistic regression, larger absolute coefficients = more important.
        """
        if hasattr(model, 'coef_'):
            coef = model.coef_
        elif hasattr(model, 'named_steps'):
            # Pipeline - find the classifier
            for step in model.named_steps.values():
                if hasattr(step, 'coef_'):
                    coef = step.coef_
                    break
        else:
            raise ValueError("Model doesn't have coef_ attribute")
        
        # For multi-class, select the specified class
        if coef.ndim > 1:
            coef = coef[class_idx]
        
        df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coef,
            'Abs_Coefficient': np.abs(coef)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        return df.reset_index(drop=True)


# =============================================================================
# INTERPRETATION REPORT GENERATOR
# =============================================================================

class InterpretationReport:
    """
    Generate comprehensive interpretation reports.
    
    Combines multiple interpretation methods into a single report
    suitable for stakeholders.
    """
    
    def __init__(self, 
                 model: Any,
                 X: np.ndarray,
                 y: np.ndarray,
                 feature_names: List[str],
                 class_names: List[str] = None):
        """
        Initialize report generator.
        
        Parameters:
        -----------
        model : fitted model
        X : features
        y : labels
        feature_names : list of feature names
        class_names : list of class names
        """
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.class_names = class_names or ['Dropout', 'Enrolled', 'Graduate']
        
    def generate_report(self, 
                        sample_explanations: int = 5) -> Dict[str, Any]:
        """
        Generate comprehensive interpretation report.
        
        Includes:
        1. Global feature importance
        2. Sample-level explanations
        3. Key insights
        """
        report = {
            'global_importance': None,
            'sample_explanations': [],
            'insights': []
        }
        
        # Global importance
        try:
            importance = SimpleFeatureImportance.from_tree_model(
                self.model, self.feature_names
            )
            report['global_importance'] = importance
        except:
            try:
                importance = SimpleFeatureImportance.from_linear_model(
                    self.model, self.feature_names
                )
                report['global_importance'] = importance
            except:
                report['insights'].append(
                    "Could not extract feature importance from model type."
                )
        
        # SHAP explanations if available
        if SHAP_AVAILABLE:
            try:
                explainer = SHAPExplainer(self.model, model_type='tree')
                explainer.fit(self.X, self.feature_names)
                
                # Explain a few samples
                for i in range(min(sample_explanations, len(self.X))):
                    narrative = explainer.generate_narrative(i, self.class_names)
                    report['sample_explanations'].append(narrative)
                    
            except Exception as e:
                report['insights'].append(f"SHAP analysis failed: {str(e)}")
        
        # Generate insights from importance
        if report['global_importance'] is not None:
            top_features = report['global_importance'].head(5)['Feature'].tolist()
            report['insights'].append(
                f"Top 5 predictive features: {', '.join(top_features)}"
            )
        
        return report
    
    def print_report(self, report: Dict = None) -> None:
        """Print formatted interpretation report."""
        if report is None:
            report = self.generate_report()
        
        print("\n" + "=" * 70)
        print("📊 MODEL INTERPRETATION REPORT")
        print("=" * 70)
        
        # Global importance
        if report['global_importance'] is not None:
            print("\n🌍 GLOBAL FEATURE IMPORTANCE")
            print("-" * 40)
            print(report['global_importance'].head(10).to_string(index=False))
        
        # Sample explanations
        if report['sample_explanations']:
            print("\n📋 SAMPLE EXPLANATIONS")
            print("-" * 40)
            for explanation in report['sample_explanations'][:3]:
                print(explanation)
                print("-" * 40)
        
        # Key insights
        if report['insights']:
            print("\n💡 KEY INSIGHTS")
            print("-" * 40)
            for insight in report['insights']:
                print(f"  • {insight}")
        
        print("\n" + "=" * 70)


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    """Demonstrate model interpretation capabilities."""
    
    print("📊 MODEL INTERPRETATION DEMONSTRATION")
    print("=" * 60)
    
    # Create synthetic model and data
    from sklearn.ensemble import RandomForestClassifier
    
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.2, 0.5])
    
    feature_names = [
        'grade_improvement', 'approval_rate', 'financial_risk',
        'course_load', 'attendance', 'age', 'scholarship',
        'parent_education', 'economic_stress', 'engagement_score'
    ]
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Permutation importance
    print("\n🔄 PERMUTATION IMPORTANCE")
    print("-" * 40)
    
    perm_imp = PermutationImportance(model, n_repeats=5)
    perm_imp.fit(X, y, feature_names)
    print(perm_imp.get_importance(top_n=5))
    print(perm_imp.explain())
    
    # Simple importance
    print("\n🌳 TREE-BASED IMPORTANCE")
    print("-" * 40)
    simple_imp = SimpleFeatureImportance.from_tree_model(model, feature_names)
    print(simple_imp.head(5))
    
    # SHAP (if available)
    if SHAP_AVAILABLE:
        print("\n🎯 SHAP ANALYSIS")
        print("-" * 40)
        
        shap_exp = SHAPExplainer(model, model_type='tree')
        shap_exp.fit(X, feature_names)
        
        print("\nGlobal SHAP Importance:")
        print(shap_exp.get_global_importance().head(5))
        
        print("\nSample Explanation:")
        print(shap_exp.generate_narrative(0))
    
    print("\n" + "=" * 60)
    print("💡 INTERPRETATION BEST PRACTICES")
    print("=" * 60)
    print("""
    1. USE MULTIPLE METHODS: Different methods have different strengths.
       Permutation importance is reliable; SHAP provides local explanations.
    
    2. VALIDATE IMPORTANCE: If a surprising feature is important, investigate!
       It might be a data leakage issue or spurious correlation.
    
    3. COMMUNICATE CLEARLY: Stakeholders need simple explanations.
       "Grade improvement is the #1 predictor" beats technical jargon.
    
    4. ACTIONABLE INSIGHTS: Good interpretation leads to action.
       "Students with declining grades AND financial stress need immediate support."
    
    5. BEWARE OF CORRELATION: Feature importance doesn't mean causation.
       "Older students graduate more" doesn't mean "age causes graduation."
    """)
