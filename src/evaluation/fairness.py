"""
Student Success Prediction - Fairness Monitoring
================================================

Fairness in ML is not optional, especially in education. A model that 
systematically disadvantages certain groups can perpetuate inequity 
and cause real harm to students' futures.

THE ETHICAL IMPERATIVE:
----------------------
Our model predicts who might dropout. This prediction could trigger:
- Early intervention and support (GOOD)
- Denial of opportunities or resources (BAD)
- Self-fulfilling prophecies (VERY BAD)

We MUST ensure predictions don't unfairly target protected groups.

Author: ML Engineering Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# FAIRNESS CONCEPTS EXPLAINED
# =============================================================================
"""
FAIRNESS METRICS - AN INTUITIVE GUIDE:

Imagine two groups of students: Group A and Group B.
We want our model to treat them fairly. But what does "fair" mean?

1. DEMOGRAPHIC PARITY (Statistical Parity)
   -----------------------------------------
   Definition: Positive prediction rates should be equal across groups.
   
   Formula: P(Ŷ=1 | Group=A) = P(Ŷ=1 | Group=B)
   
   Analogy: If 30% of Group A is predicted to graduate, 30% of Group B 
   should also be predicted to graduate.
   
   When it matters: When the action based on prediction (like scholarship 
   allocation) should be distributed equally regardless of group.
   
   Limitation: Ignores actual outcomes. If Group A genuinely has 50% 
   graduation rate and Group B has 30%, forcing equal predictions 
   would be inaccurate.

2. EQUAL OPPORTUNITY (True Positive Rate Parity)
   ---------------------------------------------
   Definition: Among actual positives, prediction rates should be equal.
   
   Formula: P(Ŷ=1 | Y=1, Group=A) = P(Ŷ=1 | Y=1, Group=B)
   
   Analogy: Among students who WILL graduate, we should correctly 
   identify the same proportion from each group.
   
   When it matters: When false negatives are costly. Missing a 
   potential graduate is a missed opportunity for support.

3. EQUALIZED ODDS (True Positive + False Positive Rate Parity)
   ----------------------------------------------------------
   Definition: Both TPR and FPR should be equal across groups.
   
   Formula: 
   P(Ŷ=1 | Y=1, Group=A) = P(Ŷ=1 | Y=1, Group=B)  AND
   P(Ŷ=1 | Y=0, Group=A) = P(Ŷ=1 | Y=0, Group=B)
   
   Analogy: We correctly identify graduates at equal rates AND we 
   falsely flag non-graduates at equal rates across groups.
   
   When it matters: When both false positives and false negatives matter.

4. PREDICTIVE PARITY (Precision Parity)
   ------------------------------------
   Definition: Among predicted positives, accuracy should be equal.
   
   Formula: P(Y=1 | Ŷ=1, Group=A) = P(Y=1 | Ŷ=1, Group=B)
   
   Analogy: When we predict "will graduate," we should be equally 
   accurate for both groups.
   
   When it matters: When resources are allocated based on predictions 
   and we want equal "hit rate" across groups.

THE IMPOSSIBILITY THEOREM:
--------------------------
It's mathematically proven that you CANNOT satisfy all fairness metrics 
simultaneously (except in trivial cases). You must choose which metrics 
matter most for your application.

FOR STUDENT SUCCESS PREDICTION:
------------------------------
We prioritize EQUAL OPPORTUNITY because:
1. Missing at-risk students (false negatives) is costly
2. We want equal chance of catching students who need help
3. Interventions can be supportive rather than punitive
"""


# =============================================================================
# FAIRNESS CALCULATOR
# =============================================================================

class FairnessCalculator:
    """
    Calculate fairness metrics across demographic groups.
    
    USAGE:
    -----
    calc = FairnessCalculator(sensitive_feature='Gender')
    report = calc.calculate_fairness(y_true, y_pred, sensitive_values)
    """
    
    def __init__(self, 
                 sensitive_feature: str,
                 positive_class: int = 2,  # Graduate
                 threshold: float = 0.1):
        """
        Initialize fairness calculator.
        
        Parameters:
        -----------
        sensitive_feature : str
            Name of the demographic attribute (for reporting)
            
        positive_class : int
            Which class is considered "positive" for fairness metrics.
            For student success: Graduate (2) is typically positive.
            
        threshold : float
            Maximum acceptable difference between groups.
            Industry standard is often 0.1 (10 percentage points).
        """
        self.sensitive_feature = sensitive_feature
        self.positive_class = positive_class
        self.threshold = threshold
        
    def _get_group_metrics(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for a single group."""
        # Binary conversion for fairness metrics
        y_true_binary = (y_true == self.positive_class).astype(int)
        y_pred_binary = (y_pred == self.positive_class).astype(int)
        
        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(
            y_true_binary, y_pred_binary, labels=[0, 1]
        ).ravel()
        
        # Calculate rates (with protection against division by zero)
        metrics = {
            'positive_rate': y_pred_binary.mean(),  # P(Ŷ=1)
            'tpr': tp / max(tp + fn, 1),           # P(Ŷ=1|Y=1)
            'fpr': fp / max(fp + tn, 1),           # P(Ŷ=1|Y=0)
            'precision': tp / max(tp + fp, 1),      # P(Y=1|Ŷ=1)
            'base_rate': y_true_binary.mean(),      # P(Y=1)
            'n_samples': len(y_true)
        }
        
        return metrics
    
    def calculate_fairness(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          sensitive_values: np.ndarray) -> Dict[str, Any]:
        """
        Calculate fairness metrics across all groups.
        
        Parameters:
        -----------
        y_true : array
            True class labels
            
        y_pred : array  
            Predicted class labels
            
        sensitive_values : array
            Group membership for each sample
            
        Returns:
        --------
        Dictionary with fairness analysis results
        """
        groups = np.unique(sensitive_values)
        
        # Calculate metrics per group
        group_metrics = {}
        for group in groups:
            mask = sensitive_values == group
            group_metrics[group] = self._get_group_metrics(
                y_true[mask], y_pred[mask]
            )
        
        # Calculate pairwise disparities
        disparities = self._calculate_disparities(group_metrics)
        
        # Determine if model is "fair" by our threshold
        is_fair = all(
            abs(d) <= self.threshold 
            for d in disparities.values() 
            if isinstance(d, (int, float))
        )
        
        return {
            'sensitive_feature': self.sensitive_feature,
            'groups': list(groups),
            'group_metrics': group_metrics,
            'disparities': disparities,
            'is_fair': is_fair,
            'threshold': self.threshold,
            'positive_class': self.positive_class
        }
    
    def _calculate_disparities(self, 
                               group_metrics: Dict) -> Dict[str, float]:
        """
        Calculate disparity metrics between groups.
        
        For binary groups, this is the difference.
        For multiple groups, this is max - min.
        """
        groups = list(group_metrics.keys())
        
        if len(groups) < 2:
            return {'error': 'Need at least 2 groups for comparison'}
        
        # Extract metric values across groups
        positive_rates = [m['positive_rate'] for m in group_metrics.values()]
        tprs = [m['tpr'] for m in group_metrics.values()]
        fprs = [m['fpr'] for m in group_metrics.values()]
        precisions = [m['precision'] for m in group_metrics.values()]
        
        disparities = {
            # Demographic Parity Difference
            'demographic_parity_diff': max(positive_rates) - min(positive_rates),
            
            # Equal Opportunity Difference (TPR parity)
            'equal_opportunity_diff': max(tprs) - min(tprs),
            
            # Equalized Odds (max of TPR and FPR differences)
            'equalized_odds_diff': max(
                max(tprs) - min(tprs),
                max(fprs) - min(fprs)
            ),
            
            # Predictive Parity Difference
            'predictive_parity_diff': max(precisions) - min(precisions)
        }
        
        return disparities


# =============================================================================
# FAIRNESS AUDITOR
# =============================================================================

class FairnessAuditor:
    """
    Comprehensive fairness audit for ML models.
    
    This class runs fairness checks across multiple sensitive attributes
    and generates actionable reports.
    
    USAGE:
    -----
    auditor = FairnessAuditor()
    report = auditor.audit(model, X_test, y_test, sensitive_features_df)
    auditor.print_report(report)
    """
    
    def __init__(self, 
                 positive_class: int = 2,
                 threshold: float = 0.1):
        """
        Initialize the fairness auditor.
        
        Parameters:
        -----------
        positive_class : int
            Class considered "positive" (Graduate = 2 for our problem)
            
        threshold : float
            Maximum acceptable disparity (default 10%)
        """
        self.positive_class = positive_class
        self.threshold = threshold
        
    def audit(self,
              y_true: np.ndarray,
              y_pred: np.ndarray,
              sensitive_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Run fairness audit across all sensitive features.
        
        Parameters:
        -----------
        y_true : array
            True class labels
            
        y_pred : array
            Predicted class labels
            
        sensitive_features : DataFrame
            DataFrame with columns for each sensitive attribute
            (e.g., Gender, Age_Group, Scholarship_Status)
            
        Returns:
        --------
        Complete audit report
        """
        audit_results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'n_samples': len(y_true),
            'positive_class': self.positive_class,
            'threshold': self.threshold,
            'features_audited': list(sensitive_features.columns),
            'feature_reports': {},
            'overall_fair': True,
            'recommendations': []
        }
        
        # Audit each sensitive feature
        for feature in sensitive_features.columns:
            calculator = FairnessCalculator(
                sensitive_feature=feature,
                positive_class=self.positive_class,
                threshold=self.threshold
            )
            
            report = calculator.calculate_fairness(
                y_true, y_pred, sensitive_features[feature].values
            )
            
            audit_results['feature_reports'][feature] = report
            
            if not report['is_fair']:
                audit_results['overall_fair'] = False
                audit_results['recommendations'].append(
                    self._generate_recommendation(feature, report)
                )
        
        return audit_results
    
    def _generate_recommendation(self, 
                                 feature: str, 
                                 report: Dict) -> str:
        """Generate actionable recommendation for unfair feature."""
        disparities = report['disparities']
        
        # Find the worst disparity
        worst_metric = max(disparities.items(), key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0)
        
        recommendations = []
        
        if worst_metric[0] == 'demographic_parity_diff':
            recommendations.append(
                f"📊 {feature}: Prediction rates differ by {worst_metric[1]:.1%} between groups. "
                f"Consider: (1) Threshold adjustment per group, (2) Reweighting training data, "
                f"(3) Adding {feature}-specific features to capture legitimate differences."
            )
            
        elif worst_metric[0] == 'equal_opportunity_diff':
            recommendations.append(
                f"🎯 {feature}: True positive rates differ by {worst_metric[1]:.1%}. "
                f"The model catches actual graduates better for some groups. "
                f"Consider: (1) Collect more data for underperforming groups, "
                f"(2) Feature engineering for group-specific patterns."
            )
            
        elif worst_metric[0] == 'equalized_odds_diff':
            recommendations.append(
                f"⚖️ {feature}: Error rates differ by {worst_metric[1]:.1%}. "
                f"Consider: (1) Post-processing calibration, "
                f"(2) In-processing fairness constraints during training."
            )
        
        return '\n'.join(recommendations)
    
    def print_report(self, audit_results: Dict) -> None:
        """Print a formatted fairness report."""
        print("\n" + "=" * 70)
        print("🔍 FAIRNESS AUDIT REPORT")
        print("=" * 70)
        
        print(f"\n📋 Audit Summary:")
        print(f"   Samples evaluated: {audit_results['n_samples']}")
        print(f"   Features audited: {audit_results['features_audited']}")
        print(f"   Fairness threshold: {audit_results['threshold']:.0%}")
        
        overall_status = "✅ PASS" if audit_results['overall_fair'] else "⚠️ ISSUES DETECTED"
        print(f"   Overall status: {overall_status}")
        
        # Detailed results per feature
        for feature, report in audit_results['feature_reports'].items():
            print(f"\n{'─' * 70}")
            print(f"📊 {feature.upper()}")
            print('─' * 70)
            
            status = "✅ Fair" if report['is_fair'] else "⚠️ Unfair"
            print(f"   Status: {status}")
            
            # Group-level metrics
            print(f"\n   Group Metrics:")
            for group, metrics in report['group_metrics'].items():
                print(f"   └─ {group} (n={metrics['n_samples']}):")
                print(f"      Positive prediction rate: {metrics['positive_rate']:.1%}")
                print(f"      True positive rate: {metrics['tpr']:.1%}")
                print(f"      False positive rate: {metrics['fpr']:.1%}")
                print(f"      Base rate (actual): {metrics['base_rate']:.1%}")
            
            # Disparities
            print(f"\n   Disparities:")
            for metric, value in report['disparities'].items():
                if isinstance(value, (int, float)):
                    flag = "⚠️" if abs(value) > self.threshold else "✓"
                    print(f"   {flag} {metric}: {value:.1%}")
        
        # Recommendations
        if audit_results['recommendations']:
            print(f"\n{'=' * 70}")
            print("💡 RECOMMENDATIONS")
            print('=' * 70)
            for rec in audit_results['recommendations']:
                print(f"\n{rec}")
        
        print("\n" + "=" * 70)


# =============================================================================
# FAIRNESS-AWARE PREDICTION
# =============================================================================

class FairnessAwarePredictor:
    """
    Wrapper that applies fairness constraints to predictions.
    
    POST-PROCESSING APPROACH:
    ------------------------
    Rather than modifying the model itself, we adjust the decision 
    threshold per group to achieve fairness.
    
    Trade-off: May slightly reduce overall accuracy in exchange for fairness.
    
    USAGE:
    -----
    fair_predictor = FairnessAwarePredictor(base_model, 'Gender')
    fair_predictor.fit(X_val, y_val, gender_values)
    fair_predictions = fair_predictor.predict(X_test, test_gender_values)
    """
    
    def __init__(self,
                 base_model: Any,
                 sensitive_feature: str,
                 positive_class: int = 2,
                 fairness_constraint: str = 'demographic_parity'):
        """
        Initialize fairness-aware predictor.
        
        Parameters:
        -----------
        base_model : sklearn-compatible model
            The trained base model (must have predict_proba)
            
        sensitive_feature : str
            Name of the sensitive attribute
            
        positive_class : int
            Which class is "positive"
            
        fairness_constraint : str
            Which fairness metric to optimize:
            - 'demographic_parity': Equal prediction rates
            - 'equal_opportunity': Equal true positive rates
        """
        self.base_model = base_model
        self.sensitive_feature = sensitive_feature
        self.positive_class = positive_class
        self.fairness_constraint = fairness_constraint
        self.group_thresholds = {}
        
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            sensitive_values: np.ndarray,
            target_rate: float = None) -> 'FairnessAwarePredictor':
        """
        Learn group-specific thresholds to achieve fairness.
        
        Parameters:
        -----------
        X : array
            Validation features
            
        y : array
            Validation labels
            
        sensitive_values : array
            Group membership for validation set
            
        target_rate : float
            Target positive prediction rate (for demographic parity).
            If None, uses the overall rate from the base model.
        """
        # Get base model probabilities
        if hasattr(self.base_model, 'predict_proba'):
            probs = self.base_model.predict_proba(X)[:, self.positive_class]
        else:
            raise ValueError("Base model must support predict_proba")
        
        # Determine target rate
        if target_rate is None:
            target_rate = (probs > 0.5).mean()
        
        # Find threshold for each group
        groups = np.unique(sensitive_values)
        
        for group in groups:
            mask = sensitive_values == group
            group_probs = probs[mask]
            group_y = y[mask]
            
            # Find threshold that achieves target rate for this group
            if self.fairness_constraint == 'demographic_parity':
                # Find threshold such that prediction rate = target_rate
                threshold = np.percentile(group_probs, (1 - target_rate) * 100)
            
            elif self.fairness_constraint == 'equal_opportunity':
                # Find threshold to maximize TPR while constraining disparity
                # This is more complex - using simple approximation
                positive_mask = group_y == self.positive_class
                if positive_mask.sum() > 0:
                    positive_probs = group_probs[positive_mask]
                    threshold = np.percentile(positive_probs, 25)  # Catch 75% of positives
                else:
                    threshold = 0.5
            
            self.group_thresholds[group] = threshold
        
        print(f"📊 Learned group thresholds for {self.sensitive_feature}:")
        for group, thresh in self.group_thresholds.items():
            print(f"   {group}: {thresh:.3f}")
        
        return self
    
    def predict(self,
                X: np.ndarray,
                sensitive_values: np.ndarray) -> np.ndarray:
        """
        Make fairness-aware predictions.
        
        Uses group-specific thresholds to ensure fairness.
        """
        probs = self.base_model.predict_proba(X)
        predictions = np.zeros(len(X), dtype=int)
        
        for group, threshold in self.group_thresholds.items():
            mask = sensitive_values == group
            group_probs = probs[mask, self.positive_class]
            
            # Apply group-specific threshold
            predictions[mask] = np.where(
                group_probs > threshold,
                self.positive_class,
                probs[mask].argmax(axis=1)  # Use argmax for non-positive
            )
        
        return predictions


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    """Demonstrate fairness monitoring capabilities."""
    
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate predictions with built-in bias
    # Group A: Higher graduation rates
    # Group B: Lower graduation rates (unfair!)
    
    groups = np.random.choice(['Group_A', 'Group_B'], size=n_samples, p=[0.6, 0.4])
    
    # True labels (same distribution for both groups)
    y_true = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.2, 0.5])
    
    # Biased predictions (model favors Group A)
    y_pred = y_true.copy()
    
    # Introduce bias: Group B gets fewer "Graduate" predictions
    group_b_mask = groups == 'Group_B'
    flip_mask = (y_pred == 2) & group_b_mask & (np.random.random(n_samples) < 0.3)
    y_pred[flip_mask] = 1  # Demote some graduates to enrolled
    
    print("🔍 FAIRNESS MONITORING DEMONSTRATION")
    print("=" * 60)
    
    # Create sensitive features DataFrame
    sensitive_df = pd.DataFrame({
        'Demographic_Group': groups,
        'Age_Category': np.random.choice(['Young', 'Mature'], size=n_samples)
    })
    
    # Run audit
    auditor = FairnessAuditor(positive_class=2, threshold=0.1)
    report = auditor.audit(y_true, y_pred, sensitive_df)
    auditor.print_report(report)
    
    print("\n" + "=" * 60)
    print("💡 KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    1. ALWAYS audit models for fairness before deployment
    
    2. Different fairness metrics may conflict - choose based on:
       - What action will be taken on predictions?
       - What are the costs of different error types?
       - What does fairness mean in your context?
    
    3. If unfairness is detected:
       - First, investigate root cause (data? model? feature?)
       - Consider fairness-aware training or post-processing
       - Document trade-offs between accuracy and fairness
    
    4. For student success prediction:
       - Equal Opportunity is often most appropriate
       - We want to catch at-risk students equally across groups
       - Interventions should be supportive, not punitive
    """)
