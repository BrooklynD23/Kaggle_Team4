"""
Student Success Prediction - Feature Engineering
================================================

Feature engineering is the art of transforming raw data into features that 
better represent the underlying problem. Think of it as translating raw 
ingredients into a recipe that's easier for models to understand.

THE ANALOGY:
-----------
Raw data is like raw ingredients (flour, eggs, sugar).
Engineered features are like prepared ingredients (batter, dough, glaze).

A chef (model) can work with raw ingredients, but prepared ingredients 
make it MUCH easier to create a great dish (prediction).

Author: ML Engineering Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# FEATURE ENGINEERING RATIONALE
# =============================================================================
"""
WHY FEATURE ENGINEERING MATTERS:
-------------------------------
Raw features often don't capture the TRUE signal for prediction.

Example: We have semester 1 and semester 2 grades.
- Raw: sem1_grade=70, sem2_grade=80
- Engineered: grade_improvement = +10 (upward trend!)

The improvement (+10) is MORE predictive than either raw grade because it 
captures TRAJECTORY, which is what we care about for student success.

FEATURE ENGINEERING CATEGORIES FOR STUDENT SUCCESS:
--------------------------------------------------
1. Academic Performance: Grade trends, approval rates, efficiency
2. Engagement Signals: Course load consistency, withdrawal patterns
3. Financial Risk: Debt indicators, economic stress
4. Demographic Context: Family education, support systems
5. Interaction Terms: Compound risk factors
"""


# =============================================================================
# FEATURE CATEGORY 1: ACADEMIC PERFORMANCE
# =============================================================================

class AcademicFeatures(BaseEstimator, TransformerMixin):
    """
    Create features that capture academic performance patterns.
    
    RAW FEATURES WE HAVE (typical for this dataset):
    -----------------------------------------------
    - Curricular units 1st sem (enrolled)
    - Curricular units 1st sem (approved)
    - Curricular units 1st sem (grade)
    - Curricular units 2nd sem (enrolled)
    - Curricular units 2nd sem (approved)
    - Curricular units 2nd sem (grade)
    - Curricular units 1st sem (evaluations)
    - Curricular units 2nd sem (evaluations)
    
    WHY THESE FEATURES MATTER:
    -------------------------
    Raw grades tell us WHERE a student is.
    Engineered features tell us WHERE THEY'RE GOING.
    
    A student with grade=60 but improving by 10 points per semester
    is likely to succeed. A student with grade=75 but declining by 
    10 points per semester is at risk.
    """
    
    def __init__(self, 
                 grade_col_sem1: str = 'Curricular units 1st sem (grade)',
                 grade_col_sem2: str = 'Curricular units 2nd sem (grade)',
                 enrolled_col_sem1: str = 'Curricular units 1st sem (enrolled)',
                 enrolled_col_sem2: str = 'Curricular units 2nd sem (enrolled)',
                 approved_col_sem1: str = 'Curricular units 1st sem (approved)',
                 approved_col_sem2: str = 'Curricular units 2nd sem (approved)',
                 eval_col_sem1: str = 'Curricular units 1st sem (evaluations)',
                 eval_col_sem2: str = 'Curricular units 2nd sem (evaluations)'):
        """Initialize with column names from your dataset."""
        self.grade_col_sem1 = grade_col_sem1
        self.grade_col_sem2 = grade_col_sem2
        self.enrolled_col_sem1 = enrolled_col_sem1
        self.enrolled_col_sem2 = enrolled_col_sem2
        self.approved_col_sem1 = approved_col_sem1
        self.approved_col_sem2 = approved_col_sem2
        self.eval_col_sem1 = eval_col_sem1
        self.eval_col_sem2 = eval_col_sem2
        
    def fit(self, X: pd.DataFrame, y=None):
        """Nothing to learn - all transformations are deterministic."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create academic performance features.
        
        Each feature has a specific rationale explained below.
        """
        X = X.copy()
        
        # Safe column access (handle missing columns gracefully)
        def safe_get(df, col, default=0):
            return df[col] if col in df.columns else default
        
        # ------------------------------------------------------------------
        # FEATURE: Grade Improvement
        # ------------------------------------------------------------------
        # WHY: Trajectory matters more than absolute grade.
        # A student going from 60→70 is on an upward path.
        # A student going from 80→70 might be struggling.
        # ------------------------------------------------------------------
        if self.grade_col_sem1 in X.columns and self.grade_col_sem2 in X.columns:
            X['grade_improvement'] = (
                X[self.grade_col_sem2] - X[self.grade_col_sem1]
            )
            
            # Categorize trajectory
            X['grade_trajectory'] = pd.cut(
                X['grade_improvement'],
                bins=[-np.inf, -5, 0, 5, np.inf],
                labels=['declining', 'slight_decline', 'slight_improve', 'improving']
            ).astype(str)
        
        # ------------------------------------------------------------------
        # FEATURE: Approval Rate (per semester)
        # ------------------------------------------------------------------
        # WHY: Are students passing what they attempt?
        # Low approval rate = student is struggling
        # approval_rate = approved / enrolled
        # ------------------------------------------------------------------
        if self.approved_col_sem1 in X.columns and self.enrolled_col_sem1 in X.columns:
            X['approval_rate_sem1'] = (
                X[self.approved_col_sem1] / X[self.enrolled_col_sem1].replace(0, 1)
            )
            
        if self.approved_col_sem2 in X.columns and self.enrolled_col_sem2 in X.columns:
            X['approval_rate_sem2'] = (
                X[self.approved_col_sem2] / X[self.enrolled_col_sem2].replace(0, 1)
            )
        
        # Overall approval rate
        if 'approval_rate_sem1' in X.columns and 'approval_rate_sem2' in X.columns:
            X['approval_rate_overall'] = (
                X['approval_rate_sem1'] + X['approval_rate_sem2']
            ) / 2
            
            # Approval trend
            X['approval_rate_trend'] = (
                X['approval_rate_sem2'] - X['approval_rate_sem1']
            )
        
        # ------------------------------------------------------------------
        # FEATURE: Total Units (Approved)
        # ------------------------------------------------------------------
        # WHY: Cumulative progress toward graduation
        # ------------------------------------------------------------------
        if self.approved_col_sem1 in X.columns and self.approved_col_sem2 in X.columns:
            X['total_approved_units'] = (
                X[self.approved_col_sem1] + X[self.approved_col_sem2]
            )
        
        # ------------------------------------------------------------------
        # FEATURE: Evaluation Efficiency
        # ------------------------------------------------------------------
        # WHY: Are students getting evaluated on what they enroll in?
        # Low efficiency might indicate disengagement or withdrawal
        # ------------------------------------------------------------------
        if self.eval_col_sem1 in X.columns and self.enrolled_col_sem1 in X.columns:
            X['eval_efficiency_sem1'] = (
                X[self.eval_col_sem1] / X[self.enrolled_col_sem1].replace(0, 1)
            )
            
        if self.eval_col_sem2 in X.columns and self.enrolled_col_sem2 in X.columns:
            X['eval_efficiency_sem2'] = (
                X[self.eval_col_sem2] / X[self.enrolled_col_sem2].replace(0, 1)
            )
        
        # ------------------------------------------------------------------
        # FEATURE: Grade-to-Effort Ratio
        # ------------------------------------------------------------------
        # WHY: Are high grades coming from lighter course loads?
        # This normalizes performance by effort.
        # ------------------------------------------------------------------
        if self.grade_col_sem1 in X.columns and self.enrolled_col_sem1 in X.columns:
            X['grade_per_unit_sem1'] = (
                X[self.grade_col_sem1] / X[self.enrolled_col_sem1].replace(0, 1)
            )
            
        if self.grade_col_sem2 in X.columns and self.enrolled_col_sem2 in X.columns:
            X['grade_per_unit_sem2'] = (
                X[self.grade_col_sem2] / X[self.enrolled_col_sem2].replace(0, 1)
            )
        
        return X
    
    def get_feature_names(self) -> List[str]:
        """Return names of created features."""
        return [
            'grade_improvement', 'grade_trajectory',
            'approval_rate_sem1', 'approval_rate_sem2', 'approval_rate_overall',
            'approval_rate_trend', 'total_approved_units',
            'eval_efficiency_sem1', 'eval_efficiency_sem2',
            'grade_per_unit_sem1', 'grade_per_unit_sem2'
        ]


# =============================================================================
# FEATURE CATEGORY 2: ENGAGEMENT SIGNALS
# =============================================================================

class EngagementFeatures(BaseEstimator, TransformerMixin):
    """
    Create features that capture student engagement patterns.
    
    THE INSIGHT:
    -----------
    Disengagement often precedes dropout. Before students leave, they often:
    - Reduce course load
    - Stop attending evaluations  
    - Show inconsistent patterns
    
    These features detect early warning signs.
    """
    
    def __init__(self,
                 enrolled_col_sem1: str = 'Curricular units 1st sem (enrolled)',
                 enrolled_col_sem2: str = 'Curricular units 2nd sem (enrolled)',
                 eval_col_sem1: str = 'Curricular units 1st sem (evaluations)',
                 eval_col_sem2: str = 'Curricular units 2nd sem (evaluations)',
                 approved_col_sem1: str = 'Curricular units 1st sem (approved)',
                 approved_col_sem2: str = 'Curricular units 2nd sem (approved)'):
        self.enrolled_col_sem1 = enrolled_col_sem1
        self.enrolled_col_sem2 = enrolled_col_sem2
        self.eval_col_sem1 = eval_col_sem1
        self.eval_col_sem2 = eval_col_sem2
        self.approved_col_sem1 = approved_col_sem1
        self.approved_col_sem2 = approved_col_sem2
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # ------------------------------------------------------------------
        # FEATURE: Course Load Consistency
        # ------------------------------------------------------------------
        # WHY: Large changes in course load signal instability
        # Student going from 6 courses to 2 courses = red flag
        # ------------------------------------------------------------------
        if self.enrolled_col_sem1 in X.columns and self.enrolled_col_sem2 in X.columns:
            X['course_load_change'] = (
                X[self.enrolled_col_sem2] - X[self.enrolled_col_sem1]
            )
            X['course_load_change_pct'] = (
                X['course_load_change'] / X[self.enrolled_col_sem1].replace(0, 1)
            )
            
            # Absolute consistency (large swings in either direction are concerning)
            X['course_load_volatility'] = np.abs(X['course_load_change'])
        
        # ------------------------------------------------------------------
        # FEATURE: Non-Evaluation Rate (Withdrawal Proxy)
        # ------------------------------------------------------------------
        # WHY: Units without evaluations suggest withdrawal or giving up
        # ------------------------------------------------------------------
        if self.eval_col_sem1 in X.columns and self.enrolled_col_sem1 in X.columns:
            # Units where student didn't even try to be evaluated
            X['units_without_eval_sem1'] = (
                X[self.enrolled_col_sem1] - X[self.eval_col_sem1]
            ).clip(lower=0)  # Can't have negative
            
        if self.eval_col_sem2 in X.columns and self.enrolled_col_sem2 in X.columns:
            X['units_without_eval_sem2'] = (
                X[self.enrolled_col_sem2] - X[self.eval_col_sem2]
            ).clip(lower=0)
        
        # Total withdrawal signal
        if 'units_without_eval_sem1' in X.columns and 'units_without_eval_sem2' in X.columns:
            X['total_units_without_eval'] = (
                X['units_without_eval_sem1'] + X['units_without_eval_sem2']
            )
        
        # ------------------------------------------------------------------
        # FEATURE: Zero-Enrolled Semester Flag
        # ------------------------------------------------------------------
        # WHY: A semester with zero enrollments is a strong dropout signal
        # ------------------------------------------------------------------
        if self.enrolled_col_sem1 in X.columns:
            X['zero_enrolled_sem1'] = (X[self.enrolled_col_sem1] == 0).astype(int)
        if self.enrolled_col_sem2 in X.columns:
            X['zero_enrolled_sem2'] = (X[self.enrolled_col_sem2] == 0).astype(int)
        
        # ------------------------------------------------------------------
        # FEATURE: Dropout Risk Score (Composite)
        # ------------------------------------------------------------------
        # WHY: Combine multiple signals into single risk indicator
        # ------------------------------------------------------------------
        risk_score = 0
        if 'course_load_change' in X.columns:
            risk_score += (X['course_load_change'] < -2).astype(int) * 1
        if 'total_units_without_eval' in X.columns:
            risk_score += (X['total_units_without_eval'] > 2).astype(int) * 2
        if 'zero_enrolled_sem2' in X.columns:
            risk_score += X['zero_enrolled_sem2'] * 3
        X['engagement_risk_score'] = risk_score
        
        return X


# =============================================================================
# FEATURE CATEGORY 3: FINANCIAL RISK
# =============================================================================

class FinancialFeatures(BaseEstimator, TransformerMixin):
    """
    Create features that capture financial stress indicators.
    
    THE INSIGHT:
    -----------
    Financial stress is a leading indicator of dropout.
    Students who can't pay tuition or have debt are at higher risk.
    
    We combine individual financial flags and economic indicators
    into composite risk measures.
    """
    
    def __init__(self,
                 debtor_col: str = 'Debtor',
                 tuition_col: str = 'Tuition fees up to date',
                 scholarship_col: str = 'Scholarship holder',
                 unemployment_col: str = 'Unemployment rate',
                 inflation_col: str = 'Inflation rate',
                 gdp_col: str = 'GDP'):
        self.debtor_col = debtor_col
        self.tuition_col = tuition_col
        self.scholarship_col = scholarship_col
        self.unemployment_col = unemployment_col
        self.inflation_col = inflation_col
        self.gdp_col = gdp_col
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # ------------------------------------------------------------------
        # FEATURE: Financial Risk Flag
        # ------------------------------------------------------------------
        # WHY: Either being a debtor OR having unpaid tuition indicates risk
        # ------------------------------------------------------------------
        if self.debtor_col in X.columns and self.tuition_col in X.columns:
            X['financial_risk'] = (
                (X[self.debtor_col] == 1) | (X[self.tuition_col] == 0)
            ).astype(int)
        
        # ------------------------------------------------------------------
        # FEATURE: Scholarship Safety Net
        # ------------------------------------------------------------------
        # WHY: Scholarships reduce financial stress
        # ------------------------------------------------------------------
        if self.scholarship_col in X.columns:
            X['has_financial_support'] = X[self.scholarship_col].astype(int)
        
        # ------------------------------------------------------------------
        # FEATURE: Economic Stress Index
        # ------------------------------------------------------------------
        # WHY: Macro-economic conditions affect student ability to continue
        # High unemployment + high inflation + low GDP = stressful economy
        # 
        # We normalize and combine these factors.
        # ------------------------------------------------------------------
        econ_features = []
        
        if self.unemployment_col in X.columns:
            # Higher unemployment = more stress
            X['unemployment_norm'] = (
                X[self.unemployment_col] - X[self.unemployment_col].mean()
            ) / X[self.unemployment_col].std()
            econ_features.append('unemployment_norm')
            
        if self.inflation_col in X.columns:
            # Higher inflation = more stress
            X['inflation_norm'] = (
                X[self.inflation_col] - X[self.inflation_col].mean()
            ) / X[self.inflation_col].std()
            econ_features.append('inflation_norm')
            
        if self.gdp_col in X.columns:
            # Higher GDP = less stress (so we negate)
            X['gdp_norm'] = -(
                X[self.gdp_col] - X[self.gdp_col].mean()
            ) / X[self.gdp_col].std()
            econ_features.append('gdp_norm')
        
        # Combine into single index
        if econ_features:
            X['economic_stress_index'] = X[econ_features].mean(axis=1)
        
        # ------------------------------------------------------------------
        # FEATURE: Combined Financial Vulnerability
        # ------------------------------------------------------------------
        # WHY: Personal financial risk + poor economy = double trouble
        # ------------------------------------------------------------------
        if 'financial_risk' in X.columns and 'economic_stress_index' in X.columns:
            X['financial_vulnerability'] = (
                X['financial_risk'] + (X['economic_stress_index'] > 0).astype(int)
            )
        
        return X


# =============================================================================
# FEATURE CATEGORY 4: DEMOGRAPHIC CONTEXT
# =============================================================================

class DemographicFeatures(BaseEstimator, TransformerMixin):
    """
    Create features that capture demographic context.
    
    IMPORTANT ETHICAL NOTE:
    ----------------------
    These features should INFORM interventions, not EXCLUDE students.
    The goal is to identify who needs MORE support, not to deny opportunities.
    
    We must monitor for bias in how these features affect predictions.
    """
    
    def __init__(self,
                 mother_qual_col: str = "Mother's qualification",
                 father_qual_col: str = "Father's qualification",
                 mother_occ_col: str = "Mother's occupation",
                 father_occ_col: str = "Father's occupation",
                 age_col: str = 'Age at enrollment'):
        self.mother_qual_col = mother_qual_col
        self.father_qual_col = father_qual_col
        self.mother_occ_col = mother_occ_col
        self.father_occ_col = father_occ_col
        self.age_col = age_col
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # ------------------------------------------------------------------
        # FEATURE: Family Education Level
        # ------------------------------------------------------------------
        # WHY: First-generation students may need more support
        # This is for INTERVENTION, not exclusion!
        # ------------------------------------------------------------------
        if self.mother_qual_col in X.columns and self.father_qual_col in X.columns:
            X['family_education_avg'] = (
                X[self.mother_qual_col] + X[self.father_qual_col]
            ) / 2
            
            # Flag potential first-generation students
            # (Assuming lower values = lower education in the encoding)
            X['low_family_education'] = (
                X['family_education_avg'] < X['family_education_avg'].median()
            ).astype(int)
        
        # ------------------------------------------------------------------
        # FEATURE: Age-Related Patterns
        # ------------------------------------------------------------------
        # WHY: Non-traditional students (older) have different challenges
        # ------------------------------------------------------------------
        if self.age_col in X.columns:
            X['is_mature_student'] = (X[self.age_col] > 25).astype(int)
            X['age_group'] = pd.cut(
                X[self.age_col],
                bins=[0, 20, 25, 30, 100],
                labels=['young', 'typical', 'mature', 'returning']
            ).astype(str)
        
        return X


# =============================================================================
# FEATURE CATEGORY 5: INTERACTION TERMS
# =============================================================================

class InteractionFeatures(BaseEstimator, TransformerMixin):
    """
    Create interaction features that capture compound risk factors.
    
    THE INSIGHT:
    -----------
    Risk factors often COMPOUND. Two moderate risks together can be 
    more dangerous than one severe risk alone.
    
    Example:
    - Financial stress alone: Manageable
    - Academic decline alone: Manageable
    - Financial stress + Academic decline: Critical!
    
    Interaction terms capture these compound effects.
    """
    
    def __init__(self):
        pass
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # ------------------------------------------------------------------
        # INTERACTION: Financial Risk × Academic Decline
        # ------------------------------------------------------------------
        # WHY: The combination is particularly dangerous
        # ------------------------------------------------------------------
        if 'financial_risk' in X.columns and 'grade_improvement' in X.columns:
            X['financial_academic_risk'] = (
                (X['financial_risk'] == 1) & (X['grade_improvement'] < 0)
            ).astype(int)
        
        # ------------------------------------------------------------------
        # INTERACTION: First-Gen × Financial Risk
        # ------------------------------------------------------------------
        # WHY: First-gen students with financial issues lack both 
        # financial and social capital safety nets
        # ------------------------------------------------------------------
        if 'low_family_education' in X.columns and 'financial_risk' in X.columns:
            X['first_gen_financial_risk'] = (
                (X['low_family_education'] == 1) & (X['financial_risk'] == 1)
            ).astype(int)
        
        # ------------------------------------------------------------------
        # INTERACTION: Engagement Drop × Low Grades
        # ------------------------------------------------------------------
        # WHY: Disengaging while already struggling = high risk
        # ------------------------------------------------------------------
        if 'course_load_change' in X.columns and 'approval_rate_overall' in X.columns:
            X['disengaging_while_struggling'] = (
                (X['course_load_change'] < -2) & 
                (X['approval_rate_overall'] < 0.5)
            ).astype(int)
        
        # ------------------------------------------------------------------
        # INTERACTION: Multiple Risk Score
        # ------------------------------------------------------------------
        # WHY: Count how many risk factors are present
        # ------------------------------------------------------------------
        risk_columns = [
            'financial_risk', 'low_family_education', 
            'zero_enrolled_sem2', 'engagement_risk_score'
        ]
        available_risks = [col for col in risk_columns if col in X.columns]
        
        if available_risks:
            X['total_risk_factors'] = X[available_risks].sum(axis=1)
            X['high_risk_student'] = (X['total_risk_factors'] >= 2).astype(int)
        
        return X


# =============================================================================
# COMPLETE FEATURE ENGINEERING PIPELINE
# =============================================================================

class StudentFeatureEngineer:
    """
    Complete feature engineering pipeline for student success prediction.
    
    USAGE:
    -----
    engineer = StudentFeatureEngineer()
    X_transformed = engineer.fit_transform(df)
    
    This applies all feature engineering steps in the correct order.
    """
    
    def __init__(self, 
                 apply_academic: bool = True,
                 apply_engagement: bool = True,
                 apply_financial: bool = True,
                 apply_demographic: bool = True,
                 apply_interactions: bool = True):
        """
        Initialize feature engineer.
        
        You can disable specific feature categories if they don't apply 
        to your dataset.
        """
        self.apply_academic = apply_academic
        self.apply_engagement = apply_engagement
        self.apply_financial = apply_financial
        self.apply_demographic = apply_demographic
        self.apply_interactions = apply_interactions
        
        self.transformers = []
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit all transformers."""
        self.transformers = []
        
        if self.apply_academic:
            self.transformers.append(AcademicFeatures())
        if self.apply_engagement:
            self.transformers.append(EngagementFeatures())
        if self.apply_financial:
            self.transformers.append(FinancialFeatures())
        if self.apply_demographic:
            self.transformers.append(DemographicFeatures())
        if self.apply_interactions:
            self.transformers.append(InteractionFeatures())
            
        for transformer in self.transformers:
            transformer.fit(X, y)
            
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations."""
        X_transformed = X.copy()
        
        for transformer in self.transformers:
            X_transformed = transformer.transform(X_transformed)
            
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_new_feature_names(self) -> List[str]:
        """Get list of all created features."""
        all_features = []
        
        for transformer in self.transformers:
            if hasattr(transformer, 'get_feature_names'):
                all_features.extend(transformer.get_feature_names())
                
        return all_features


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    """Example usage of feature engineering."""
    
    # Create synthetic data mimicking student dataset structure
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        'Curricular units 1st sem (grade)': np.random.uniform(0, 20, n_samples),
        'Curricular units 2nd sem (grade)': np.random.uniform(0, 20, n_samples),
        'Curricular units 1st sem (enrolled)': np.random.randint(1, 8, n_samples),
        'Curricular units 2nd sem (enrolled)': np.random.randint(0, 8, n_samples),
        'Curricular units 1st sem (approved)': np.random.randint(0, 6, n_samples),
        'Curricular units 2nd sem (approved)': np.random.randint(0, 6, n_samples),
        'Curricular units 1st sem (evaluations)': np.random.randint(0, 8, n_samples),
        'Curricular units 2nd sem (evaluations)': np.random.randint(0, 8, n_samples),
        'Debtor': np.random.randint(0, 2, n_samples),
        'Tuition fees up to date': np.random.randint(0, 2, n_samples),
        'Scholarship holder': np.random.randint(0, 2, n_samples),
        'Unemployment rate': np.random.uniform(5, 15, n_samples),
        'Inflation rate': np.random.uniform(0, 5, n_samples),
        'GDP': np.random.uniform(-2, 3, n_samples),
        "Mother's qualification": np.random.randint(1, 40, n_samples),
        "Father's qualification": np.random.randint(1, 40, n_samples),
        'Age at enrollment': np.random.randint(17, 55, n_samples)
    })
    
    print("🔧 FEATURE ENGINEERING DEMONSTRATION")
    print("=" * 60)
    print(f"\nOriginal features: {df.shape[1]}")
    print(f"Original columns: {list(df.columns)}")
    
    # Apply feature engineering
    engineer = StudentFeatureEngineer()
    df_engineered = engineer.fit_transform(df)
    
    print(f"\nAfter engineering: {df_engineered.shape[1]} features")
    print(f"\nNew features created:")
    for col in df_engineered.columns:
        if col not in df.columns:
            print(f"  - {col}")
    
    print("\n" + "=" * 60)
    print("📊 SAMPLE OF ENGINEERED FEATURES")
    print("=" * 60)
    
    new_cols = [col for col in df_engineered.columns if col not in df.columns]
    print(df_engineered[new_cols[:10]].head())
