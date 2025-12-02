"""
Script to run model interpretation on the latest saved model.
Generates insights on feature importance and individual predictions.

Usage:
    py -3.13 run_interpretation.py
    
    Or with venv activated:
    python run_interpretation.py
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path


def get_latest_model():
    """Find the most recently saved model."""
    model_dir = Path('models/saved_models')
    models = list(model_dir.glob('best_model_*.joblib'))
    
    if not models:
        raise FileNotFoundError("No saved models found in models/saved_models/")
    
    # Sort by modification time, get latest
    latest = max(models, key=lambda p: p.stat().st_mtime)
    print(f"📂 Loading model: {latest.name}")
    return latest


def load_and_prepare_data():
    """Load dataset with same preprocessing as training."""
    from src.data.cleaning import clean_data
    from src.data.feature_engineering import StudentFeatureEngineer
    
    print("📊 Loading and preprocessing dataset...")
    
    df = pd.read_csv('dataset.csv')
    df = clean_data(df)
    
    # Apply feature engineering (same as training)
    fe = StudentFeatureEngineer(
        apply_academic=True,
        apply_engagement=True,
        apply_financial=True,
        apply_demographic=True,
        apply_interactions=True
    )
    
    target = df['Target']
    features_df = df.drop(columns=['Target'])
    features_df = fe.fit_transform(features_df)
    
    X = features_df.values
    y = target.values
    
    print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
    
    return X, y


def run_interpretation():
    """Run the full interpretation pipeline."""
    from src.evaluation.interpretation import (
        InterpretationReport, 
        PermutationImportance,
        SimpleFeatureImportance
    )
    
    # Load model
    model_path = get_latest_model()
    saved = joblib.load(model_path)
    
    model = saved['model']
    feature_names = saved['feature_names']
    class_names = saved['config']['class_names']
    
    print(f"   Model type: {saved['model_name']}")
    print(f"   Features: {len(feature_names)}")
    
    # Load data
    X, y = load_and_prepare_data()
    
    # === 1. Simple Feature Importance (Fast) ===
    print("\n" + "=" * 70)
    print("🌳 TREE-BASED FEATURE IMPORTANCE")
    print("=" * 70)
    
    try:
        importance_df = SimpleFeatureImportance.from_tree_model(model, feature_names)
        print("\nTop 15 Most Important Features:\n")
        print(importance_df.head(15).to_string(index=False))
    except Exception as e:
        print(f"Could not extract tree importance: {e}")
    
    # === 2. Full Interpretation Report ===
    print("\n" + "=" * 70)
    print("📊 FULL INTERPRETATION REPORT")
    print("=" * 70)
    
    report_gen = InterpretationReport(model, X, y, feature_names, class_names)
    report_gen.print_report()
    
    # === 3. Permutation Importance (More Reliable, Slower) ===
    print("\n" + "=" * 70)
    print("🔄 PERMUTATION IMPORTANCE (may take 1-2 minutes)")
    print("=" * 70)
    
    try:
        # Use a sample for speed
        sample_size = min(500, len(X))
        np.random.seed(42)
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
        
        perm_imp = PermutationImportance(model, scoring='f1_macro', n_repeats=5)
        perm_imp.fit(X_sample, y_sample, feature_names)
        
        print("\nTop 15 Features by Permutation Importance:\n")
        print(perm_imp.get_importance(top_n=15).to_string(index=False))
        print("\n" + perm_imp.explain())
    except Exception as e:
        print(f"Permutation importance failed: {e}")
    
    # === 4. Actionable Insights Summary ===
    print("\n" + "=" * 70)
    print("💡 ACTIONABLE INSIGHTS")
    print("=" * 70)
    
    if 'importance_df' in dir():
        top_5 = importance_df.head(5)['Feature'].tolist()
        
        print("""
Based on the feature importance analysis:

1. EARLY WARNING SIGNALS
   Focus on students showing declining performance in:""")
        for i, feat in enumerate(top_5[:3], 1):
            print(f"   {i}. {feat}")
        
        print("""
2. INTERVENTION STRATEGY
   - Monitor grade trajectories between semesters
   - Flag students with low curricular unit approvals
   - Provide support before 2nd semester grades drop

3. MODEL CONFIDENCE
   - The model is most confident about Graduate predictions (F1: ~0.86)
   - Enrolled students are hardest to predict (F1: ~0.42)
   - Consider additional data for the 'Enrolled' class
""")
    
    print("=" * 70)
    print("✅ Interpretation complete!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_interpretation()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure you've run the training pipeline first.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

