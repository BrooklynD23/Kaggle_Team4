# 🎯 Quick Model Selection Guide
## Student Success Prediction - Decision Framework

---

## When to Use Each Model

### 📊 Quick Reference Table

| Scenario | Recommended Model | Why |
|----------|------------------|-----|
| **First baseline** | Logistic Regression | Fast, interpretable, often surprisingly good |
| **Need accuracy + speed** | Random Forest | Best balance for most cases |
| **Maximum accuracy** | XGBoost + Stacking | State-of-art, but needs tuning |
| **Production deployment** | Cascading Ensemble | Fast for easy cases, accurate for hard ones |
| **High interpretability required** | Logistic Regression | Coefficients directly explain predictions |
| **Limited data (<1000 samples)** | Random Forest or LR | Less overfitting risk |
| **Class imbalance severe** | XGBoost with class weights | Built-in handling |

---

## Decision Flowchart

```
START
  │
  ▼
Is interpretability critical? ──YES──► Logistic Regression
  │                                     (with feature importance)
  NO
  │
  ▼
Dataset size < 2000? ──YES──► Random Forest
  │                           (robust, hard to overfit)
  NO
  │
  ▼
Is inference speed critical? ──YES──► Cascading Ensemble
  │                                    (fast path for easy cases)
  NO
  │
  ▼
Do you have time for tuning? ──YES──► XGBoost/LightGBM
  │                                    (best accuracy potential)
  NO
  │
  ▼
Use Voting Ensemble (soft)
(Good default, minimal tuning)
```

---

## Expected Performance Ranges

For Student Success (3-class classification with ~4400 samples):

| Model | Expected Macro F1 | Training Time | Tuning Effort |
|-------|-------------------|---------------|---------------|
| Most Frequent Baseline | 0.20-0.30 | Instant | None |
| Stratified Random | 0.30-0.35 | Instant | None |
| Logistic Regression | 0.65-0.75 | Seconds | Low |
| Random Forest | 0.70-0.80 | Minutes | Medium |
| XGBoost | 0.72-0.82 | Minutes | High |
| LightGBM | 0.72-0.82 | Seconds | High |
| Voting Ensemble | 0.73-0.83 | Minutes | Low |
| Stacking Ensemble | 0.75-0.85 | Minutes | Medium |

---

## Hyperparameter Starting Points

### Random Forest
```python
RandomForestClassifier(
    n_estimators=200,      # Start here, increase to 500 if needed
    max_depth=10,          # Prevent overfitting
    min_samples_leaf=5,    # Regularization
    class_weight='balanced',
    random_state=42
)
```

### XGBoost
```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,           # Shallower than RF
    learning_rate=0.1,     # Lower for more trees
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

### LightGBM
```python
LGBMClassifier(
    n_estimators=200,
    num_leaves=31,         # Key parameter
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42
)
```

---

## Red Flags to Watch For

### 🚨 Overfitting Indicators
- Training F1 >> Validation F1 (gap > 0.05)
- Perfect training accuracy
- Unstable CV scores (high std)

### 🚨 Data Leakage Signs
- Unrealistically high performance (F1 > 0.95)
- Features too predictive (check feature importance)
- Target information in features

### 🚨 Class Imbalance Issues
- Very high accuracy but low Macro F1
- Some classes have F1 = 0
- Confusion matrix shows all predictions are one class

---

## Feature Engineering Priorities

1. **Grade trajectory** (improvement over semesters) - Usually top predictor
2. **Approval rate** (passing what they attempt) - Strong signal
3. **Financial stress indicators** - Leading indicator of dropout
4. **Engagement signals** (course load changes) - Early warning
5. **Interaction terms** (financial × academic risk) - Compound effects

---

## Fairness Checklist

Before deployment, verify:

- [ ] Demographic parity difference < 10%
- [ ] Equal opportunity difference < 10%
- [ ] No significant accuracy gaps across groups
- [ ] Feature importance doesn't rely heavily on protected attributes
- [ ] Interventions based on predictions are supportive, not punitive

---

## Quick Commands

```python
# Full pipeline
from src.train_pipeline import train_student_success_model
model, results = train_student_success_model('data/dataset.csv')

# Just baselines
from src.models.baselines import compare_baselines
comparison = compare_baselines(X_train, y_train, X_test, y_test)

# Fairness audit
from src.evaluation.fairness import FairnessAuditor
auditor = FairnessAuditor()
report = auditor.audit(y_true, y_pred, sensitive_features_df)
auditor.print_report(report)

# Interpret predictions
from src.evaluation.interpretation import SHAPExplainer
explainer = SHAPExplainer(model, model_type='tree')
explainer.fit(X_test, feature_names)
print(explainer.generate_narrative(sample_idx=0))
```

---

*"The best model is not always the most complex one. It's the one that solves your problem reliably, fairly, and interpretably."*
