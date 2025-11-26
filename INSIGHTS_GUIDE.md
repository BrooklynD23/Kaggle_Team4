# 🧠 Model Insights & Interpretation Guide

This guide explains how to extract meaningful insights from the Student Success Prediction models. We go beyond simple accuracy metrics to understand *why* the model makes predictions and ensure it acts fairly.

## 1. Feature Importance & Interpretation

We use **SHAP (SHapley Additive exPlanations)** and **Permutation Importance** to understand drivers of student success.

### How to Run Interpretation
You can generate a full interpretation report using the `InterpretationReport` class.

```python
from src.evaluation.interpretation import InterpretationReport
from src.train_pipeline import train_student_success_model

# 1. Train the model
model, results = train_student_success_model('dataset.csv', save_model=False)

# 2. Load data (or use the data from the pipeline if you modify the return values)
import pandas as pd
from src.data.cleaning import clean_data
from src.train_pipeline import DataLoader

loader = DataLoader()
df = loader.load_data('dataset.csv')
df = clean_data(df)
X, y, feature_names = loader.prepare_data(df)

# 3. Generate Report
report_gen = InterpretationReport(model, X, y, feature_names)
report_gen.print_report()
```

### What to Look For
*   **Global Importance**: Which features are the biggest drivers overall? (e.g., `Curricular units 2nd sem (grade)` is often #1).
*   **Sample Explanations**: Look at specific students. Why was a student predicted to dropout?
    *   *Example*: "This student was flagged because their 1st semester approval rate was low, despite having average grades."

## 2. Fairness Audit

It is critical to ensure the model does not discriminate against specific demographic groups.

### How to Run a Fairness Audit
Use the `FairnessAuditor` to check for disparities.

```python
from src.evaluation.fairness import FairnessAuditor

# Assume you have your model, X_test, y_test from the pipeline
# And a dataframe 'sensitive_df' containing columns like 'Gender', 'Marital status'

auditor = FairnessAuditor(positive_class=2) # 2 = Graduate
report = auditor.audit(y_test, model.predict(X_test), sensitive_df)
auditor.print_report(report)
```

### Key Metrics
*   **Demographic Parity**: Are we predicting "Graduate" at the same rate for all groups?
*   **Equal Opportunity**: Are we correctly identifying actual graduates at the same rate? (Crucial for not missing successful students from underrepresented groups).

## 3. Error Analysis

Understanding *where* the model fails is as important as knowing its accuracy.

### Confusion Matrix Analysis
Look at the confusion matrix in the training output:
*   **Dropout vs. Enrolled**: Confusing these is common. Dropouts often look like Enrolled students until they suddenly leave.
*   **Enrolled vs. Graduate**: "Enrolled" is often a middle ground.

### Actionable Insights
*   **If Grade Trajectory is top predictor**: Intervene early when grades slip between Sem 1 and Sem 2.
*   **If Financial Factors are high**: Financial aid counseling might be more effective than academic tutoring for some students.
