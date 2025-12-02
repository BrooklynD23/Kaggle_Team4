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

## 4. Latest Run Highlights (2025-11-30)

The refreshed LightGBM model (with leakage guard + tuned regularization) delivered the following **test-set** performance:

| Metric | Score |
| --- | --- |
| Macro F1 | **0.7174** |
| Weighted F1 | 0.7699 |
| Accuracy | 0.7590 |
| Dropout F1 | 0.7795 |
| Enrolled F1 | 0.5190 |
| Graduate F1 | 0.8536 |

Key takeaways:

* **Leakage guard** removed 11 post–Sem 2 columns, yet the model retained strong Dropout/Graduate recall, indicating generalization improved.
* **Enrolled remains the main gap** (0.52 F1). Threshold optimization lifted it +5.6 pts vs. the tuned model but the class is still ambiguous by nature—communicate uncertainty in demos.
* **Grade trajectory + financial proxies** are now the top drivers. Focus interventions on early grade slides plus tuition delinquency.

## 5. Visual Assets for the Deck/Demo

Fresh PNG + SVG assets are generated under `artifacts/plots/` after running:

```bash
py run_pipeline.py
py -m src.evaluation.visuals
```

| Plot | Path | Talking Point |
| --- | --- | --- |
| Per-class F1 bars | `artifacts/plots/per_class_f1.png` | Shows the Dropout/Graduate strength vs. Enrolled ambiguity. Use it when motivating next steps. |
| Confusion Matrix | `artifacts/plots/confusion_matrix.png` | Highlights the main confusion pockets (Enrolled predicted as Graduate). Pair with intervention ideas. |
| Feature Importance | `artifacts/plots/feature_importance.png` | Visual proof that grade trajectory + parent education + finance dominate; anchors explainability story. |

> Tip: the SVG variants scale cleanly for slides. Keep PNGs for quick email drops.

## 6. Suggested Narrative for Presentations

1. **Start with the KPI card** (Macro F1 0.7174 / Dropout F1 0.78) and segue into the per-class bar chart.
2. **Show the confusion matrix** to explain why Enrolled is difficult and how threshold tuning mitigates false negatives.
3. **Close with the feature importance plot** to demonstrate actionable levers (grade trends, financial support, family education).
4. Highlight that the UI/API demo is driven by the same artifacts, so stakeholders see live numbers as models improve.