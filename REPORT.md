# Student Success Prediction - Experiment Report

**Generated**: 2025-12-03 02:36:06
**Duration**: 347.2 seconds

## Executive Summary

| Metric | Baseline | Final | Improvement |
|--------|----------|-------|-------------|
| Macro F1 | 0.7174 | 0.7174 | **-0.0000** (-0.0%) |
| Dropout F1 | 0.7759 | 0.7795 | +0.0036 |
| Enrolled F1 | 0.5333 | 0.5190 | -0.0143 |
| Graduate F1 | 0.8429 | 0.8536 | +0.0107 |

## Baseline Results

- **Macro F1**: 0.7174
- **Accuracy**: 0.7590
- **Per-class F1**: Dropout=0.7759, Enrolled=0.5333, Graduate=0.8429

### Leakage Guard

The following potential post-outcome features were masked to prevent leakage:
- Curricular units 2nd sem (approved), Curricular units 2nd sem (credited), Curricular units 2nd sem (enrolled), Curricular units 2nd sem (evaluations), Curricular units 2nd sem (grade), Curricular units 2nd sem (without evaluations), approval_rate_sem2, eval_efficiency_sem2, grade_per_unit_sem2, units_without_eval_sem2, zero_enrolled_sem2

## Phase Results

### Model Training

*Best model: LightGBM (Tuned)*

- **Macro F1**: 0.7260
- **Accuracy**: 0.7892
- **Delta from baseline**: +0.0086 (+1.2%)
- **Per-class F1**: Dropout=0.8065, Enrolled=0.5072, Graduate=0.8642

### Threshold Optimization

*Optimized thresholds: [0.21428571 0.72857143 0.27142857]*

- **Macro F1**: 0.7450
- **Accuracy**: 0.7861
- **Delta from baseline**: +0.0276 (+3.8%)
- **Per-class F1**: Dropout=0.8099, Enrolled=0.5670, Graduate=0.8580

### Final Evaluation (Test Set)

*Final evaluation on held-out test set*

- **Macro F1**: 0.7174
- **Accuracy**: 0.7590
- **Delta from baseline**: -0.0000 (-0.0%)
- **Per-class F1**: Dropout=0.7795, Enrolled=0.5190, Graduate=0.8536

## Final Test Metrics

- **Model**: LightGBM (Tuned)
- **Macro F1**: 0.7174
- **Weighted F1**: 0.7699
- **Accuracy**: 0.7590
- **Per-class F1**: Dropout=0.7795, Enrolled=0.5190, Graduate=0.8536

## Best Hyperparameters

### Random Forest
```
max_depth: 19
max_features: sqrt
min_samples_leaf: 1
min_samples_split: 8
n_estimators: 108
```

### XGBoost
```
colsample_bytree: 0.6431565707973218
gamma: 0.015714592843367126
learning_rate: 0.19455901926649632
max_depth: 6
min_child_weight: 4
n_estimators: 321
reg_alpha: 0.1393314544058757
reg_lambda: 1.2088347585556345
subsample: 0.8159364365206693
```

### LightGBM
```
colsample_bytree: 0.9272059063689972
learning_rate: 0.2596118691443396
max_depth: 9
min_child_samples: 10
n_estimators: 236
num_leaves: 81
reg_alpha: 0.22210781047073025
reg_lambda: 0.2397307346673656
subsample: 0.7350460685614512
```

## Top Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | grade_improvement | 7.65% |
| 2 | grade_per_unit_sem1 | 6.80% |
| 3 | Curricular units 1st sem (grade) | 6.49% |
| 4 | Father's occupation | 5.88% |
| 5 | Course | 5.39% |
| 6 | family_education_avg | 5.34% |
| 7 | Mother's occupation | 4.91% |
| 8 | Age at enrollment | 4.58% |
| 9 | approval_rate_trend | 4.19% |
| 10 | eval_efficiency_sem1 | 3.65% |
| 11 | approval_rate_overall | 3.37% |
| 12 | Curricular units 1st sem (evaluations) | 3.15% |
| 13 | Application mode | 2.66% |
| 14 | Mother's qualification | 2.65% |
| 15 | Inflation rate | 2.51% |

## Confusion Matrices

### Final
```
              Dropout    Enrolled    Graduate
   Dropout        152          46          15
  Enrolled         19          75          25
  Graduate          6          49         277
```

## Ablation Study

Contribution of each phase to overall improvement:

| Phase | Delta F1 | Cumulative F1 |
|-------|----------|---------------|
| Model Training | +0.0086 | 0.7260 |
| Threshold Optimization | +0.0190 | 0.7450 |
| Final Evaluation (Test Set) | -0.0276 | 0.7174 |
