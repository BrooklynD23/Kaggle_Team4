# Student Success Prediction - Experiment Report

**Generated**: 2025-11-29 09:44:33
**Duration**: 394.7 seconds

## Executive Summary

| Metric | Baseline | Final | Improvement |
|--------|----------|-------|-------------|
| Macro F1 | 0.6981 | 0.6873 | **+-0.0108** (-1.5%) |
| Dropout F1 | 0.7628 | 0.7877 | +0.0249 |
| Enrolled F1 | 0.5072 | 0.4167 | +-0.0906 |
| Graduate F1 | 0.8243 | 0.8576 | +0.0333 |

## Baseline Results

- **Macro F1**: 0.6981
- **Accuracy**: 0.7395
- **Per-class F1**: Dropout=0.7628, Enrolled=0.5072, Graduate=0.8243

## Phase Results

### Model Training

*Best model: XGBoost (Tuned)*

- **Macro F1**: 0.7252
- **Accuracy**: 0.7846
- **Delta from baseline**: +0.0271 (+3.9%)
- **Per-class F1**: Dropout=0.8083, Enrolled=0.5068, Graduate=0.8605

### Threshold Optimization

*Optimized thresholds: [0.9        0.67142857 0.84285714]*

- **Macro F1**: 0.7385
- **Accuracy**: 0.7982
- **Delta from baseline**: +0.0403 (+5.8%)
- **Per-class F1**: Dropout=0.8163, Enrolled=0.5314, Graduate=0.8676

### Final Evaluation (Test Set)

*Final evaluation on held-out test set*

- **Macro F1**: 0.6873
- **Accuracy**: 0.7636
- **Delta from baseline**: -0.0108 (-1.5%)
- **Per-class F1**: Dropout=0.7877, Enrolled=0.4167, Graduate=0.8576

## Best Hyperparameters

### Random Forest
```
max_depth: 19
max_features: None
min_samples_leaf: 1
min_samples_split: 5
n_estimators: 149
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
| 1 | None | 10.80% |
| 2 | None | 6.29% |
| 3 | None | 5.84% |
| 4 | None | 5.03% |
| 5 | None | 5.03% |
| 6 | None | 4.07% |
| 7 | None | 3.77% |
| 8 | None | 3.32% |
| 9 | None | 2.30% |
| 10 | None | 2.18% |
| 11 | None | 1.97% |
| 12 | None | 1.73% |
| 13 | None | 1.64% |
| 14 | None | 1.62% |
| 15 | None | 1.59% |

## Confusion Matrices

### Final
```
              Dropout    Enrolled    Graduate
   Dropout        167          25          21
  Enrolled         34          45          40
  Graduate         10          27         295
```

## Ablation Study

Contribution of each phase to overall improvement:

| Phase | Delta F1 | Cumulative F1 |
|-------|----------|---------------|
| Model Training | +0.0271 | 0.7252 |
| Threshold Optimization | +0.0132 | 0.7385 |
| Final Evaluation (Test Set) | -0.0511 | 0.6873 |
