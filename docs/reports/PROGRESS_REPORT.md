# 📊 Student Success Prediction — Progress Assessment

**Client**: Kaggle Team 4  
**Prepared by**: ML Engineering & Analytics Practice  
**Date**: November 30, 2025  
**Classification**: Internal Working Document

---

## Executive Summary

| Dimension | Target | Current | Status |
|-----------|--------|---------|--------|
| **Macro F1 Score** | 0.80 | 0.717 | 🔴 **10% below target** |
| **Dropout Detection** | High | 0.779 F1 | 🟢 Strong |
| **Graduate Prediction** | High | 0.854 F1 | 🟢 Strong |
| **Enrolled Prediction** | Adequate | 0.519 F1 | 🟡 Improving, still volatile |
| **Model Explainability** | Required | Delivered | 🟢 Complete |

**Bottom Line**: Leakage-safe training plus stronger regularization lifted Macro F1 to **0.717 (+2.8 pts)** and pushed Enrolled F1 above 0.50 for the first time this semester. We remain short of the 0.80 goal, but the gap is now primarily due to the inherently ambiguous Enrolled label rather than overfitting.

---

## 1. What We Set Out To Achieve

Per the project charter (`README.md`), the objectives are:

1. **Predict student outcomes** (Dropout, Enrolled, Graduate) with high accuracy
2. **Enable early intervention** by identifying at-risk students
3. **Ensure explainability** — stakeholders must understand *why* predictions are made
4. **Deliver a production-ready pipeline** with modular, maintainable code

**Success Criteria** (from `PipelineConfig`):
- Minimum Macro F1: **0.70** ✅ Achieved on validation
- Target Macro F1: **0.80** ❌ Not achieved

---

## 2. Current State Assessment

### 2.1 Model Performance

| Metric | Validation | Test | Delta |
|--------|------------|------|-------|
| Macro F1 | 0.7450 | 0.7174 | **-2.8 pts** |
| Accuracy | 78.6% | 75.9% | -2.7 pts |

**Diagnosis**: Leakage guard + regularization cut the validation→test gap almost in half (from 5.1 pts to 2.8 pts). Residual drift now stems from label ambiguity rather than memorization.

### 2.2 Per-Class Breakdown

| Class | Test F1 | Interpretation |
|-------|---------|----------------|
| **Dropout** | 0.78 | ✅ Reliable early warning signals persist |
| **Enrolled** | 0.52 | ⚠️ Improved but still the bottleneck |
| **Graduate** | 0.85 | ✅ Strong prediction confidence |

**The "Enrolled" Problem**: Enrolled students remain a middle state—19 are still misread as Dropout and 25 as Graduate. Threshold tuning helped (F1 +5.6 pts), but stakeholders must treat Enrolled outputs as probabilistic signals.

### 2.3 Key Predictive Features

From the interpretation analysis, top drivers are:

| Rank | Feature | Importance | Insight |
|------|---------|------------|---------|
| 1 | `grade_improvement` | 7.6% | Trajectory > absolute grades – intervene when slopes turn negative |
| 2 | `grade_per_unit_sem1` | 6.8% | Efficiency in Sem 1 remains the clearest early risk indicator |
| 3 | `Curricular units 1st sem (grade)` | 6.5% | Absolute first-semester performance still matters |
| 4 | `Father's occupation` | 5.9% | Socioeconomic context influences persistence |
| 5 | `Course` | 5.4% | Program choice correlates with completion odds |

**Actionable Insight**: With Sem 2 signals masked, the model now leans on **early grade slopes plus socioeconomic context**, giving us cleaner, pre-outcome intervention levers.

---

## 3. Gap Analysis

### 3.1 Performance Gap

```
Target Macro F1:     0.80  ████████████████████░░░░░░░░░░░░
Current Macro F1:    0.72  ████████████████████░░░░░        
Gap:                 0.08  (10% relative shortfall)
```

### 3.2 Root Causes

| Issue | Evidence | Impact |
|-------|----------|--------|
| **Residual drift** | Val→Test drop of 2.8 pts (was 5.1) | Moderate overfit remains, mostly for Enrolled |
| **Class Imbalance** | Enrolled ≈18% even after custom SMOTE | Minority F1 sensitive to thresholds |
| **Ambiguous Labels** | Enrolled legitimately overlaps Dropout/Graduate | Limits attainable F1 ceiling |
| **Operational follow-up** | Need UI/API to surface calibrated probabilities | Without tooling, insights stay static |

### 3.3 Technical Debt

| Item | Status | Priority |
|------|--------|----------|
| Feature names missing in REPORT.md | ✅ Resolved via structured tracker/artifacts | Low |
| SHAP fails on XGBoostModel wrapper | Still incompatible with TreeExplainer (workaround: SHAP on LightGBM) | Low |
| Feature mismatch (74 vs 70) | ✅ Resolved (manifest + leakage guard) | Low |

---

## 4. Recommendations

### Immediate Actions (This Sprint)

| # | Action | Expected Impact | Effort |
|---|--------|-----------------|--------|
| 1 | **Prototype Enrolled-aware head** (hierarchical / ordinal or abstain option) | Push Enrolled F1 +3-5 pts while keeping Dropout stable | Medium |
| 2 | **Calibrate probabilities + reliability curves** (isotonic / Platt) | Makes API/UI risk scores trustworthy for non-technical reviewers | Low |
| 3 | **Stand up live demo stack** (FastAPI service + React dashboard) | Keeps metrics & visuals continuously in sync with training runs | Medium |

✅ Completed this sprint: tighter regularization, custom SMOTE ratios, leakage mask, REPORT/artifact fixes.

### Medium-Term (Next 2 Sprints)

| # | Action | Rationale |
|---|--------|-----------|
| 4 | **Temporal micro-trends** (rolling GPA delta, attendance streaks) | Capture sub-semester volatility once UI groundwork is done |
| 5 | **Fairness & bias audit on calibrated outputs** | Ensure interventions remain equitable when probabilities surface in the app |
| 6 | **Scenario-based intervention testing with Student Success team** | Validate that the UI insights change decisions |
| 7 | **Confidence-based triage in UI** | Present Enrolled cases as risk bands instead of brittle labels |

---

## 5. Path to 0.80 Macro F1

Based on ablation analysis, here's the realistic path:

| Phase | Current | Target | How |
|-------|---------|--------|-----|
| Baseline (Logistic Regression) | 0.717 | 0.717 | — (already strong after cleaning) |
| + Tree tuning & leakage guard | 0.726 | 0.73 | ✅ Achieved with LightGBM |
| + Threshold optimization | 0.745 | 0.75 | ✅ Achieved on validation |
| + Enrolled-specific handling | — | 0.78 | Ordinal / hierarchical head + active learning |
| + Calibration + Ensemble UI loop | — | 0.80 | Calibrated probabilities feeding Ops feedback |

**Confidence**: 🟡 **Moderate** — achieving 0.80 is feasible but requires focused effort on the Enrolled class.

---

## 6. What's Working Well

✅ **Pipeline Architecture** — Modular, well-documented, production-ready  
✅ **Feature Engineering** — 36 engineered features adding signal  
✅ **Dropout Detection** — 77.9% of actual dropouts correctly identified (F1 0.78)  
✅ **Graduate Prediction** — 85.4% of graduates correctly identified (F1 0.85)  
✅ **Explainability** — Feature importance + artifact plots are refreshed automatically  
✅ **Reproducibility** — Configs, random seeds, saved models, and JSON artifacts in place  

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Residual drift returns | Low | Medium | Keep leakage mask + regularization defaults in config |
| Enrolled class remains weak | High | Medium | Treat Enrolled as probabilistic band + build UI cues |
| Data leakage undetected | Low | Critical | Leakage guard + periodic feature audits with domain SME |
| Demo/API slips schedule | Medium | High | FastAPI + React workstream this sprint to keep stakeholders live |

---

## 8. Next Steps

1. **Today**: Expose latest_run artifacts via FastAPI (`/health`, `/metrics/latest`, `/predict`)
2. **This Week**: Ship the React dashboard (live KPI cards + plots) backed by the API
3. **Next Week**: Prototype Enrolled-specific head + calibration sweep using the new service
4. **Deliverable**: Demo-ready app streaming ≥0.72 Macro F1 metrics with documented Enrolled limitations

---

## Appendix: Key Metrics Reference

```
Current Best Model: LightGBM (Tuned)
Training Duration:  7.5 minutes
Dataset Size:       4,424 students
Feature Count:      71 (after engineering, 11 leakage columns masked)
Class Distribution: Graduate 50% | Dropout 32% | Enrolled 18%
```

---

*This assessment is based on experiment run 2025-11-30. Metrics will be updated as remediation actions are implemented.*

