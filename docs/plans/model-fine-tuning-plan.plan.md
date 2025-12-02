<!-- 6a9beaec-3048-4b17-bb16-19956e1c3128 c381e029-54ef-48d2-a7bc-2973588c25d1 -->
# Model Fine-Tuning Plan for Student Success Prediction

## Current Baseline

- **Best Model**: Stacking Ensemble (Macro F1: 0.7076)
- **Weak Point**: Enrolled class (F1: 0.52)
- **Class Distribution**: Graduate 50%, Dropout 32%, Enrolled 18%

---

## Phase 1: Feature Engineering Integration (~5 min setup)

**Problem**: The `StudentFeatureEngineer` in [src/data/feature_engineering.py](src/data/feature_engineering.py) creates 20+ engineered features but is NOT used in the pipeline.

**Action**: Integrate feature engineering into [src/train_pipeline.py](src/train_pipeline.py) `DataLoader.load_data()` method.

**Expected Impact**: +2-5% Macro F1 (engineered features like `grade_improvement`, `approval_rate_trend`, `financial_risk` capture predictive signals)

---

## Phase 2: Class Imbalance Handling (~10 min)

**Problem**: Enrolled class (18%) is underrepresented and underperforming.

**Actions**:

1. **Add SMOTE oversampling** for minority class in training data
2. **Optimize class weights** - compute optimal weights based on inverse frequency
3. **Add threshold tuning** - adjust classification thresholds per class

**Files to modify**: [src/train_pipeline.py](src/train_pipeline.py)

**Expected Impact**: +3-8% on Enrolled F1 specifically

---

## Phase 3: Hyperparameter Tuning (~30-40 min runtime)

**Action**: Add `RandomizedSearchCV` with 50 iterations for top models.

**Key parameters to tune**:

| Model | Parameters | Search Space |

|-------|-----------|--------------|

| XGBoost | `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight` | 50 combinations |

| LightGBM | `num_leaves`, `max_depth`, `learning_rate`, `n_estimators`, `reg_alpha`, `reg_lambda` | 50 combinations |

| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` | 30 combinations |

**Files to modify**: [src/train_pipeline.py](src/train_pipeline.py) - add `tune_hyperparameters()` method

**Expected Impact**: +2-4% Macro F1

---

## Phase 4: Ensemble Optimization (~5 min)

**Actions**:

1. **Optimize voting weights** - instead of equal weights, use validation performance as weights
2. **Add calibration** - CalibratedClassifierCV for better probability estimates
3. **Try different meta-learners** for stacking (Ridge, GradientBoosting)

**Files to modify**: [src/models/ensembles.py](src/models/ensembles.py)

**Expected Impact**: +1-2% Macro F1

---

## Phase 5: Threshold Optimization (~5 min)

**Action**: Instead of argmax(probabilities), find optimal thresholds per class to maximize Macro F1.

**Implementation**: Grid search over threshold combinations on validation set.

**Expected Impact**: +1-3% on minority class (Enrolled) without sacrificing other classes

---

## Phase 6: Documentation & Reporting

**Action**: Create `REPORT.md` in project root that documents:

- Baseline metrics before any changes
- Results after each phase with delta improvements
- Best hyperparameters found
- Feature importance rankings
- Confusion matrices comparison
- Final recommendations

**Structure**:

```
REPORT.md
├── Executive Summary (final vs baseline)
├── Phase 1: Feature Engineering Results
├── Phase 2: Class Imbalance Results  
├── Phase 3: Hyperparameter Tuning Results
├── Phase 4: Ensemble Optimization Results
├── Phase 5: Threshold Optimization Results
├── Ablation Study (contribution of each phase)
└── Appendix: Best Parameters & Feature Importance
```

**Implementation**: Add `ResultsTracker` class to log metrics after each phase, then generate markdown report.

---

## Phase 7: Comprehensive Experiment Tracking & Analysis

**Purpose**: Document ALL steps, approaches, and their individual contributions to show the technical behind-the-scenes of our work.

### 7.1 Experiment Logging Structure

Each experiment in `REPORT.md` will be documented with:

```markdown
### Experiment: [Name]
**Date**: YYYY-MM-DD HH:MM
**Phase**: [Phase Number]
**Hypothesis**: What we expected to happen
**Approach**: Detailed description of what we tried
**Configuration**: 
  - Parameters used
  - Data transformations applied
  - Model settings

**Results**:
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Macro F1 | X.XX | X.XX | +X.XX |
| Enrolled F1 | X.XX | X.XX | +X.XX |
| ...

**Analysis**: Why it worked/didn't work
**Decision**: Keep/Discard/Modify
```

### 7.2 Approach Comparison Tables

For each phase, document ALL approaches tried (even failed ones):

```markdown
## Phase X: [Name] - Approach Comparison

| Approach | Description | Macro F1 | Enrolled F1 | Kept? | Notes |
|----------|-------------|----------|-------------|-------|-------|
| Approach A | ... | 0.72 | 0.55 | ✅ | Best performer |
| Approach B | ... | 0.70 | 0.51 | ❌ | Overfitting |
| Approach C | ... | 0.71 | 0.58 | ✅ | Good for minority |
```

### 7.3 Ablation Study

Document the contribution of each component:

```markdown
## Ablation Study: What Actually Worked

| Configuration | Macro F1 | Δ from Baseline | % Contribution |
|---------------|----------|-----------------|----------------|
| Baseline (no changes) | 0.7076 | - | - |
| + Feature Engineering | 0.73 | +0.022 | 28% |
| + SMOTE | 0.74 | +0.010 | 13% |
| + Class Weights | 0.75 | +0.010 | 13% |
| + Hyperparameter Tuning | 0.77 | +0.020 | 26% |
| + Ensemble Optimization | 0.78 | +0.010 | 13% |
| + Threshold Tuning | 0.785 | +0.005 | 6% |
| **Total Improvement** | **0.785** | **+0.077** | **100%** |
```

### 7.4 Failed Approaches & Lessons Learned

Document what DIDN'T work (equally important):

```markdown
## What Didn't Work (And Why)

### Attempt: [Name]
- **What we tried**: Description
- **Expected outcome**: What we hoped for
- **Actual outcome**: What happened
- **Why it failed**: Root cause analysis
- **Lesson learned**: What we took away from this
```

### 7.5 Technical Implementation Details

For transparency, document exact code changes:

```markdown
## Technical Changes Log

### Change 1: Added Feature Engineering
**File**: `src/train_pipeline.py`
**Lines modified**: 45-67
**Before**: Raw features only
**After**: 
```python
# Code snippet showing the change
engineer = StudentFeatureEngineer()
X = engineer.fit_transform(X)
```
**Impact**: Added 23 new features
```

---

## REPORT.md Final Structure

The complete `REPORT.md` will follow this structure:

```markdown
# Student Success Prediction - Model Fine-Tuning Report

## 📊 Executive Summary
- Final Macro F1: X.XX (from baseline 0.7076)
- Total improvement: +X.XX%
- Best performing model: [Name]
- Key insight: [Main takeaway]

## 📈 Progress Timeline
| Date | Phase | Key Change | Macro F1 | Notes |
|------|-------|------------|----------|-------|
| ... | ... | ... | ... | ... |

## 🔬 Phase-by-Phase Analysis

### Phase 1: Feature Engineering
- Approaches tried: [list]
- Best approach: [name]
- Results: [metrics table]
- Contribution to final score: X%

### Phase 2: Class Imbalance
[Same structure...]

### Phase 3: Hyperparameter Tuning
[Same structure...]

### Phase 4: Ensemble Optimization
[Same structure...]

### Phase 5: Threshold Optimization
[Same structure...]

## 🧪 Ablation Study
[Contribution breakdown table]

## ❌ What Didn't Work
[Failed approaches and lessons]

## 🏆 Final Model Configuration
- Model type: [name]
- Hyperparameters: [table]
- Features used: [list]
- Preprocessing: [steps]

## 📊 Feature Importance Analysis
[Top 20 features with importance scores]

## 🎯 Confusion Matrix Comparison
- Before optimization
- After optimization
- Per-class analysis

## 📝 Recommendations for Future Work
1. [Recommendation 1]
2. [Recommendation 2]
...

## 📎 Appendix
- A: Complete hyperparameter search results
- B: All experiment logs
- C: Code snippets for key changes
- D: Runtime and resource usage
```

---

## Implementation: ResultsTracker Class

Add to `src/train_pipeline.py`:

```python
class ResultsTracker:
    """Track all experiments and generate REPORT.md"""
    
    def __init__(self):
        self.experiments = []
        self.baseline = None
        self.phases = {}
        
    def set_baseline(self, metrics: dict):
        """Record baseline metrics"""
        self.baseline = {
            'timestamp': datetime.now(),
            'metrics': metrics
        }
    
    def log_experiment(self, phase: int, name: str, approach: str,
                       config: dict, metrics: dict, kept: bool, notes: str):
        """Log an individual experiment"""
        self.experiments.append({
            'timestamp': datetime.now(),
            'phase': phase,
            'name': name,
            'approach': approach,
            'config': config,
            'metrics': metrics,
            'delta': self._calculate_delta(metrics),
            'kept': kept,
            'notes': notes
        })
    
    def log_failed_attempt(self, name: str, tried: str, expected: str, 
                           actual: str, reason: str, lesson: str):
        """Log what didn't work"""
        self.experiments.append({
            'type': 'failed',
            'timestamp': datetime.now(),
            'name': name,
            'tried': tried,
            'expected': expected,
            'actual': actual,
            'reason': reason,
            'lesson': lesson
        })
    
    def generate_report(self, output_path: str = 'REPORT.md'):
        """Generate the full markdown report"""
        # Implementation to create structured report
        pass
```

---

## Summary of Changes

| File | Changes |

|------|---------|

| [src/train_pipeline.py](src/train_pipeline.py) | Add feature engineering, SMOTE, hyperparameter tuning, threshold optimization, ResultsTracker |

| [src/models/ensembles.py](src/models/ensembles.py) | Add weighted voting, meta-learner options |

| [requirements.txt](requirements.txt) | Add `imbalanced-learn` for SMOTE |

| `REPORT.md` (new) | Document all experiments, approaches, and results |

---

## Estimated Improvements

| Metric | Current | Target | Method |

|--------|---------|--------|--------|

| Macro F1 | 0.7076 | 0.75-0.78 | All phases combined |

| Enrolled F1 | 0.52 | 0.60-0.65 | SMOTE + threshold tuning |

| Dropout F1 | 0.77 | 0.80+ | Hyperparameter tuning |

| Graduate F1 | 0.83 | 0.85+ | Feature engineering |

| Accuracy | 73.95% | 77-80% | All phases combined |

---

## Execution Order

1. Feature Engineering (quick win, foundational) → Log to REPORT.md
2. Class Imbalance (addresses biggest weakness) → Log to REPORT.md
3. Hyperparameter Tuning (systematic improvement) → Log to REPORT.md
4. Ensemble Optimization (final polish) → Log to REPORT.md
5. Threshold Optimization (fine-tuning) → Log to REPORT.md
6. Generate final REPORT.md with ablation study

### To-dos

- [ ] Integrate StudentFeatureEngineer into DataLoader.load_data()
- [ ] Add SMOTE oversampling for Enrolled class + optimize class weights
- [ ] Implement RandomizedSearchCV for XGBoost, LightGBM, RandomForest
- [ ] Add weighted voting and calibration to ensembles
- [ ] Implement per-class threshold optimization
- [ ] Implement ResultsTracker class for comprehensive experiment logging
- [ ] Document all approaches (successful AND failed) in REPORT.md
- [ ] Create ablation study showing each phase's contribution
- [ ] Add "What Didn't Work" section with lessons learned

