# Student Success Prediction - Kaggle Team 4

**CPP's DS & AI: Kaggle Team 4 Semester Project**

A machine learning project focused on predicting student academic outcomes (Dropout, Enrolled, or Graduate) using demographic, socioeconomic, and academic performance data.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Repository Structure](#repository-structure)
- [Current Implementation](#current-implementation)
- [Getting Started](#getting-started)
- [Results & Insights](#results--insights)
- [Future Work](#future-work)
- [Requirements](#requirements)
- [Team](#team)

---

## Project Overview

This project addresses the critical challenge of early identification of at-risk students in higher education. By leveraging machine learning techniques, we aim to predict student outcomes and provide insights that can inform intervention strategies to improve student retention and success rates.

### Context

Student dropout rates in higher education represent a significant challenge for educational institutions worldwide. Understanding the factors that contribute to student success or dropout can help:

- **Educational Institutions**: Allocate resources more effectively and implement targeted support programs
- **Students**: Receive timely interventions and support when needed
- **Policy Makers**: Develop evidence-based policies to improve educational outcomes
- **Researchers**: Understand the complex interplay of factors affecting student success

### Project Goals

1. **Predict** student outcomes with high accuracy using machine learning models
2. **Identify** key factors that most strongly influence student success or dropout
3. **Provide** actionable insights for early intervention strategies
4. **Compare** multiple machine learning approaches to find the most effective solution

---

## Problem Statement

**Primary Question**: Can we accurately predict whether a student will dropout, remain enrolled, or graduate based on their demographic, socioeconomic, and academic performance data?

**Challenge**: Student success is influenced by a complex combination of factors including:
- Academic preparedness and performance
- Socioeconomic background
- Financial stability (tuition payment status)
- Family education levels
- Economic conditions
- Age and enrollment timing

**Objective**: Develop a predictive model that can:
- Classify students into three categories: **Dropout (0)**, **Enrolled (1)**, or **Graduate (2)**
- Achieve high precision and recall across all classes
- Identify the most important predictive features
- Provide interpretable results for educational stakeholders

---

## Dataset Description

### Overview
- **Source**: Student academic records from higher education institution
- **Size**: 4,424 student records
- **Features**: 35 variables (34 predictors + 1 target)
- **Target Variable**: Student outcome (Dropout, Enrolled, Graduate)

### Feature Categories

#### 1. Demographic Information
- **Marital status**: Student's marital status
- **Gender**: Student's gender (0 = Female, 1 = Male)
- **Age at enrollment**: Student's age when enrolling
- **Nationality**: Student's nationality code
- **International**: International student status (0 = No, 1 = Yes)

#### 2. Educational Background
- **Previous qualification**: Type of previous education completed
- **Application mode**: Method of application to the institution
- **Application order**: Application preference order
- **Course**: Enrolled course/program code
- **Daytime/evening attendance**: Class attendance mode

#### 3. Family Background
- **Mother's qualification**: Educational level of student's mother
- **Father's qualification**: Educational level of student's father
- **Mother's occupation**: Occupational category of mother
- **Father's occupation**: Occupational category of father

#### 4. Socioeconomic Factors
- **Displaced**: Whether student is displaced from their home region
- **Educational special needs**: Special educational needs status
- **Debtor**: Student debt status (0 = No, 1 = Yes)
- **Tuition fees up to date**: Payment status (0 = No, 1 = Yes)
- **Scholarship holder**: Scholarship status (0 = No, 1 = Yes)

#### 5. Academic Performance - First Semester
- **Curricular units 1st sem (credited)**: Number of credited units
- **Curricular units 1st sem (enrolled)**: Number of enrolled units
- **Curricular units 1st sem (evaluations)**: Number of evaluations
- **Curricular units 1st sem (approved)**: Number of approved units
- **Curricular units 1st sem (grade)**: Average grade
- **Curricular units 1st sem (without evaluations)**: Units without evaluation

#### 6. Academic Performance - Second Semester
- **Curricular units 2nd sem (credited)**: Number of credited units
- **Curricular units 2nd sem (enrolled)**: Number of enrolled units
- **Curricular units 2nd sem (evaluations)**: Number of evaluations
- **Curricular units 2nd sem (approved)**: Number of approved units
- **Curricular units 2nd sem (grade)**: Average grade
- **Curricular units 2nd sem (without evaluations)**: Units without evaluation

#### 7. Economic Indicators
- **Unemployment rate**: National unemployment rate at enrollment time
- **Inflation rate**: National inflation rate at enrollment time
- **GDP**: Gross Domestic Product indicator at enrollment time

#### 8. Target Variable
- **Target**: Student outcome
  - **0 = Dropout**: Student left the program
  - **1 = Enrolled**: Student currently enrolled
  - **2 = Graduate**: Student successfully graduated

---

## Repository Structure

```
Kaggle_Team4/
│
├── README.md                      # This file - comprehensive project documentation
├── DECISION_TREE_README.md        # Detailed decision tree model documentation
│
├── dataset.csv                    # Complete student success dataset (4,424 records)
│
├── main.ipynb                     # Jupyter notebook with exploratory data analysis
│   ├── Data loading and cleaning
│   ├── Statistical analysis
│   ├── Correlation analysis
│   ├── Distribution visualizations
│   └── Feature relationship exploration
│
└── decision_tree_model.py         # Decision tree classifier implementation
    ├── StudentSuccessDecisionTree class
    ├── Data preprocessing pipeline
    ├── Model training and evaluation
    ├── Visualization methods
    └── Hyperparameter tuning
```

---

## Current Implementation

### 1. Exploratory Data Analysis (main.ipynb)

Our initial analysis includes:

**Data Quality Assessment**
- No missing values detected in the dataset
- All 4,424 records are complete
- Proper data types for all features

**Key Findings from EDA**
- **Class Distribution**: Dataset contains all three target classes with varying proportions
- **Strong Predictors Identified**:
  - Semester grades (both 1st and 2nd semester) show strong correlation with outcomes
  - Number of approved curricular units is highly predictive
  - Tuition fee payment status significantly affects outcomes
  - Age at enrollment shows varying impact across outcomes

**Visualizations Created**
- Correlation heatmap of all numerical features
- Distribution plots for semester grades by outcome
- Box plots showing grade distributions across target classes
- Count plots for categorical variables like tuition payment status

### 2. Decision Tree Model (decision_tree_model.py)

**Implementation Highlights**

```python
class StudentSuccessDecisionTree:
    """Complete decision tree pipeline for student success prediction"""
```

**Key Features**:
- **Automated Pipeline**: End-to-end workflow from data loading to evaluation
- **Configurable Parameters**: Easy adjustment of tree depth, split criteria, etc.
- **Comprehensive Evaluation**: Multiple metrics for thorough assessment
- **Rich Visualizations**: Tree diagrams, confusion matrices, feature importance
- **Cross-Validation**: K-fold validation for robust performance estimates
- **Hyperparameter Tuning**: Grid search for optimal parameter selection

**Model Capabilities**:
1. Data loading and preprocessing with train/test split
2. Model training with customizable hyperparameters
3. Performance evaluation with multiple metrics
4. Confusion matrix generation
5. Feature importance analysis
6. Decision tree visualization
7. Cross-validation
8. Automated hyperparameter tuning

---

## Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

### Running the Analysis

#### Option 1: Exploratory Data Analysis

```bash
jupyter notebook main.ipynb
```

Run all cells to see:
- Data inspection and cleaning
- Statistical summaries
- Correlation analysis
- Distribution visualizations
- Feature relationships

#### Option 2: Decision Tree Model

**Quick Start** - Run the complete pipeline:
```bash
python decision_tree_model.py
```

This will:
1. Load and preprocess the data
2. Train a decision tree classifier
3. Evaluate model performance
4. Generate all visualizations
5. Display results and metrics

**Custom Usage** - Use as a module:
```python
from decision_tree_model import StudentSuccessDecisionTree

# Initialize model
model = StudentSuccessDecisionTree(random_state=42)

# Load data (80/20 train/test split)
model.load_and_preprocess_data('dataset.csv', test_size=0.2)

# Train with specific parameters
model.train_model(
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    criterion='gini'
)

# Evaluate performance
metrics = model.evaluate_model()

# Generate visualizations
model.plot_confusion_matrix()
model.plot_feature_importance(top_n=15)
model.plot_tree_diagram(max_depth=3)

# Perform cross-validation
model.cross_validate(cv=5)

# Optional: Tune hyperparameters
best_params = model.tune_hyperparameters()
```

---

## Results & Insights

### Model Performance

**Expected Metrics** (will vary based on hyperparameters):
- **Training Accuracy**: ~85-95%
- **Testing Accuracy**: ~75-85%
- **Precision**: ~75-85% (weighted average)
- **Recall**: ~75-85% (weighted average)
- **F1-Score**: ~75-85% (weighted average)

### Key Insights

**Most Important Features for Prediction**:
1. **Curricular units 2nd sem (grade)**: Second semester average grade
2. **Curricular units 1st sem (grade)**: First semester average grade
3. **Curricular units 2nd sem (approved)**: Number of approved units in semester 2
4. **Curricular units 1st sem (approved)**: Number of approved units in semester 1
5. **Tuition fees up to date**: Payment compliance status
6. **Age at enrollment**: Student age when enrolling
7. **Curricular units enrolled**: Course load indicators
8. **Scholarship holder**: Financial support status

**Patterns Observed**:
- Students with higher grades in both semesters are significantly more likely to graduate
- Tuition payment status is a strong predictor of dropout risk
- Course load (enrolled vs. approved units) ratio indicates student struggle
- Older students at enrollment show different outcome patterns
- Economic indicators have moderate but measurable impact

**Class-Specific Performance**:
- **Dropout Prediction**: Generally high recall (catches most at-risk students)
- **Graduate Prediction**: Typically highest precision and recall
- **Enrolled Prediction**: Most challenging due to smaller sample size and temporal nature

---

## Future Work

### Immediate Next Steps

1. **Model Comparison**
   - Implement Random Forest classifier
   - Develop Gradient Boosting models (XGBoost, LightGBM)
   - Test Support Vector Machines (SVM)
   - Experiment with Neural Networks
   - Ensemble multiple models for improved predictions

2. **Feature Engineering**
   - Create ratio features (approved/enrolled units)
   - Generate interaction terms between key features
   - Develop time-based features
   - Create aggregate family education metrics
   - Engineer economic trend indicators

3. **Advanced Analysis**
   - Perform SHAP analysis for better interpretability
   - Conduct feature selection studies
   - Analyze model behavior across different student segments
   - Investigate bias and fairness across demographic groups

### Medium-Term Goals

4. **Model Optimization**
   - Extensive hyperparameter optimization using Bayesian methods
   - Implement class balancing techniques (SMOTE, class weights)
   - Develop ensemble voting classifiers
   - Create stacking models combining multiple algorithms

5. **Validation & Robustness**
   - Implement stratified k-fold cross-validation
   - Time-series validation if temporal data available
   - Test model stability across different data subsets
   - Assess generalization to similar institutions

6. **Deployment Preparation**
   - Package model as API service
   - Create web interface for predictions
   - Develop real-time prediction pipeline
   - Build monitoring and retraining workflows

### Long-Term Vision

7. **Advanced Techniques**
   - Deep learning models (LSTM for sequential data)
   - Transfer learning from similar educational datasets
   - Automated machine learning (AutoML) implementation
   - Reinforcement learning for intervention timing

8. **Intervention System**
   - Build early warning system dashboard
   - Develop intervention recommendation engine
   - Create personalized student success plans
   - Implement A/B testing framework for interventions

9. **Expanded Scope**
   - Incorporate additional data sources (attendance, engagement)
   - Multi-institutional model comparison
   - Longitudinal analysis of intervention effectiveness
   - Causal inference studies

---

## Requirements

### Core Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
jupyter>=1.0.0
```

### Optional (for future implementations)
```
xgboost>=1.7.0
lightgbm>=3.3.0
shap>=0.41.0
optuna>=3.0.0
```

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Modern web browser for Jupyter notebooks

---

## Team

**Kaggle Team 4**
California State Polytechnic University, Pomona
Data Science & AI Course - Semester Project

---

## Contributing

Team members can contribute by:
1. Creating a new branch for your feature
2. Making your changes with clear commits
3. Running tests to ensure code quality
4. Creating a pull request with detailed description

---

## License

This project is part of an educational course at Cal Poly Pomona.

---

## Acknowledgments

- Dataset provided as part of Kaggle competition
- California State Polytechnic University, Pomona - Data Science & AI Program
- Course instructors and teaching assistants

---

## Contact

For questions or collaboration opportunities, please contact the team through the course portal.

---

## Project Status

**Current Phase**: Initial Model Development
**Last Updated**: November 2025
**Status**: Active Development

### Completed
- ✅ Exploratory Data Analysis
- ✅ Decision Tree baseline model
- ✅ Basic visualizations
- ✅ Feature importance analysis

### In Progress
- 🔄 Model performance optimization
- 🔄 Additional model implementations

### Planned
- 📋 Ensemble methods
- 📋 Advanced feature engineering
- 📋 Model deployment preparation
- 📋 Comprehensive comparison study

---

**Repository**: [BrooklynD23/Kaggle_Team4](https://github.com/BrooklynD23/Kaggle_Team4)
**Branch**: `claude/decision-tree-template-01Y8tnu2gMtHnpLnugY5UiJN`
