# Decision Tree Model Template

## Overview
This decision tree template is designed to predict student success (Dropout, Enrolled, or Graduate) based on various demographic, academic, and economic features from the dataset.

## Features

### Model Capabilities
- **Data Loading & Preprocessing**: Automatic loading and preparation of the student success dataset
- **Model Training**: Configurable decision tree classifier with customizable hyperparameters
- **Evaluation Metrics**: Comprehensive evaluation including accuracy, precision, recall, F1-score
- **Visualizations**:
  - Confusion matrix
  - Feature importance plot
  - Decision tree diagram
- **Cross-validation**: K-fold cross-validation for robust performance estimation
- **Hyperparameter Tuning**: Grid search for optimal model parameters

### Target Variable
- **0**: Dropout
- **1**: Enrolled
- **2**: Graduate

## Quick Start

### Basic Usage

```python
from decision_tree_model import StudentSuccessDecisionTree

# Initialize the model
model = StudentSuccessDecisionTree(random_state=42)

# Load and preprocess data
model.load_and_preprocess_data('dataset.csv', test_size=0.2)

# Train the model
model.train_model(max_depth=10, min_samples_split=5, min_samples_leaf=2)

# Evaluate the model
metrics = model.evaluate_model()

# Generate visualizations
model.plot_confusion_matrix()
model.plot_feature_importance(top_n=15)
model.plot_tree_diagram(max_depth=3)
```

### Running the Complete Pipeline

Simply run the script to execute the full pipeline:

```bash
python decision_tree_model.py
```

## Key Methods

### `load_and_preprocess_data(filepath, test_size)`
Loads the CSV dataset and splits it into training and testing sets.

**Parameters:**
- `filepath` (str): Path to the CSV file (default: 'dataset.csv')
- `test_size` (float): Proportion of data for testing (default: 0.2)

### `train_model(max_depth, min_samples_split, min_samples_leaf, criterion)`
Trains the decision tree classifier.

**Parameters:**
- `max_depth` (int): Maximum depth of the tree (default: None)
- `min_samples_split` (int): Minimum samples to split a node (default: 2)
- `min_samples_leaf` (int): Minimum samples in a leaf (default: 1)
- `criterion` (str): Split quality measure - 'gini' or 'entropy' (default: 'gini')

### `evaluate_model()`
Evaluates model performance and prints detailed metrics.

**Returns:** Dictionary with accuracy, precision, recall, and F1-score

### `plot_confusion_matrix(figsize)`
Displays a confusion matrix heatmap.

### `plot_feature_importance(top_n, figsize)`
Shows the most important features for predictions.

**Parameters:**
- `top_n` (int): Number of top features to display (default: 15)

### `plot_tree_diagram(max_depth, figsize)`
Visualizes the decision tree structure.

**Parameters:**
- `max_depth` (int): Maximum depth to display (default: 3)

### `cross_validate(cv)`
Performs k-fold cross-validation.

**Parameters:**
- `cv` (int): Number of folds (default: 5)

### `tune_hyperparameters(param_grid, cv)`
Performs grid search to find optimal hyperparameters.

**Parameters:**
- `param_grid` (dict): Parameters to tune (default grid provided)
- `cv` (int): Number of folds (default: 5)

## Advanced Usage

### Custom Hyperparameter Tuning

```python
# Define custom parameter grid
param_grid = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}

# Tune hyperparameters
best_params = model.tune_hyperparameters(param_grid=param_grid, cv=5)

# Re-evaluate with tuned model
metrics = model.evaluate_model()
```

### Making Predictions on New Data

```python
# Predict classes
predictions = model.predict(X_new)

# Get prediction probabilities
probabilities = model.predict_proba(X_new)
```

## Model Performance

The decision tree model provides:
- **Training and testing accuracy** to detect overfitting
- **Precision, recall, and F1-score** for each class
- **Confusion matrix** to understand misclassifications
- **Feature importance** to identify key predictors

## Important Features

Based on the dataset, key features typically include:
- Curricular units (1st and 2nd semester) - enrolled, approved, grades
- Age at enrollment
- Tuition fees up to date
- Previous qualification
- Scholarship holder status
- Debtor status
- Economic indicators (unemployment rate, inflation, GDP)

## Tips for Better Performance

1. **Prevent Overfitting**:
   - Set `max_depth` to limit tree depth (e.g., 5-15)
   - Increase `min_samples_split` (e.g., 5-20)
   - Increase `min_samples_leaf` (e.g., 2-10)

2. **Improve Accuracy**:
   - Use hyperparameter tuning with `tune_hyperparameters()`
   - Try both 'gini' and 'entropy' criteria
   - Use cross-validation to validate performance

3. **Handle Imbalanced Data**:
   - Consider using `class_weight='balanced'` parameter
   - Evaluate per-class metrics (precision/recall)

## Example Output

```
===========================================================
MODEL EVALUATION
===========================================================

Training Accuracy: 0.9245
Testing Accuracy:  0.7856
Precision:         0.7821
Recall:            0.7856
F1 Score:          0.7802

Classification Report (Test Set):
-----------------------------------------------------------
              precision    recall  f1-score   support

     Dropout       0.78      0.82      0.80       450
    Enrolled       0.65      0.58      0.61       125
    Graduate       0.82      0.81      0.81       512

    accuracy                           0.79      1087
   macro avg       0.75      0.74      0.74      1087
weighted avg       0.78      0.79      0.78      1087
```

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Next Steps

After running the basic template, consider:
1. Experimenting with different hyperparameters
2. Trying ensemble methods (Random Forest, Gradient Boosting)
3. Feature engineering to create new predictive features
4. Comparing with other algorithms (Logistic Regression, SVM, Neural Networks)
