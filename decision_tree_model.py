"""
Decision Tree Model Template for Student Success Prediction
============================================================
This template provides a complete decision tree implementation for predicting
student success (Dropout, Enrolled, or Graduate) based on various features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class StudentSuccessDecisionTree:
    """
    A decision tree classifier for predicting student success.

    Attributes:
        model: DecisionTreeClassifier instance
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing labels
        feature_names: Names of features used in the model
        target_names: Names of target classes
    """

    def __init__(self, random_state=42):
        """
        Initialize the decision tree model.

        Args:
            random_state (int): Random seed for reproducibility
        """
        self.model = DecisionTreeClassifier(random_state=random_state)
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_names = ['Dropout', 'Enrolled', 'Graduate']

    def load_and_preprocess_data(self, filepath='dataset.csv', test_size=0.2):
        """
        Load and preprocess the student success dataset.

        Args:
            filepath (str): Path to the CSV file
            test_size (float): Proportion of data to use for testing

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("Loading dataset...")
        df = pd.read_csv(filepath)

        # Rename nationality column if needed
        if 'Nacionality' in df.columns:
            df.rename(columns={'Nacionality': 'Nationality'}, inplace=True)

        print(f"Dataset shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")

        # Encode target variable if it's categorical
        if df['Target'].dtype == 'object':
            target_mapping = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
            df['Target'] = df['Target'].map(target_mapping)

        # Separate features and target
        X = df.drop('Target', axis=1)
        y = df['Target']

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        print(f"\nTraining set size: {len(self.X_train)}")
        print(f"Testing set size: {len(self.X_test)}")
        print(f"Target distribution in training set:\n{self.y_train.value_counts(normalize=True)}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                    criterion='gini', **kwargs):
        """
        Train the decision tree model.

        Args:
            max_depth (int): Maximum depth of the tree
            min_samples_split (int): Minimum samples required to split a node
            min_samples_leaf (int): Minimum samples required in a leaf node
            criterion (str): Function to measure split quality ('gini' or 'entropy')
            **kwargs: Additional parameters for DecisionTreeClassifier
        """
        print("\nTraining Decision Tree Model...")

        # Create model with specified parameters
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=self.random_state,
            **kwargs
        )

        # Train the model
        self.model.fit(self.X_train, self.y_train)

        print(f"Model trained successfully!")
        print(f"Tree depth: {self.model.get_depth()}")
        print(f"Number of leaves: {self.model.get_n_leaves()}")

    def evaluate_model(self):
        """
        Evaluate the model performance on training and testing sets.

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)

        # Predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        # Training metrics
        train_accuracy = accuracy_score(self.y_train, y_train_pred)

        # Testing metrics
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        test_precision = precision_score(self.y_test, y_test_pred, average='weighted')
        test_recall = recall_score(self.y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(self.y_test, y_test_pred, average='weighted')

        # Print results
        print(f"\nTraining Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy:  {test_accuracy:.4f}")
        print(f"Precision:         {test_precision:.4f}")
        print(f"Recall:            {test_recall:.4f}")
        print(f"F1 Score:          {test_f1:.4f}")

        print("\nClassification Report (Test Set):")
        print("-" * 60)
        print(classification_report(self.y_test, y_test_pred,
                                   target_names=self.target_names))

        # Return metrics dictionary
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1
        }

    def plot_confusion_matrix(self, figsize=(8, 6)):
        """
        Plot confusion matrix for the test set predictions.

        Args:
            figsize (tuple): Figure size for the plot
        """
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.target_names,
                   yticklabels=self.target_names)
        plt.title('Confusion Matrix - Student Success Prediction')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, top_n=15, figsize=(10, 8)):
        """
        Plot feature importance from the decision tree.

        Args:
            top_n (int): Number of top features to display
            figsize (tuple): Figure size for the plot
        """
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Select top N features
        top_indices = indices[:top_n]
        top_features = [self.feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]

        plt.figure(figsize=figsize)
        plt.barh(range(len(top_features)), top_importances, color='steelblue')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features for Student Success Prediction')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

        # Print feature importances
        print("\nFeature Importances:")
        print("-" * 60)
        for i in range(len(top_features)):
            print(f"{top_features[i]:45} {top_importances[i]:.4f}")

    def plot_tree_diagram(self, max_depth=3, figsize=(20, 10)):
        """
        Visualize the decision tree structure.

        Args:
            max_depth (int): Maximum depth to display (None for full tree)
            figsize (tuple): Figure size for the plot
        """
        plt.figure(figsize=figsize)
        plot_tree(self.model,
                 feature_names=self.feature_names,
                 class_names=self.target_names,
                 filled=True,
                 rounded=True,
                 max_depth=max_depth,
                 fontsize=10)
        plt.title(f'Decision Tree Visualization (max_depth={max_depth})')
        plt.tight_layout()
        plt.show()

    def cross_validate(self, cv=5):
        """
        Perform cross-validation on the model.

        Args:
            cv (int): Number of cross-validation folds

        Returns:
            tuple: (mean_score, std_score)
        """
        print(f"\nPerforming {cv}-fold cross-validation...")

        # Combine training and test data for full CV
        X_full = pd.concat([self.X_train, self.X_test])
        y_full = pd.concat([self.y_train, self.y_test])

        scores = cross_val_score(self.model, X_full, y_full, cv=cv,
                                scoring='accuracy')

        print(f"Cross-validation scores: {scores}")
        print(f"Mean CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return scores.mean(), scores.std()

    def tune_hyperparameters(self, param_grid=None, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV.

        Args:
            param_grid (dict): Dictionary of parameters to tune
            cv (int): Number of cross-validation folds

        Returns:
            dict: Best parameters found
        """
        if param_grid is None:
            param_grid = {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }

        print("\nTuning hyperparameters...")
        print(f"Parameter grid: {param_grid}")

        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=self.random_state),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.X_train, self.y_train)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        # Update model with best parameters
        self.model = grid_search.best_estimator_

        return grid_search.best_params_

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Features for prediction

        Returns:
            array: Predicted classes
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Get prediction probabilities for new data.

        Args:
            X: Features for prediction

        Returns:
            array: Prediction probabilities for each class
        """
        return self.model.predict_proba(X)


def main():
    """
    Main function to demonstrate the decision tree template usage.
    """
    print("="*60)
    print("STUDENT SUCCESS PREDICTION - DECISION TREE MODEL")
    print("="*60)

    # Initialize the model
    dt_model = StudentSuccessDecisionTree(random_state=42)

    # Load and preprocess data
    dt_model.load_and_preprocess_data('dataset.csv', test_size=0.2)

    # Train the model with default parameters
    print("\n" + "="*60)
    print("TRAINING MODEL WITH DEFAULT PARAMETERS")
    print("="*60)
    dt_model.train_model(max_depth=10, min_samples_split=5, min_samples_leaf=2)

    # Evaluate the model
    metrics = dt_model.evaluate_model()

    # Visualizations
    print("\nGenerating visualizations...")

    # Confusion Matrix
    dt_model.plot_confusion_matrix()

    # Feature Importance
    dt_model.plot_feature_importance(top_n=15)

    # Decision Tree Diagram (limited depth for readability)
    dt_model.plot_tree_diagram(max_depth=3)

    # Cross-validation
    dt_model.cross_validate(cv=5)

    # Optional: Hyperparameter tuning (commented out by default as it takes time)
    # print("\n" + "="*60)
    # print("HYPERPARAMETER TUNING")
    # print("="*60)
    # best_params = dt_model.tune_hyperparameters()
    #
    # # Re-evaluate with tuned parameters
    # print("\n" + "="*60)
    # print("EVALUATION WITH TUNED PARAMETERS")
    # print("="*60)
    # metrics_tuned = dt_model.evaluate_model()

    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
