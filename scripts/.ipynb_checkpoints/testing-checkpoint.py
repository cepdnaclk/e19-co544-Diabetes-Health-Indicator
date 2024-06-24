#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the test data
X_test = pd.read_csv('/home/e19452/ml_project/e19-co544-Diabetes-Health-Indicator/data/X_test.csv')
y_test = pd.read_csv('/home/e19452/ml_project/e19-co544-Diabetes-Health-Indicator/data/y_test.csv').squeeze()  # Ensure y_test is a Series, not a DataFrame

# Load the training data for evaluation purposes
X_train = pd.read_csv('/home/e19452/ml_project/e19-co544-Diabetes-Health-Indicator/data/X_train.csv')
y_train = pd.read_csv('/home/e19452/ml_project/e19-co544-Diabetes-Health-Indicator/data/y_train.csv').squeeze()  # Ensure y_train is a Series, not a DataFrame

# List of models and their corresponding paths
models = {
    'Random Forest': '/home/e19452/ml_project/e19-co544-Diabetes-Health-Indicator/models/random_forest.pkl',
    'Logistic Regression': '/home/e19452/ml_project/e19-co544-Diabetes-Health-Indicator/models/logistic_regression.pkl',
    'Gradient Boosting': '/home/e19452/ml_project/e19-co544-Diabetes-Health-Indicator/models/gradient_boosting.pkl',
    'SVM': '/home/e19452/ml_project/e19-co544-Diabetes-Health-Indicator/models/svm_model.pkl'
}

# Dictionary to store evaluation results
results = {}

# Function to evaluate a model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Training data evaluation
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_confusion_mat = confusion_matrix(y_train, y_train_pred)
    
    # Testing data evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    test_confusion_mat = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, test_confusion_mat, train_accuracy, train_confusion_mat

# Evaluate each model
for model_name, model_path in models.items():
    # Load the model from file
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # Evaluate the model
    accuracy, report, test_confusion_mat, train_accuracy, train_confusion_mat = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Store the results
    results[model_name] = {
        'accuracy': accuracy,
        'training set accuracy': train_accuracy,
        'report': report,
        'test_confusion_matrix': test_confusion_mat,
        'train_confusion_matrix': train_confusion_mat
    }

# # Print and visualize results for each model
# for model_name, result in results.items():
#     print(f"Model: {model_name}")
#     print(f"Testing data Accuracy: {result['accuracy']:.4f}")
#     print(f"Training data Accuracy: {result['training set accuracy']:.4f}")
#     print("Classification Report:")
#     print(result['report'])
#     print("Test Confusion Matrix:")
#     print(result['test_confusion_matrix'])
#     print("Train Confusion Matrix:")
#     print(result['train_confusion_matrix'])
#     print("\n")
    
#     # Visualize confusion matrices
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#     # Test Confusion Matrix
#     sns.heatmap(result['test_confusion_matrix'], annot=True, fmt="d", cmap="Blues", ax=axes[0])
#     axes[0].set_title(f'Confusion Matrix: {model_name} (Test Data)')
#     axes[0].set_xlabel('Predicted')
#     axes[0].set_ylabel('Actual')

#     # Train Confusion Matrix
#     sns.heatmap(result['train_confusion_matrix'], annot=True, fmt="d", cmap="Blues", ax=axes[1])
#     axes[1].set_title(f'Confusion Matrix: {model_name} (Train Data)')
#     axes[1].set_xlabel('Predicted')
#     axes[1].set_ylabel('Actual')

#     plt.tight_layout()
#     plt.show()


# In[ ]:




