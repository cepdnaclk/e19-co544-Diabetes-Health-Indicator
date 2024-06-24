#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load the training data
X_train = pd.read_csv('/home/e19452/ml_project/e19-co544-Diabetes-Health-Indicator/data/X_train.csv')
y_train = pd.read_csv('/home/e19452/ml_project/e19-co544-Diabetes-Health-Indicator/data/y_train.csv').squeeze()  # Ensure y_train is a Series, not a DataFrame

# Instantiate the model
model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, n_jobs=-1, max_iter=1000))

# Train the model
model.fit(X_train, y_train)

# Specify the path and model name
path = '/home/e19452/ml_project/e19-co544-Diabetes-Health-Indicator/models/'  # Replace with your desired directory path
model_name = 'logistic_regression.pkl'  # Replace with your desired model name

# Save the trained model to a pickle file
with open(f'{path}{model_name}', 'wb') as file:
    pickle.dump(model, file, protocol=2)


# In[ ]:




