#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle


# Load the training data
X_train = pd.read_csv('/home/e19452/ml_project/e19-co544-Diabetes-Health-Indicator/data/X_train.csv')
y_train = pd.read_csv('/home/e19452/ml_project/e19-co544-Diabetes-Health-Indicator/data/y_train.csv').squeeze()  # Ensure y_train is a Series, not a DataFrame

# Instantiate the model
model = GradientBoostingClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Specify the path and model name
path = '/home/e19452/ml_project/e19-co544-Diabetes-Health-Indicator/models/'  # Replace with your desired directory path
model_name = 'gradient_boosting.pkl'  # Replace with your desired model name

# Save the trained model to a pickle file
with open(f'{path}{model_name}', 'wb') as file:
    pickle.dump(model, file, protocol=2)


# In[ ]:




