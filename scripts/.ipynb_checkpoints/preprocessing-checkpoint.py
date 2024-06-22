#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import os
import random
random.seed(1)
import matplotlib.pyplot as plt
import kaggle
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Define the path to the CSV file
file_path = './../../data/2015.csv'

# Load the CSV file into a DataFrame
brfss_2015_dataset = pd.read_csv(file_path)

# Calculate null value percentages for each column
null_value_percentages = (brfss_2015_dataset.isnull().sum() / len(brfss_2015_dataset)) * 100

"""drop feature wise"""
# Filter columns with null values less than or equal to 10%
columns_to_keep = null_value_percentages[null_value_percentages <= 10].index

target_column = 'DIABETE3'
if target_column not in columns_to_keep:
        columns_to_keep.append(target_column)

# Create a new Dataset with the filtered columns
filtered_brfss_2015_dataset = brfss_2015_dataset[columns_to_keep]

"""drop null instances row wise"""
# Drop the row that has missing values in the filtered_brfss_2015_dataset
filtered_brfss_2015_dataset = filtered_brfss_2015_dataset.dropna()

"""apply label encoder"""
# Identify categorical columns
categorical_columns = filtered_brfss_2015_dataset.select_dtypes(include=['object']).columns

# Initialize label encoder
label_encoders = {}

# Apply label encoder to each categorical column
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    filtered_brfss_2015_dataset[column] = label_encoders[column].fit_transform(filtered_brfss_2015_dataset[column])

filtered_brfss_df_selected = filtered_brfss_2015_dataset

# _RFHYPE5 = HighBP
# Change 1 to 0 so it represetnts No high blood pressure and 2 to 1 so it represents high blood pressure
filtered_brfss_df_selected['_RFHYPE5'] = filtered_brfss_df_selected['_RFHYPE5'].replace({1:0, 2:1})
filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected._RFHYPE5 != 9]


# _LMTSCL1
# 0-no, 1-arthritis and limited a lot  , 2- arthritis and limited a little
filtered_brfss_df_selected['_LMTSCL1'] = filtered_brfss_df_selected['_LMTSCL1'].replace({3: 0})
filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected['_LMTSCL1'] != 9]


# variable changed to USEEQUIP
# USEEQUIP
# 0-no 1-yes
# Health Problems Requiring Special Equipment
filtered_brfss_df_selected['USEEQUIP'] = filtered_brfss_df_selected['USEEQUIP'].replace({2:0})
filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected._RFHYPE5 != 9]


# DIFFWALK
# Difficulty Walking or Climbing Stairs
# change 2 to 0 for no. 1 is already yes
# remove 7 and 9 for don't know not sure and refused
filtered_brfss_df_selected['DIFFWALK'] = filtered_brfss_df_selected['DIFFWALK'].replace({2:0})
filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected.DIFFWALK != 7]
filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected.DIFFWALK != 9]


#_HAVARTH3 Told Have Arthritis
# change 2 to 0 for no. 1 is already yes
# remove 7 and 9 for don't know not sure and refused
filtered_brfss_df_selected['HAVARTH3'] = filtered_brfss_df_selected['HAVARTH3'].replace({2:0})
filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected.DIFFWALK != 7]
filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected.DIFFWALK != 9]


# _LMTACT1
# Limited usual activities
# 0-no, 1-arthritis and have limited usual activities , 2- arthritis and no limited usual activities
filtered_brfss_df_selected['_LMTACT1'] = filtered_brfss_df_selected['_LMTACT1'].replace({3: 0})
filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected['_LMTACT1'] != 9]
filtered_brfss_df_selected = filtered_brfss_df_selected.dropna(subset=['_LMTACT1'])




# QLACTLM2
# Activity Limitation Due to Health Problems
# 0-No, 1- Yes
filtered_brfss_df_selected['QLACTLM2'] = filtered_brfss_df_selected['QLACTLM2'].replace({2: 0})
filtered_brfss_df_selected = filtered_brfss_df_selected[~filtered_brfss_df_selected['QLACTLM2'].isin([7, 9])]
filtered_brfss_df_selected = filtered_brfss_df_selected.dropna(subset=['QLACTLM2'])



# PNEUVAC3
# Pneumonia shot ever
# 0-No 1 -yes
filtered_brfss_df_selected['PNEUVAC3'] = filtered_brfss_df_selected['PNEUVAC3'].replace({2: 0})
filtered_brfss_df_selected = filtered_brfss_df_selected[~filtered_brfss_df_selected['PNEUVAC3'].isin([7, 9])]
filtered_brfss_df_selected = filtered_brfss_df_selected.dropna(subset=['PNEUVAC3'])


# INTERNET
# Internet use in the past 30 days?
# 0-no, 1-yes
filtered_brfss_df_selected['INTERNET'] = filtered_brfss_df_selected['INTERNET'].replace({2: 0})
filtered_brfss_df_selected = filtered_brfss_df_selected[~filtered_brfss_df_selected['INTERNET'].isin([7, 9])]
filtered_brfss_df_selected = filtered_brfss_df_selected.dropna(subset=['INTERNET'])


# _HCVU651
# Respondents aged 18-64 with health care coverage
# n0-0, 1-yes

filtered_brfss_df_selected['_HCVU651'] = filtered_brfss_df_selected['_HCVU651'].replace({2: 0})
filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected['_HCVU651'] != 9]



# ALCDAY5
# Days in past 30 had alcoholic beverage
# Number of days

def convert_alcday5(value):
    if 101 <= value <= 199:  # Days per week
        return (value - 100) * 4.3  # Approximate conversion to days in 30 days
    elif 201 <= value <= 299:  # Days in past 30 days
        return value - 200
    elif value == 888:
        return 0
    else:
        return value

filtered_brfss_df_selected['ALCDAY5'] = filtered_brfss_df_selected['ALCDAY5'].apply(lambda x: convert_alcday5(x) if pd.notnull(x) else x)
filtered_brfss_df_selected['ALCDAY5'] = filtered_brfss_df_selected['ALCDAY5'].replace({888: 0})
filtered_brfss_df_selected = filtered_brfss_df_selected[~filtered_brfss_df_selected['ALCDAY5'].isin([777, 999])]
filtered_brfss_df_selected = filtered_brfss_df_selected.dropna(subset=['ALCDAY5'])



# _RFBMI5
# Overweight or obese calculated variable
# 0 - no, 1- yes

filtered_brfss_df_selected['_RFBMI5'] = filtered_brfss_df_selected['_RFBMI5'].replace({1: 0, 2: 1})
filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected['_RFBMI5'] != 9]


# WTKG3
# Computed Weight in Kilograms
# Function to convert weight in pounds to kilograms
def convert_weight_to_kg(weight):
    if 1 <= weight <= 650:
        return weight / 2.2046
    elif 9023 <= weight <= 9295:
        return (weight - 9000) / 2.2046
    else:
        return weight

filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected['WTKG3'] != 99999]
filtered_brfss_df_selected['WTKG3'] = filtered_brfss_df_selected['WTKG3'].apply(convert_weight_to_kg)


# EMPLOY1
# Employyement status
'''
1 Employed for wages 179,163 40.58 47.55
2 Self-employed 36,609 8.29 8.51
3 Out of work for 1 year or more 9,594 2.17 2.87
4 Out of work for less than 1 year 9,012 2.04 2.99
5 A homemaker 27,107 6.14 6.75
6 A student 11,551 2.62 5.72
7 Retired 132,648 30.05 17.75
8 Unable to work 31,977 7.24 6.86
'''
filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected['EMPLOY1'] != 9]

# (Optional): Combine categories 3 and 4 into a single category representing unemployed individuals
# filtered_brfss_df_selected['EMPLOY1'] = filtered_brfss_df_selected['EMPLOY1'].replace({4: 3})



# _RFHLTH
# Adults with good or better health
# poor - 0,  1- good

filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected['_RFHLTH'] != 9]
filtered_brfss_df_selected['_RFHLTH'] = filtered_brfss_df_selected['_RFHLTH'].replace({2: 0})



'''
1 Underweight
Notes: _BMI5 < 1850 (_BMI5 has 2 implied decimal places)
2 Normal Weight
Notes: 1850 <= _BMI5 < 2500
3 Overweight
Notes: 2500 <= _BMI5 < 3000
4 Obese
Notes: 30000 <= _BMI5 < 9999
'''



# _BMI5 (no changes, just note that these are BMI * 100. So for example a BMI of 4018 is really 40.18)
# BMI in your one
filtered_brfss_df_selected['_BMI5'] = filtered_brfss_df_selected['_BMI5'].div(100).round(0)



# _RFHYPE5
# Change 1 to 0 so it represetnts No high blood pressure and 2 to 1 so it represents high blood pressure
filtered_brfss_df_selected['_RFHYPE5'] = filtered_brfss_df_selected['_RFHYPE5'].replace({1:0, 2:1})
filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected._RFHYPE5 != 9]

# GENHLTH - GenHlth in your one. Original Dataset name is the first one
# This is an ordinal variable that I want to keep (1 is Excellent -> 5 is Poor)
# Remove 7 and 9 for don't know and refused
filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected.GENHLTH != 7]
filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected.GENHLTH != 9]

# DIABETE3
# going to make this ordinal. 0 is for no diabetes or only during pregnancy, 1 is for pre-diabetes or borderline diabetes, 2 is for yes diabetes
# Remove all 7 (dont knows)
# Remove all 9 (refused)
filtered_brfss_df_selected['DIABETE3'] = filtered_brfss_df_selected['DIABETE3'].replace({2:0, 3:0, 1:2, 4:1})
filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected.DIABETE3 != 7]
filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected.DIABETE3 != 9]


# Standardize MAXV02 feature (Maximal Oxygen Consumption or VO2 Max) (range 15 - 60)
# Standardize the given data into common oxygen consumption range of human
original_min = filtered_brfss_df_selected['MAXVO2_'].min()
original_max = filtered_brfss_df_selected['MAXVO2_'].max()
new_min = 15
new_max = 60

# Apply the scaling formula
filtered_brfss_df_selected['MAXVO2_'] = new_min + (filtered_brfss_df_selected['MAXVO2_'] - original_min) * (new_max - new_min) / (original_max - original_min)


# DRNKANY5 (binary value)
# alcoholic beverage in the past 30 days. (1 - yes 2 - no)
filtered_brfss_df_selected['DRNKANY5'] = filtered_brfss_df_selected['DRNKANY5'].replace({2:0})


# EDUCA 
# Education Level (options to vote - 1,2,3,4,5,6)
filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected.EDUCA != 9]


"""PHYSHLTH(physical health) typically refers to the number of days within a month
that a person reports their physical health was not good.(illness or injury)"""
filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected.PHYSHLTH != 88 ]


"""Are you limited in any way in any activities because of physical, mental, or emotional problems? (binary value)"""
# answers are 0 , 1, 2
filtered_brfss_df_selected['_LMTWRK1'] = filtered_brfss_df_selected['_LMTWRK1'].replace({1:0, 2:1, 3:2})
filtered_brfss_df_selected = filtered_brfss_df_selected[filtered_brfss_df_selected._LMTWRK1 != 9]



filtered_brfss_2015_dataset = filtered_brfss_df_selected

"""Drop features with low variance"""
# Separate features and target variable
target = "DIABETE3"
X = filtered_brfss_2015_dataset.drop(columns=[target])
y = filtered_brfss_2015_dataset[target]

# filter data according the variance
# Calculate the variance of each feature in X
feature_variances = X.var()

# Filter features with variance greater than 0.1
selected_features = feature_variances[feature_variances > 0.1].index

# Filter X to include only selected features
filtered_X = X[selected_features]

data = pd.concat([filtered_X, y], axis=1)

"""filter according to correlation of feature with the target"""
# Feature Selection: Correlation with the target variable
correlation_with_target = data.corr()[target].drop(target).sort_values(ascending=False)

# Select features based on a correlation threshold
correlation_threshold = 0.15
selected_features = correlation_with_target[abs(correlation_with_target) > correlation_threshold].index


# # select specific columns
# filtered_brfss_df_selected = data[['BPHIGH4', '_LMTSCL1', 'USEEQUIP', 'DIFFWALK', 'HAVARTH3', '_DRDXAR1',
#                                    '_LMTACT1', 'QLACTLM2', 'PNEUVAC3', 'INTERNET', '_AGE65YR', '_HCVU651',
#                                    'ALCDAY5', '_RFBMI5', 'WTKG3', 'EMPLOY1', '_RFHLTH', '_AGEG5YR',
#                                    '_AGE80', '_AGE_G', '_BMI5CAT', '_BMI5', '_RFHYPE5', 'GENHLTH',"DIABETE3"]]

import seaborn as sns

# Plot the correlation graph
plt.figure(figsize=(12, 8))
sns.barplot(x=correlation_with_target[selected_features].values, y=correlation_with_target[selected_features].index, palette="viridis")
plt.title('Correlation of Features with Target Variable (DIABETE3)')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.show()


# In[19]:


# select specific columns
selected_features = list(selected_features)
selected_features.append("DIABETE3")
filtered_brfss_df_selected = data[selected_features]


filtered_brfss_df_selected.head()


# In[20]:


"""Checking the pairwise correlation"""

# see the pairwise correlatioin of the features

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your DataFrame with 25 features is named `data`

# Calculate the correlation matrix
correlation_matrix = filtered_brfss_df_selected.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title('Pairwise Correlation Heatmap of All Features')
plt.show()


# In[21]:


# Identifying redundant features
threshold = 0.95  # You can adjust this threshold based on your criteria
redundant_pairs = set()
for i in range(correlation_matrix.shape[0]):
    for j in range(i + 1, correlation_matrix.shape[1]):
        if abs(correlation_matrix.iat[i, j]) > threshold:
            redundant_pairs.add((correlation_matrix.columns[i], correlation_matrix.columns[j]))

print("Redundant feature pairs (correlation > 0.95 or < -0.95):")
print(redundant_pairs)


# In[22]:


#filtered_brfss_df_selected = filtered_brfss_df_selected.drop(columns=['_AGE80'])
#filtered_brfss_df_selected = filtered_brfss_df_selected.drop(columns=['_AGE_G','_AGEG5YR', '_EDUCAG', 'DROCDY3_', '_DRDXAR1', '_LMTSCL1', 'FC60_'])
filtered_brfss_df_selected = filtered_brfss_df_selected.drop(columns=['_AGE_G','_AGEG5YR','_DRDXAR1','FC60_'])


# In[23]:


# Rename the columns
filtered_brfss_df_selected.rename(columns={
    'BPHIGH4': 'High_Blood_Pressure',
    '_LMTSCL1': 'Limited_Activities',
    'USEEQUIP': 'Use_of_Equipment',
    'DIFFWALK': 'Difficulty_Walking',
    'HAVARTH3': 'Arthritis_Diagnosis',
    '_DRDXAR1': 'Doctor_Diagnosed_Arthritis',
    '_LMTACT1': 'Limited_in_Activities',
    'QLACTLM2': 'Quality_of_Life_Limited',
    'PNEUVAC3': 'Pneumonia_Vaccine',
    'INTERNET': 'Internet_Usage',
    '_HCVU651': 'Health_Coverage_Under_65',
    'ALCDAY5': 'Alcohol_Consumption',
    '_RFBMI5': 'Risk_Factor_BMI',
    'WTKG3': 'Weight_in_Kilograms',
    'EMPLOY1': 'Employment_Status',
    '_RFHLTH': 'Risk_Factors_for_Poor_Health',
    '_AGE80': 'Age',
    '_BMI5CAT': 'BMI_Categories',
    '_BMI5': 'Body_Mass_Index',
    '_RFHYPE5': 'Risk_Factor_for_Hypertension',
    'GENHLTH': 'General_Health',
    'MAXVO2_' : 'Maximul Oxygen Consumption', # ml (kg * min)
    'DRNKANY5' : 'alcoholic beverage in the past 30 days',
    'EDUCA' : 'EDUCATION LEVEL',
    '_LMTWRK1' : 'Limited of acitivity',
    'DIABETE3': 'Diabetes_Diagnosis'
}, inplace=True)


# In[24]:


all_features = filtered_brfss_df_selected.columns.tolist()

# Display the list of all features
print(all_features)
print(len(all_features))


# In[25]:


from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.combine import SMOTETomek
#from sklearn.preprocessing import StandardScaler

# Assuming filtered_brfss_df_selected is your DataFrame and the target vector is 'Diabetes_Diagnosis'
# Prepare the features (X) and target (y)
X = filtered_brfss_df_selected.drop(columns=['Diabetes_Diagnosis'])
y = filtered_brfss_df_selected['Diabetes_Diagnosis']

# Apply SMOTE + Tomek Links (combination of oversampling and undersampling)
smote_tomek = SMOTETomek(random_state=42)
X, y= smote_tomek.fit_resample(X, y)


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.to_csv('./../data/X_train.csv', index=False)
X_test.to_csv('./../data/X_test.csv', index=False)
y_train.to_csv('./../data/y_train.csv', index=False)
y_test.to_csv('./../data/y_test.csv', index=False)


# In[27]:


filtered_brfss_df_selected = pd.concat([X, y], axis=1)
filtered_brfss_df_selected.to_csv('./../data/pre_processed_data.csv', index=False)


# In[ ]:




