 # -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:39:25 2024

@author: vaish
"""

'''
prepare a classification model using the Naive Bayes algorithm for the salary 
dataset dataset Train and test datasets are given separately 
use both for model buliding

1. Business Problem
    The business problem presented by the dataset revolves around understanding 
    the factors influencing individuals' salaries and developing a predictive model 
    to classify them into salary categories based on their demographic and employment 
    attributes. Here's a breakdown of the business problem:
    
    1. Predictive Modeling for Salary Classification: 
        The primary objective is to build a predictive model that accurately categorizes
        individuals into two salary classes: <=50K or >50K. This classification is vital 
        for various business applications, including targeted marketing, financial 
        planning, and resource allocation.
    
    2. Optimizing Resource Allocation: 
        By accurately predicting individuals' salary levels, organizations can 
        optimize their resource allocation strategies. 
        For instance, in marketing campaigns, companies can target high-income 
        individuals with products or services tailored to their preferences, 
        thereby maximizing return on investment (ROI).
    
    3. HR and Recruitment: 
        Predictive models can assist human resources (HR) 
        departments in identifying candidates likely to command higher salaries 
        based on their attributes. This can streamline the recruitment process by 
        focusing efforts on candidates who meet the salary criteria for specific roles.
    
    4. Policy-making and Economic Analysis: 
        Understanding the distribution of salary levels across demographic groups can inform policy-making decisions 
        and facilitate economic analysis. For example, policymakers can use insights 
        from the predictive model to design interventions aimed at reducing income 
        inequality or promoting economic growth.
    
    5. Risk Management and Financial Planning: 
        Financial institutions and insurance companies can use salary classification 
        models to assess individuals' creditworthiness, determine insurance premiums,
        and manage financial risks effectively. By accurately predicting salary levels, 
        these organizations can make informed decisions about lending, underwriting, 
        and risk mitigation.
    
        Overall, the business problem involves leveraging data-driven insights 
        to better understand salary determinants, improve decision-making processes,
        and drive business outcomes across various domains.
        
2.Business Understanding
    The dataset provided contains information about individuals' demographic and employment attributes, such as age, education level, occupation, work hours per week, and native country, along with their corresponding salary levels (<=50K or >50K). Here's a business understanding based on this data:
    
    1. Demographic and Employment Attributes: The dataset includes various demographic attributes like age, education, marital status, race, and sex, as well as employment-related attributes like work class, occupation, capital gain, capital loss, and hours worked per week.
    
    2. Salary Classification: The main objective appears to be predicting whether an individual earns more or less than $50,000 annually based on their demographic and employment attributes. This classification task is crucial for various purposes, including targeted marketing, financial planning, and policy-making.
    
    3. Predictive Modeling: By analyzing this dataset, we aim to develop a predictive model that accurately classifies individuals into salary categories based on their attributes. This model can help organizations identify individuals likely to earn above or below a certain income threshold, enabling them to tailor their strategies accordingly.
    
    4. Insights Generation: Through exploratory data analysis (EDA) and predictive modeling, we can uncover insights about the factors influencing salary levels. This can include understanding the demographic and employment characteristics associated with higher salaries, identifying patterns in salary distribution across different demographic groups, and discovering any correlations or trends present in the data.
    
    5. Decision Support: The insights gained from analyzing this data can inform decision-making processes in various domains, such as recruitment, human resources management, and targeted marketing campaigns. By understanding the factors driving salary levels, organizations can make more informed decisions to optimize resource allocation and achieve their goals effectively.
'''


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

#load the dataset
salary_data_train = pd.read_csv("c:/2-dataset/SalaryData_Train.csv.xls")
salary_data_test = pd.read_csv("c:/2-dataset/SalaryData_Test.csv.xls")

# Separate features (X) and target variable (y) for both datasets
X_train = salary_data_train.drop(columns=['Salary'])  # Features for training data
y_train = salary_data_train['Salary']  # Target variable for training data
X_test = salary_data_test.drop(columns=['Salary'])  # Features for testing data

# Initialize LabelEncoder to convert categorical variables to numerical values
label_encoder = LabelEncoder()

# Encode categorical variables in both training and testing datasets
for col in X_train.select_dtypes(include=['object']).columns:
    X_train[col] = label_encoder.fit_transform(X_train[col])  # Transform and encode training data
    X_test[col] = label_encoder.transform(X_test[col])  # Transform testing data based on training data's encoding

# Train-test split for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize and train the Naive Bayes classifier
# Initialize Naive Bayes classifier
nb_classifier = GaussianNB()  
# Train the classifier using training data
nb_classifier.fit(X_train, y_train) 

# Predictions on the validation set
y_pred = nb_classifier.predict(X_val)  # Make predictions on the validation set

# Evaluate classifier performance (e.g., accuracy)
# Calculate accuracy by comparing predicted values with actual values
accuracy = (y_pred == y_val).mean() 
# Print the accuracy of the classifier on the validation set
print("Accuracy:", accuracy)  