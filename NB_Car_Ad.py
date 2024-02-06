# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 21:01:03 2024

@author: vaish
"""
'''
Business Problem:
    The business problem revolves around predicting whether a customer will purchase a car 
    based on certain demographic information such as age, gender, and estimated salary. 
    By understanding the factors influencing a customer's decision to purchase a car, businesses 
    can tailor their marketing strategies and target potential customers more effectively. 
    This problem is crucial for car dealerships and marketing agencies to optimize their sales 
    efforts and maximize revenue.

Business Understanding:
    Car dealerships and marketing agencies aim to understand customer behavior to enhance 
    their sales strategies. By analyzing demographic data such as age, gender, and estimated 
    salary, they can identify patterns and trends that influence purchasing decisions. 
    The goal is to develop predictive models that accurately classify customers into groups 
    based on their likelihood of purchasing a car. This understanding helps businesses target 
    their marketing efforts towards the most promising leads, ultimately increasing sales and revenue.
    Therefore, leveraging machine learning techniques like Naive Bayes classification 
    can provide valuable insights into customer behavior and inform strategic 
    decision-making processes in the automotive industry.
'''

# Import necessary libraries
import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.preprocessing import LabelEncoder  # For encoding categorical variables
from sklearn.naive_bayes import GaussianNB  # For Gaussian Naive Bayes classifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # For model evaluation

# Load the dataset
car = pd.read_csv("c:/2-dataset/NB_Car_Ad.csv.xls")

# Display basic information about the dataset
print(car.shape)  # Display the dimensions of the dataset (rows, columns)
print(car.columns)  # Display the column names
print(car.describe())  # Display summary statistics of numeric columns
print(car.info())  # Display concise summary of the dataset

# Encode categorical variables (Gender) to numerical values
label_encoder = LabelEncoder()  # Initialize label encoder object
car['Gender'] = label_encoder.fit_transform(car['Gender'])  # Encode 'Gender' column

# Define features (X) and target variable (y)
X = car[['Gender', 'Age', 'EstimatedSalary']]  # Features (input variables)
y = car['Purchased']  # Target variable (output variable)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
nb_classifier = GaussianNB()

# Train the classifier on the training data
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy score
conf_matrix = confusion_matrix(y_test, y_pred)  # Generate confusion matrix
class_report = classification_report(y_test, y_pred)  # Generate classification report

# Display the results
print(f"Accuracy: {accuracy}")  # Print accuracy score
print(f"Confusion Matrix:\n{conf_matrix}")  # Print confusion matrix
print(f"Classification Report:\n{class_report}")  # Print classification report
