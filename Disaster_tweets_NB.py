# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 21:10:03 2024

@author: vaish
"""
'''
Business Understanding:
    The business problem involves classifying tweets as either related to real disasters or not. 
    This classification is important for various applications, such as disaster response coordination, 
    news filtering, and social media monitoring. By accurately identifying tweets that discuss real 
    disasters, emergency responders and relevant organizations can efficiently allocate resources 
    and provide timely assistance. On the other hand, filtering out irrelevant tweets helps prevent 
    misinformation and ensure that users receive accurate information during critical events.

Business Problem:
    The task is to develop a machine learning model that can predict whether a tweet is about a real 
    disaster or not. This involves analyzing various features of the tweet, such as its content, 
    keywords, and location, to determine its relevance to real disasters. The goal is to build a 
    predictive model that can accurately classify tweets into two categories: those related to real 
    disasters (target = 1) and those that are not (target = 0). This model can be used by disaster 
    response teams, news agencies, and social media platforms to improve the accuracy and efficiency 
    of information dissemination during emergencies.
'''

# Importing necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Loading the dataset
tweet = pd.read_csv("c:/2-dataset/Disaster_tweets_NB.csv.xls")

# Preprocessing the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(tweet['text'])

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, tweet['target'], test_size=0.2, random_state=42)

# Initializing the Naive Bayes classifier
nb_classifier = MultinomialNB()

# Training the classifier
nb_classifier.fit(X_train, y_train)

# Predicting on the test set
y_pred = nb_classifier.predict(X_test)

# Calculating the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

