#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def train_models(X, y):
    models = {
        "SVC": SVC(),
        "Gaussian NB": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=18),
    }
    pipelines = {name: Pipeline([("classifier", model)]) for name, model in models.items()}
    scores = {name: cross_val_score(pipeline, X, y, cv=10, n_jobs=-1, scoring="accuracy") for name, pipeline in pipelines.items()}
    return pipelines, scores

def evaluate_model(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    train_accuracy = accuracy_score(y_train, pipeline.predict(X_train))
    test_accuracy = accuracy_score(y_test, preds)
    cf_matrix = confusion_matrix(y_test, preds)
    return train_accuracy, test_accuracy, cf_matrix

def plot_confusion_matrix(cf_matrix, model_name):
    plt.figure(figsize=(12, 8))
    sns.heatmap(cf_matrix, annot=True)
    plt.title(f"Confusion Matrix for {model_name} Classifier on Test Data")
    plt.show()

