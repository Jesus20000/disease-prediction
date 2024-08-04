#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

file_path = r'C:\Users\isa.zeynalov\Desktop\Training.csv'
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(axis=1, inplace=True)
    encoder = LabelEncoder()
    df["prognosis"] = encoder.fit_transform(df["prognosis"])
    return df, encoder

def split_data(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

