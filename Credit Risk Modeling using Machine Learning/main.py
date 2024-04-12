# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:04:29 2024

@author: Abhishek Gupta
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import warnings
import os

# Load the dataset
a1 = pd.read_excel("dataset/case_study1.xlsx")
a2 = pd.read_excel("dataset/case_study2.xlsx")

# Make the copy of original datasets
df1 = a1.copy()
df2 = a2.copy()

# Remove nulls
df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]

colunms_to_be_removed = []
for i in df2.columns:
    if df2.loc[df2[i] == -99999].shape[0] > 10000:
        colunms_to_be_removed.append(i)
        
df2 = df2.drop(colunms_to_be_removed, axis=1)

for i in df2.columns:
    df2 = df2.loc[df2[i] != -99999]

# Checking common column name
for i in df1.columns:
    if i in df2.columns:
        print(i)
    
# Merge the two dataframes
df = pd.merge(df1, df2, how="inner", left_on=['PROSPECTID'], right_on=['PROSPECTID'])






























    
