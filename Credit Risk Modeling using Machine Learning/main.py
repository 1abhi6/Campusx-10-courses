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

# Check how many features are categorical
categorical_features = []
for i in df.columns:
    if df[i].dtype == "object":
        categorical_features.append(i)

# Chi-square test
for i in categorical_features[:-1]:
    chi2, pval, _ , _ = chi2_contingency(pd.crosstab(df[i], df[categorical_features[-1]]))
    print(i, "----", pval)
    
# Since all the categorical features have pval > 0.05, we will accept all

# Variation inflation factor(VIF) for numerical columns
numerical_columns = []
for i in df.columns:
    if df[i].dtype != "object" and i not in ["PROSPECTID", "Approved_Flag"]:
        numerical_columns.append(i)
        

# VIF sequentially check
vif_data = df[numerical_columns]
total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0

for i in range(0, total_columns):
    vif_value = variance_inflation_factor(vif_data, column_index)
    print(column_index, "---", vif_value)
    
    if vif_value <= 6:
        columns_to_be_kept.append(numerical_columns[i])
        column_index = column_index+1
    
    else:
        vif_data = vif_data.drop([numerical_columns[i]], axis=1)
        

# Check Anova for columns_to_be_kept
from scipy.stats import f_oneway

columns_to_be_kept_numerical = []

for i in columns_to_be_kept:
    a = list(df[i])
    b = list(df['Approved_Flag'])
    
    group_P1 = [value for value, group in zip(a, b) if group == "P1"]
    group_P2 = [value for value, group in zip(a, b) if group == "P2"]
    group_P3 = [value for value, group in zip(a, b) if group == "P3"]
    group_P4 = [value for value, group in zip(a, b) if group == "P4"]
    
    f_staristic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)
    
    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)
        
        














    
