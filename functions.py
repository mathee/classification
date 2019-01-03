""" needs functions for:
    EXPLORATION
    + correlation matrix
    + combined columns (--> function should already be available somewhere) / interaction terms
    
    + show mean classes for specific grouped categories
    
    FEATURE ENGINEERING
    + one hot encoding: creating columns for each code, automatic naming
    + scaling 
    + further binning? a function that creates bins depending on another function, mapping, ranges etc
        or detecting columns where number of unique values not higher than XX % of total number of rows, bin those
    + filling empty rows, gaps etc
    + slicing between first and last valid row?
    + everything else should be done individually maybe...
    
"""
import pandas as pd

def identify_irrelevant_object_columns(X, p=0.001):
    '''X = df: deletes columns that contain object data with too many unique values in comparison to 
    the total number of rows, thus not useful for creating dummies'''
    object_columns = list(X.select_dtypes(include=['object']))
    non_relevant_columns = []
    for column in object_columns:
        num_unique_values = X.groupby(by=[column]).count().shape[0]
        if num_unique_values > X.shape[0] * p:
            non_relevant_columns.append(column)
        else:
            continue
    return non_relevant_columns
    

def identify_columns_not_enough_values(X, p=0.75):
    all_columns = list(X)
    non_sufficient_columns = []
    for col in all_columns:
        if X[col].count() < X.shape[0] * p:
            non_sufficient_columns.append(col)
        else:
            continue
    return non_sufficient_columns 

def one_hot_encoding(X):
    object_columns = list(X.select_dtypes(include=['object']))
    X = pd.get_dummies(X, prefix=object_columns)
    return X

def factorize_categorical_columns(X):
    #pd.factorize(data[c], sort=True)[0].astype('int32')
    return 1