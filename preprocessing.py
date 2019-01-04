""" takes path as input, then performs necessary feature engineering on file,
returns X data, Y data ready for training / test """

import pandas as pd
from config import X_COLUMNS, Y_COLUMN

###############################################################################
# PREPROCESSING FUNCTIONS

def identify_irrelevant_object_columns(X, p=0.001):
    '''X = df: deletes columns that contain object data with too many unique values in comparison to 
    the total number of rows, thus not useful for creating dummies
    p = maximum number of unique values in relation to total number of rows, in %'''
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


def reduce_y_by_X(X, y):
    remaining_rows = X.index.values.tolist()
    y = y.loc[remaining_rows]
    return y



############################################################################### 
# NON-MODEL-SPECIFIC FEATURE ENGINEERING FOR TRAINING DATA

def pipeline_Xtrain(X_path):
    iter_csv = pd.read_csv('data/train.csv', iterator=True, chunksize=20000) #usecols = X_COLUMNS
    X = next(iter_csv)
    print(f"after loading: {X.shape}")
    
    X.drop(Y_COLUMN, axis=1, inplace=True)
    
    irrelevant_object_columns = identify_irrelevant_object_columns(X, p=0.001)
    X.drop(irrelevant_object_columns, axis=1, inplace=True)
    print(f"after dropping {irrelevant_object_columns}: {X.shape}")
    
    
    columns_not_enough_values = identify_columns_not_enough_values(X, p=0.75)
    X.drop(columns_not_enough_values, axis=1, inplace=True)
    print(f"after dropping {columns_not_enough_values}: {X.shape}")   
    
    X = X.dropna()
    print(f"after dropna: {X.shape}")  

    X = one_hot_encoding(X)
    print(f"after one-hot-encoding: {X.shape}")
    return X

def pipeline_ytrain(y_path):
    iter_csv = pd.read_csv('data/train.csv', usecols=Y_COLUMN, iterator=True, chunksize=20000)
    y = next(iter_csv)
    return y

def main_train(X_path, y_path):
    '''prepares and returns X, y'''
    print("preparing data...")
    X = pipeline_Xtrain(X_path)
    y = pipeline_ytrain(y_path)
    y = reduce_y_by_X(X, y)
    y = y.values.reshape(-1,)
    print("done")
    return X, y


#pipeline_Xtrain("data/train.csv")