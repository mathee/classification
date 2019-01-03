""" takes path as input, then performs necessary feature engineering on file,
returns X data, Y data ready for training / test """

import pandas as pd
from config import X_COLUMNS, Y_COLUMN
from functions import one_hot_encoding

###############################################################################



############################################################################### 
# NON-MODEL-SPECIFIC FEATURE ENGINEERING

def engineer_X(X_path):
    iter_csv = pd.read_csv('data/train.csv', usecols = X_COLUMNS, iterator=True, chunksize=50000) #usecols = X_COLUMNS
    X = next(iter_csv)
    X = X.dropna()
    X = one_hot_encoding(X)
    return X

def engineer_y(y_path):
    iter_csv = pd.read_csv('data/train.csv', usecols=Y_COLUMN, iterator=True, chunksize=50000)
    y = next(iter_csv)
    return y

def reduce_y_by_X(X, y):
    remaining_rows = X.index.values.tolist()
    y = y.loc[remaining_rows]
    return y

def main(X_path, y_path):
    '''prepares and returns X, y'''
    print("preparing data...")
    X = engineer_X(X_path)
    y = engineer_y(y_path)
    y = reduce_y_by_X(X, y)
    y = y.values.reshape(-1,)
    print("done")
    return X, y