""" takes path as input, then performs necessary feature engineering on file,
returns X data, Y data ready for training / test """

import pandas as pd
from config import X_COLUMNS, Y_COLUMN
from sklearn.preprocessing import MinMaxScaler

###############################################################################
# PREPROCESSING FUNCTIONS

def identify_irrelevant_object_columns(X, p=0.1):
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

def identify_insignificant_object_columns(X):

    # analyzing categorical columns: prep: grouping per categories, then 
    #1 observing standard dev over means(has detections) for grouped categories --> should be high, aka high deltas
    #2 observing std over count of values for grouped categories --> should be low, aka evenly distributed.
    threshold = 0.5 # size of quadrant that has to be kept
    categorical_columns =  list(X.select_dtypes(include=['object']))
    labels = []
    mean_std = []
    count_std = []

    for col in categorical_columns:
        means = X[[col, Y_COLUMN[0]]].groupby(col)
        #stds[col] = (means.mean().std().values[0], means.count().std().values[0])
        labels.append(col)
        mean_std.append(means.mean().std().values[0]) #--> should be high for best significance
        count_std.append(means.count().std().values[0])  #--> should be low for best significance
        #single_kpi = temp

    threshold_mean_std = max(mean_std) * threshold
    threshold_count_std = max(count_std) * threshold

    cols_to_delete = []
    status = []
    for i, col in enumerate(labels):
        if ((mean_std[i] < threshold_mean_std) or (count_std[i] > threshold_count_std)):
            status.append("delete")
            cols_to_delete.append(col)
        else:
            status.append("keep")

    scatter_df = pd.DataFrame()
    scatter_df["labels"] = labels
    scatter_df["mean_std"] = mean_std
    scatter_df["count_std"] = count_std
    scatter_df["status"] = status
    
    return cols_to_delete, scatter_df


def one_hot_encoding(X):
    object_columns = list(X.select_dtypes(include=['object']))
    X = pd.get_dummies(X, prefix=object_columns)
    return X

def factorize_categorical_columns(X):
    #pd.factorize(data[c], sort=True)[0].astype('int32')
    return 1

def scale_0_1(x_column):
    '''scales column to 0-1'''
    m = MinMaxScaler()
    X = x_column.values.reshape(-1, 1)
    m.fit(X)
    out = m.transform(X)
    return out

def scale_0_1_df_except(df, columns_not_to_be_scaled):
    '''scales all columns in df to 0-1, except the ones provided as parameter (list of strings)'''
    columns_to_be_scaled = list(df)
    for i in columns_not_to_be_scaled:
        columns_to_be_scaled.remove(i)
    for i in columns_to_be_scaled:
        df[[i]] = pd.to_numeric(df[i])
        df[[i]] = scale_0_1(df[i])
    return df

def reduce_y_by_X(X, y):
    remaining_rows = X.index.values.tolist()
    y = y.loc[remaining_rows]
    return y



############################################################################### 
# NON-MODEL-SPECIFIC FEATURE ENGINEERING FOR TRAINING DATA

def pipeline_Xtrain(X_path):
    iter_csv = pd.read_csv(X_path, iterator=True, chunksize=20000) #usecols = X_COLUMNS
    X = next(iter_csv)
    print(f"after loading: {X.shape}")
    
    irrelevant_object_columns = identify_irrelevant_object_columns(X, 0.1)
    X.drop(irrelevant_object_columns, axis=1, inplace=True)
    print(f"after dropping cols that have too many categories: {X.shape}")
    
    columns_not_enough_values = identify_columns_not_enough_values(X, p=0.75)
    X.drop(columns_not_enough_values, axis=1, inplace=True)
    print(f"after dropping cols without enough values: {X.shape}")   
    
    """
    insignificants,_ = identify_insignificant_object_columns(X)
    X.drop(insignificants, axis=1, inplace=True)
    print(f"after dropping insignificant object cols: {X.shape}")  
    """
    X = X.dropna()
    print(f"after dropna: {X.shape}")  

    X = one_hot_encoding(X)
    print(f"after one-hot-encoding: {X.shape}")

    X = scale_0_1_df_except(X, Y_COLUMN)
    
    X.drop(Y_COLUMN, axis=1, inplace=True)
    return X

def pipeline_ytrain(y_path):
    iter_csv = pd.read_csv(y_path, usecols=Y_COLUMN, iterator=True, chunksize=20000)
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

#pipeline_Xtrain("data/titanic_train.csv")
#pipeline_Xtrain("data/train.csv")