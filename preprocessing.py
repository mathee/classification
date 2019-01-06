""" takes path as input, then performs necessary feature engineering on file,
returns X data, Y data ready for training / test """

import pandas as pd
from config import X_COLUMNS, Y_COLUMN
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

###############################################################################
# PREPROCESSING FUNCTIONS

def delete_irrelevant_object_columns(X, p=0.1):
    '''X = df: deletes columns that contain object data with too many unique values in comparison to 
    the total number of rows, thus not useful for creating dummies
    p = maximum number of unique values in relation to total number of rows, in %'''
    print("### deleting columns with too many unique categorical values")
    object_columns = list(X.select_dtypes(include=['object']))
    irrelevant_object_columns = []
    for column in object_columns:
        num_unique_values = X.groupby(by=[column]).count().shape[0]
        if num_unique_values > X.shape[0] * p:
            irrelevant_object_columns.append(column)
        else:
            continue
    X.drop(irrelevant_object_columns, axis=1, inplace=True)     
    return X
    
def delete_columns_not_enough_values(X, p=0.75):
    print("### deleting columns with too less values")
    all_columns = list(X)
    non_sufficient_columns = []
    for col in all_columns:
        if X[col].count() < X.shape[0] * p:
            non_sufficient_columns.append(col)
        else:
            continue
    X.drop(non_sufficient_columns, axis=1, inplace=True)
    return X

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
    print("### performing one-hot-encoding...")
    object_columns = list(X.select_dtypes(include=['object']))
    X = pd.get_dummies(X, prefix=object_columns)
    return X

def factorization_encoding(X):
    print("### performing factorization encoding...")
    object_columns = list(X.select_dtypes(include=['object']))
    encoder = {}
    for col in object_columns:
        labels, uniques = pd.factorize(X[col], sort=True)
        encoder[col] = uniques
        X[col] = labels.astype('int32')
    # SAVE ENCODER SOMEWHERE, SAME DIRECTORY WHERE PCA GOES
    return X

def factorization_encoding_using_encoder(X):
    # for test data:
    # use the "uniques" list from the encodig step for each column to decode back
    # if value is not in dict, then apply -1
    return 1

"""
def scale_0_1(x_column):
    '''scales column to 0-1'''
    m = MinMaxScaler()
    X = x_column.values.reshape(-1, 1)
    m.fit(X)
    out = m.transform(X)
    return out

def scale_0_1_df_except(X, columns_not_to_be_scaled):
    '''scales all columns in df to 0-1, except the ones provided as parameter (list of strings)'''
    print("### performing scaling 0-1...")
    columns_to_be_scaled = list(X)
    for i in columns_not_to_be_scaled:
        columns_to_be_scaled.remove(i)
    for i in columns_to_be_scaled:
        X[[i]] = pd.to_numeric(X[i])
        X[[i]] = scale_0_1(X[i])
    return X

def scale_0_1_df(X):
    print("### performing scaling 0-1...")
    columns_to_be_scaled = list(X)
    for i in columns_to_be_scaled:
        X[[i]] = pd.to_numeric(X[i])
        X[[i]] = scale_0_1(X[i])
    return X
"""

def delete_columns_low_variance(X, threshold=0.8):
    print("### deleting columns with low variance...")
    sel = VarianceThreshold(threshold=(threshold * (1 - threshold)))
    X = sel.fit_transform(X)
    return X

def select_k_best_features(X, y, k):
    print("### selecting k-best features...")
    X = SelectKBest(chi2, k).fit_transform(X, y)
    return X

def principal_component_analysis(X, n_components):
    print("### performing PCA...")
    X = X-X.mean() # demean data
    pca = PCA(n_components=n_components)
    pca.fit(X)
    #------> EXPORT PCA SOMEWHERE FOR REUSE IN TEST
    Xt = pca.transform(X)
    print(f"{pca.explained_variance_ratio_.sum()} explained variance in {n_components} components")
    return Xt

def reduce_y_by_X(y, index_of_X):
    y = y.loc[index_of_X]
    return y



############################################################################### 
# NON-MODEL-SPECIFIC FEATURE ENGINEERING FOR TRAINING DATA

def preprocessing_Xtrain(X_path):
    ############################# LOAD DATA (CHUNKS) ##########################
    iter_csv = pd.read_csv(X_path, iterator=True, chunksize=20000) #usecols = X_COLUMNS
    X = next(iter_csv)
    print(f"after loading: {X.shape}")

    ############################# CONVERTING EVERYTHING TO NUMERIC VALUES #####
    X = factorization_encoding(X)
    print(f"{X.shape}")

    ############################# DELETING COLUMNS THAT ARE NOT REQUIRED ######
    X = delete_columns_not_enough_values(X, p=0.75)
    print(f"{X.shape}")
#    X = delete_irrelevant_object_columns(X, 0.1)
#    print(f"{X.shape}")

    """
    insignificants,_ = identify_insignificant_object_columns(X)
    X.drop(insignificants, axis=1, inplace=True)
    print(f"after dropping insignificant object cols: {X.shape}")  
    """
    ############################# DEALING WITH NANS ###########################
    X = X.dropna()
    index_of_X = X.index.values.tolist() # remaining rows
    print(f"after dropna: {X.shape}")  
    y = X[Y_COLUMN]
    X.drop(Y_COLUMN, axis=1, inplace=True)
    print(f"after dropping y: {X.shape}")  
    
    ############################# CONVERTING EVERYTHING TO NUMERIC VALUES #####
#    X = one_hot_encoding(X)
#    print(f"{X.shape}")
#    X = factorization_encoding(X)
#    print(f"{X.shape}")
    
    ############################# SCALING #####################################
#    X = scale_0_1_df(X)    

    
    ############################# NUMERIC OPERATIONS ON CLEAN DATASET #########
    X = delete_columns_low_variance(X, threshold=0.9)
    print(f"{X.shape}")
#    X = select_k_best_features(X, y, 30)
#    print(f"{X.shape}")
#    X = principal_component_analysis(X, 10)
#    print(f"{X.shape}")

    
    return X, index_of_X # for reducing y, because of dropna etc..


def preprocessing_ytrain(y_path):
    iter_csv = pd.read_csv(y_path, usecols=Y_COLUMN, iterator=True, chunksize=20000)
    y = next(iter_csv)
    return y

def main_train(X_path, y_path):
    '''prepares and returns X, y'''
    X, index_of_X = preprocessing_Xtrain(X_path)
    y = preprocessing_ytrain(y_path)
    y = reduce_y_by_X(y, index_of_X)
    y = y.values.reshape(-1,)
    return X, y

#preprocessing_Xtrain("data/titanic_train.csv")
#preprocessing_Xtrain("data/train.csv")