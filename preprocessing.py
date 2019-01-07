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

def load_chunk(path, chunksize):
    iter_csv = pd.read_csv(path, iterator=True, chunksize=chunksize) #usecols = X_COLUMNS
    df = next(iter_csv)
    index_of_df = df.index.values.tolist()
    print(f"AFTER LOADING: {type(df)} - {df.shape}\n")
    return df, index_of_df

def drop_NaNs(X):
    X = X.dropna()
    index_of_X = X.index.values.tolist()
    print(f"DROPPED NaNs: {type(X)} - {X.shape}\n")
    return X, index_of_X

def drop_y_from_X(X):
    y = X[Y_COLUMN]
    X.drop(Y_COLUMN, axis=1, inplace=True)
    print(f"DROPPED Y: {type(X)} - {X.shape}\n")
    return X,y
    
def delete_irrelevant_object_columns(X, p=0.1):
    '''X = df: deletes columns that contain object data with too many unique values in comparison to 
    the total number of rows, thus not useful for creating dummies
    p = maximum number of unique values in relation to total number of rows, in %'''
    object_columns = list(X.select_dtypes(include=['object']))
    irrelevant_object_columns = []
    for column in object_columns:
        num_unique_values = X.groupby(by=[column]).count().shape[0]
        if num_unique_values > X.shape[0] * p:
            irrelevant_object_columns.append(column)
        else:
            continue
    X.drop(irrelevant_object_columns, axis=1, inplace=True)   
    print(f"DELETED IRRELEVANT OBJECT COLUMNS: {type(X)} - {X.shape}\n")
    return X
    
def delete_columns_not_enough_values(X, p=0.75):
    all_columns = list(X)
    non_sufficient_columns = []
    for col in all_columns:
        if X[col].count() < X.shape[0] * p:
            non_sufficient_columns.append(col)
        else:
            continue
    X.drop(non_sufficient_columns, axis=1, inplace=True)
    print(f"DELETED COLUMNS WITH TOO LESS VALUES: {type(X)} - {X.shape}\n")
    return X

"""
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
"""

def one_hot_encoding(X):
    object_columns = list(X.select_dtypes(include=['object']))
    X = pd.get_dummies(X, prefix=object_columns)
    print(f"ONE-HOT-ENCODING: {type(X)} - {X.shape}\n")
    return X

def factorization_encoding(X):
    object_columns = list(X.select_dtypes(include=['object']))
    encoder = {}
    for col in object_columns:
        labels, uniques = pd.factorize(X[col], sort=True)
        encoder[col] = uniques
        X[col] = labels.astype('int32')
    # SAVE ENCODER SOMEWHERE, SAME DIRECTORY WHERE PCA GOES
    print(f"FACTORIZATION ENCODING: {type(X)} - {X.shape}\n")
    return X

def factorization_encoding_using_encoder(X):
    # for test data:
    # use the "uniques" list from the encodig step for each column to decode back
    # if value is not in dict, then apply -1
    return 1


def scale_column_0_1(column):
    '''scales column to 0-1'''
    m = MinMaxScaler()
    X = column.values.reshape(-1, 1)
    m.fit(X)
    ############### SAVE MINMAXSCALER SOMEWHERE WITH {MINMAX01_COLNAME}
    out = m.transform(X)
    return out

def scaling_0_1(X):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_columns = list(X.select_dtypes(include=numerics))
    for col in numeric_columns:
        X[col] = scale_column_0_1(X[col])
    print(f"AFTER SCALING 0-1: {type(X)} - {X.shape}\n")
    return X


def delete_columns_low_variance(X, threshold):
    #sel = VarianceThreshold(threshold=(threshold * (1 - threshold)))
    #X = sel.fit_transform(X)

    def VarianceThreshold_selector(X, threshold):
        #Select Model
        selector = VarianceThreshold(threshold) #Defaults to 0.0, e.g. only remove features with the same value in all samples
        #Fit the Model
        selector.fit(X)
        feature_indices = selector.get_support(indices = True) #returns an array of integers corresponding to nonremoved features
        features = [column for column in X] #Array of all nonremoved features' names
        filtered_features = [features[i] for i in feature_indices]
        #Format and Return
        selector = pd.DataFrame(selector.transform(X))
        selector.columns = filtered_features
        #print(filtered_features)
        return selector
    
    X = VarianceThreshold_selector(X, threshold)
    print(f"AFTER DROPPING LOW VARIANCES: {type(X)} - {X.shape}\n")

    return X

def select_k_best_features(X, y, k):
    selector = SelectKBest(chi2, k)
    selector.fit(X, y)
    feature_indices = selector.get_support(indices = True)
    features = [column for column in X] #Array of all nonremoved features' names
    filtered_features = [features[i] for i in feature_indices]
    X = pd.DataFrame(selector.transform(X))
    X.columns = filtered_features
    
    print(f"SELECTED K-BEST FEATURES: {type(X)} - {X.shape}\n")
    return X

def principal_component_analysis(X, n_components):
    X = X-X.mean() # demean data
    pca = PCA(n_components=n_components)
    pca.fit(X)
    #feature_indices = selector.get_support(indices = True)
    #------> EXPORT PCA SOMEWHERE FOR REUSE IN TEST
    Xt = pd.DataFrame(pca.transform(X))
    print(f"PERFORMED PCA: {type(Xt)} - {Xt.shape}")
    print(f"{pca.explained_variance_ratio_.sum()} explained variance in {n_components} components\n")
    return Xt

def reduce_y_by_X(y, index_of_X):
    y = y.loc[index_of_X]
    return y



############################################################################### 
# NON-MODEL-SPECIFIC FEATURE ENGINEERING FOR TRAINING DATA

def preprocessing_Xtrain(X_path):
    ############################# LOAD DATA (CHUNKS) ##########################
    X, index_of_X = load_chunk(X_path, chunksize=20000)
    ############################# CONVERTING EVERYTHING TO NUMERIC VALUES #####
    X = factorization_encoding(X)
    ############################# DELETING COLUMNS THAT ARE NOT REQUIRED ######
    X = delete_columns_not_enough_values(X, p=0.75)
#    X = delete_irrelevant_object_columns(X, 0.01)
    ############################# DEALING WITH NANS ###########################
    X, index_of_X = drop_NaNs(X)
    X, y = drop_y_from_X(X)
    ############################# SCALING #####################################
    X = scaling_0_1(X)    
    ############################# CONVERTING EVERYTHING TO NUMERIC VALUES #####
#    X = one_hot_encoding(X)
    X = factorization_encoding(X)
    ############################# NUMERIC OPERATIONS ON CLEAN DATASET #########
#    X = delete_columns_low_variance(X, threshold=0.001)
    X = select_k_best_features(X, y, 30)
#    X = principal_component_analysis(X, 10)
    return X, index_of_X # for reducing y, because of dropna etc..


def preprocessing_ytrain(y_path):
    iter_csv = pd.read_csv(y_path, usecols=Y_COLUMN, iterator=True, chunksize=20000)
    y = next(iter_csv)
    return y

def preprocess_training_data(X_path, y_path):
    '''prepares and returns X, y'''
    X, index_of_X = preprocessing_Xtrain(X_path)
    y = preprocessing_ytrain(y_path)
    y = reduce_y_by_X(y, index_of_X)
    y = y.values.reshape(-1,)
    return X, y

#preprocessing_Xtrain("data/titanic_train.csv")
#preprocessing_Xtrain("data/train.csv")