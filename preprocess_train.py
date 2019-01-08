""" takes path as input, then performs necessary feature engineering on file,
returns X data, Y data ready for training / test """

import pandas as pd
from config import Y_COLUMN, PATH_MODELS, CHUNKSIZE_TRAIN, PATH_XTRAIN, PATH_YTRAIN, PATH_XTRAIN_PREPROCESSED, PATH_YTRAIN_PREPROCESSED, SEPARATOR
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from feature_engineering import engineer_train as feature_engineer_train
from sklearn.feature_selection import SelectFromModel
import pickle
import numpy as np

###############################################################################
# GENERAL FUNCTIONS

def load_chunk(path, chunksize):
    iter_csv = pd.read_csv(path, iterator=True, chunksize=chunksize) #usecols = X_COLUMNS
    df = next(iter_csv)
    index_of_df = df.index.values.tolist()
    print(f"AFTER LOADING: {type(df)} - {df.shape}\n")
    return df, index_of_df

def drop_y_from_X(X):
    '''deletes y-column in case it is still present in X data'''
    y = X[Y_COLUMN]
    X.drop(Y_COLUMN, axis=1, inplace=True)
    print(f"DROPPED Y: {type(X)} - {X.shape}\n")
    return X,y

def save_column_structure(X):
    '''saves dataframe with one row to disc, is used than later as a "blueprint"
    for the test data, e.g. if one-hot-encoded features differe between train/test'''
    trainingdata_structure = X.head(1)
    with open(f'{PATH_MODELS}support/trainingdata.structure', 'wb') as handle:
        pickle.dump(trainingdata_structure, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def reduce_y_by_X(y, index_of_X):
    '''match rowcount in y with rowcount in X, using indices,in case some rows
    have been deleted due to del NaN or so'''
    y = y.loc[index_of_X]
    return y        
        
###############################################################################
# DEALING WITH NANs
    
def drop_NaNs(X):
    X = X.dropna()
    index_of_X = X.index.values.tolist()
    print(f"DROPPED NaNs: {type(X)} - {X.shape}\n")
    return X, index_of_X

def impute_numerical_NaNs(X, strategy="mean", missing_values=np.nan):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_columns = list(X.select_dtypes(include=numerics))
    for col in numeric_columns:
        imp = SimpleImputer(missing_values, strategy)
        columndata = X[col].values.reshape(-1, 1)
        imp.fit(columndata) 
        X[col] = imp.transform(columndata)
        with open(f'{PATH_MODELS}support/imputer_{col}.model', 'wb') as handle:
            pickle.dump(imp, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return X

###############################################################################
# COLUMN/FEATURE SELECTION

def delete_sparse_columns(X, p=0.75):
    all_columns = list(X)
    non_sufficient_columns = []
    for col in all_columns:
        if X[col].count() < X.shape[0] * p:
            non_sufficient_columns.append(col)
        else:
            continue
    X.drop(non_sufficient_columns, axis=1, inplace=True)
    save_column_structure(X)
    print(f"DELETED COLUMNS WITH TOO LESS VALUES: {type(X)} - {X.shape}\n")
    return X

def select_high_variance_threshold(X, threshold):
    #Select Model
    selector = VarianceThreshold(threshold) #Defaults to 0.0, e.g. only remove features with the same value in all samples
    #Fit the Model
    selector.fit(X)
    feature_indices = selector.get_support(indices = True) #returns an array of integers corresponding to nonremoved features
    features = [column for column in X] #Array of all nonremoved features' names
    filtered_features = [features[i] for i in feature_indices]
    #Format and Return
    X = pd.DataFrame(selector.transform(X))
    X.columns = filtered_features

    save_column_structure(X)
    print(f"AFTER DROPPING LOW VARIANCES: {type(X)} - {X.shape}\n")
    return X

def select_L1_based_classification(X, y, C):
    ''' lasso (l1) based penalty to select features, svc for classification
    use e.g. linear_model.Lasso for regression problems'''
    y = y.values.reshape(-1,)
    lsvc = LinearSVC(C=C, penalty="l1", dual=False).fit(X, y) # C, e.g. 0.007
    model = SelectFromModel(lsvc, prefit=True)
    feature_indices = model.get_support(indices = True)
    features = [column for column in X] #Array of all nonremoved features' names
    filtered_features = [features[i] for i in feature_indices]
    X = pd.DataFrame(model.transform(X))   
    X.columns = filtered_features
    save_column_structure(X)
    print(f"AFTER L1-BASED SELECTION: {type(X)} - {X.shape}\n")
    return X

def select_L1_based_regression(X, y, alpha):
    ''' lasso (l1) based penalty to select features, svc for classification
    use e.g. linear_model.Lasso for regression problems'''
    y = y.values.reshape(-1,)
    lrgr = Lasso(alpha=alpha) # alpha e.g. 0.2
    model = SelectFromModel(lrgr, prefit=True)
    feature_indices = model.get_support(indices = True)
    features = [column for column in X] #Array of all nonremoved features' names
    filtered_features = [features[i] for i in feature_indices]
    X = pd.DataFrame(model.transform(X))   
    X.columns = filtered_features
    save_column_structure(X)
    print(f"AFTER L1-BASED SELECTION: {type(X)} - {X.shape}\n")
    return X

def select_k_best_features(X, y, k):
    selector = SelectKBest(chi2, k)
    selector.fit(X, y)
    feature_indices = selector.get_support(indices = True)
    features = [column for column in X] #Array of all nonremoved features' names
    filtered_features = [features[i] for i in feature_indices]
    X = pd.DataFrame(selector.transform(X))
    X.columns = filtered_features
    
    save_column_structure(X)
    
    print(f"SELECTED K-BEST FEATURES: {type(X)} - {X.shape}\n")
    return X

############################################################################### 
# ENCODING OF TEXT FEATURES

def one_hot_encoding(X):
    object_columns = list(X.select_dtypes(include=['object']))
    X = pd.get_dummies(X, prefix=object_columns)
    save_column_structure(X)
    print(f"ONE-HOT-ENCODING: {type(X)} - {X.shape}\n")
    return X

def factorization_encoding(X):
    object_columns = list(X.select_dtypes(include=['object']))

    for col in object_columns:
        labels, uniques = pd.factorize(X[col], sort=True)
        X[col] = labels.astype('int32')
        #create encoding dict for applying on test data
        encoder = {}
        for i, value in enumerate(uniques):
            encoder[value] = i
        with open(f'{PATH_MODELS}support/factorization_encoder_{col}.dict', 'wb') as handle:
            pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"FACTORIZATION ENCODING: {type(X)} - {X.shape}\n")
    return X

############################################################################### 
# SCALING

def scale_column_0_1(columndata, columnlabel):
    '''scales column to 0-1'''
    m = MinMaxScaler()
    X = columndata.values.reshape(-1, 1)
    m.fit(X)
    with open(f'{PATH_MODELS}support/scaler_0_1_{columnlabel}.model', 'wb') as handle:
        pickle.dump(m, handle, protocol=pickle.HIGHEST_PROTOCOL)
    out = m.transform(X)
    return out

def scaling_0_1(X):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_columns = list(X.select_dtypes(include=numerics))
    for col in numeric_columns:
        X[col] = scale_column_0_1(X[col], col)
    print(f"AFTER SCALING 0-1: {type(X)} - {X.shape}\n")
    return X

############################################################################### 
# DIMENSIONALITY REDUCTION

def principal_component_analysis(X, n_components):
    X = X-X.mean() # demean data
    pca = PCA(n_components=n_components)
    pca.fit(X)
    Xt = pd.DataFrame(pca.transform(X))
    with open(f'{PATH_MODELS}support/PCA.model', 'wb') as handle:
        pickle.dump(pca, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"PERFORMED PCA: {type(Xt)} - {Xt.shape}")
    print(f"{pca.explained_variance_ratio_.sum()} explained variance in {n_components} components\n")
    return Xt

############################################################################### 
# SAVING
    
def save_preprocessed_Xtrain(X):
    X.to_csv(PATH_XTRAIN_PREPROCESSED, index = False, sep = SEPARATOR)
    print(f"SAVED X TO DISC")
    
def save_preprocessed_ytrain(y):
    y.to_csv(PATH_YTRAIN_PREPROCESSED, index = False, sep = SEPARATOR)    
    print(f"SAVED y TO DISC")

############################################################################### 
# MAIN FUNCTIONS

def preprocessing_Xtrain(path):
    X, index_of_X = load_chunk(path, chunksize=CHUNKSIZE_TRAIN)
    X = factorization_encoding(X)
#    X = one_hot_encoding(X)
    X = delete_sparse_columns(X, p=0.75)
#    X = delete_irrelevant_object_columns(X, 0.01)
#    X, index_of_X = drop_NaNs(X)
    X = impute_numerical_NaNs(X, "mean") # “median”, “most_frequent”, "mean"
    X = feature_engineer_train(X)
    X, y = drop_y_from_X(X)
    X = scaling_0_1(X)    
#    X = select_high_variance_threshold(X, threshold=0.001)
#    X = select_L1_based_classification(X, y, 0.009)
    X = select_k_best_features(X, y, 26)
#    X = principal_component_analysis(X, 10)
    return X, index_of_X # passing index for reducing y, because of dropna etc..

def preprocessing_ytrain(path):
    iter_csv = pd.read_csv(path, usecols=Y_COLUMN, iterator=True, chunksize=CHUNKSIZE_TRAIN)
    y = next(iter_csv)
    return y

def main():
    '''prepares and returns X, y'''
    X, index_of_X = preprocessing_Xtrain(PATH_XTRAIN)
    y = preprocessing_ytrain(PATH_YTRAIN)
    y = reduce_y_by_X(y, index_of_X)
    save_preprocessed_Xtrain(X)
    save_preprocessed_ytrain(y)


# just for testing:    
#preprocessing_Xtrain("data/titanic_train.csv")
#preprocessing_Xtrain(PATH_XTRAIN)