"""PREPROCESS TRAININGDATA:
IN: csv file that contains X as well as y
OUT: separate files for X and y - preprocessed for training
"""

import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import (SelectFromModel, SelectKBest,
                                       VarianceThreshold, chi2, f_classif)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from config import (PATH_MODELS, PATH_XTRAIN,
                    PATH_XTRAIN_PREPROCESSED, PATH_YTRAIN_PREPROCESSED,
                    SEPARATOR, Y_COLUMN)
from feature_engineering import engineer_train as feature_engineer_train

###############################################################################
# GENERAL FUNCTIONS

def load_chunk(path, chunksize):
    '''loads chunk of the trainingset that will be used for training model
    IN: path: string of path to file | chunksize: int, no of rows to be read
    Out: df: pd.DataFrame'''
    iter_csv = pd.read_csv(path, iterator=True, chunksize=chunksize) #usecols = X_COLUMNS
    df = next(iter_csv)
    print(f"AFTER LOADING: {type(df)} - {df.shape}\n")
    return df

def drop_y_from_X(X):
    '''deletes y-column from X
    IN: X: pd.DataFrame
    OUT: X: pd.DataFrame | y: pd.Series'''
    y = X[Y_COLUMN]
    X.drop(Y_COLUMN, axis=1, inplace=True)
    print(f"DROPPED Y: {type(X)} - {X.shape}\n")
    return X, y

def save_column_structure(X):
    '''saves minimal dataframe with only one row to disc, is used later as a "blueprint"
    for getting the column structure right in test data
    IN: pd.DataFrame'''
    trainingdata_structure = X.head(1)
    with open(f'{PATH_MODELS}support/trainingdata.structure', 'wb') as handle:
        pickle.dump(trainingdata_structure, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def reduce_y_by_X(y, index_of_X):
    '''match rows in y using indices of X
    IN: y : pd.Series, index_of_X: pd.index
    OUT: y : pd.Series'''
    y = y.loc[index_of_X]
    return y        
        
###############################################################################
# DEALING WITH NANs
    
def drop_NaNs(X):
    '''drops all rows from dataframe that conatin NULL values
    IN & OUT: X: pd.DataFrame'''
    X = X.dropna()
    print(f"DROPPED NaNs: {type(X)} - {X.shape}\n")
    return X

def impute_numerical_NaNs(X, strategy="mean", missing_values=np.nan):
    '''performs imputation on all numeric columns within dataframe.
    strategy can be "mean", "median", "most_frequent" or "constant"
    IN: X: pd.DataFrame | stategy: string | missing_values: float
    OUT: X : pd.DataFrame'''
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
# DEALING WITH OUTLIERS
    
def clip_outliers(X, minPercentile = 0.01, maxPercentile = 0.99):
    '''clips datapoints outside of the given percentiles in whole dataframe
    IN: X : pd.DataFrame | minPercentile : float | maxPercentile : float
    OUT: X : pd.DataFrame'''
    cols = list(X)
    for col in cols:
        X[col] = X[col].clip((X[col].quantile(minPercentile)),(X[col].quantile(maxPercentile)))
    print(f"CLIPPED OUTLIERS: {type(X)} - {X.shape}\n")
    return X

def delete_z_value_outliers(X, z_value = 3):
    '''deletes rows from whole dataframe that contain outlying datapoints
    by the given magnitude of z stds
    IN: X : pd.DataFrame | z_value : float
    OUT: X : pd.DataFrame'''
    X = X[(np.abs(stats.zscore(X)) < 3).all(axis=1)]
    index_of_X = X.index.values.tolist()
    print(f"DELETED Z_VALUE OUTLIERS: {type(X)} - {X.shape}\n")
    return X, index_of_X
    
###############################################################################
# COLUMN/FEATURE SELECTION

def delete_sparse_columns(X, p=0.75):
    '''deletes columns that have less that p % rows filled with actual data
    IN: X : pd.DataFrame | p: float
    OUT: X : pd.DataFrame'''
    all_columns = list(X)
    non_sufficient_columns = []
    for col in all_columns:
        if X[col].count() < X.shape[0] * p:
            non_sufficient_columns.append(col)
        else:
            continue
    X.drop(non_sufficient_columns, axis=1, inplace=True)
    save_column_structure(X)
    print(f"DELETED SPARSE COLUMNS: {type(X)} - {X.shape}\n")
    return X

def select_high_variance_threshold(X, threshold):
    '''removes columns where the variance of the values is lower than treshold
    IN: X : pd.DataFrame | treshold
    OUT: X : pd.DataFrame'''
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
    use e.g. linear_model.Lasso for regression problems
    IN: X : pd.DataFrame | y : pd.Series | C : float
    OUT: X : pd.DataFrame'''
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
    ''' lasso (l1) based penalty to select features/columns
    IN: X : pd.DataFrame | y : pd.Series | alpha : float
    OUT: X : pd.DataFrame'''
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

def select_tree_based(X, y, n_estimators):
    '''is good for selecting categorical features, might take very long on
    huge datasets
    IN: X : pd.DataFrame | y : pd.Series | n_estimators : integer
    OUT: X : pd.DataFrame'''
    y = y.values.reshape(-1,)    
    clf = ExtraTreesClassifier(n_estimators=n_estimators)
    clf = clf.fit(X, y)
    model = SelectFromModel(clf, prefit=True)
    feature_indices = model.get_support(indices = True)
    features = [column for column in X] #Array of all nonremoved features' names
    filtered_features = [features[i] for i in feature_indices]
    X = pd.DataFrame(model.transform(X))  
    X.columns = filtered_features   
    save_column_structure(X)
    print(f"AFTER TREE-BASED SELECTION: {type(X)} - {X.shape}\n")
    return X

def select_k_best_features(X, y, k, stat):
    '''works only for postivie values, so scaling needs to be done before
    choose chi2 or f_classif as stat
    IN: X : pd.DataFrame  | y : pd.Series | k : int | stat : imported function from scipy
    OUT: X : pd.DataFrame'''
    y = y.values.reshape(-1,)
    selector = SelectKBest(stat, k)
    selector.fit(X, y)
    feature_indices = selector.get_support(indices = True)
    features = [column for column in X] #Array of all nonremoved features' names
    filtered_features = [features[i] for i in feature_indices]
    X = pd.DataFrame(selector.transform(X))
    X.columns = filtered_features
    save_column_structure(X)
    print(f"SELECTED K-BEST FEATURES: {type(X)} - {X.shape}\n")
    return X

def select_uncorrelated(X, corr_val):
    ''' Drops features that are strongly correlated to other features.
    This lowers model complexity, and aids in generalizing the model.
    IN : df: features df (x) | corr_val: Columns are dropped relative to the corr_val input (e.g. 0.8)
    OUT: df that only includes uncorrelated features
    https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on/43104383#43104383
    '''

    # Creates Correlation Matrix and Instantiates
    corr_matrix = X.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterates through Correlation Matrix Table to find correlated columns
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = item.values
            if val >= corr_val:
                # Prints the correlated feature set and the corr val
#                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(i)

    drops = sorted(set(drop_cols))[::-1]
    # Drops the correlated columns
    for i in drops:
        col = X.iloc[:, (i+1):(i+2)].columns.values
        X = X.drop(col, axis=1)
    print(f"DELETED CORRELATING FEATURES: {type(X)} - {X.shape}\n")
    save_column_structure(X)
    return X

############################################################################### 
# ENCODING OF TEXT FEATURES

def one_hot_encoding(X):
    '''creates new column for each category within categorical column'''
    object_columns = list(X.select_dtypes(include=['object']))
    X = pd.get_dummies(X, prefix=object_columns)
    save_column_structure(X)
    print(f"ONE-HOT-ENCODING: {type(X)} - {X.shape}\n")
    return X

def factorization_encoding(X):
    '''encodes categorical columns with order of appearance
    . side effect: NaNs are being replaced by -1'''
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

def frequency_encoding(X):
    '''encodes categorical values with their frequency'''
    object_columns = list(X.select_dtypes(include=['object']))
    for column in object_columns:
        lst1 = X[column].tolist()
        X[column] = X[column].map(X[column].value_counts())
        #X[f"{column}_count"] = X[column].map(X[column].value_counts())
        lst2 = X[column].tolist()
        #lst2 = X[f"{column}_count"].tolist()
        encoder = dict(zip(lst1,lst2))
        with open(f'{PATH_MODELS}support/frequency_encoder_{column}.dict', 'wb') as handle:
            pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"FREQUENCY ENCODING: {type(X)} - {X.shape}\n")
    return X
############################################################################### 
# BALANCING DATA
    
def balance_dataset(X):
    '''balances dataset '''
    y_column_label = Y_COLUMN[0]
    g = X.groupby(y_column_label)
    X = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
    print(f"BALANCED DATASET: {type(X)} - {X.shape}\n")
    return X

############################################################################### 
# SCALING
    
def scale_column(columndata, columnlabel, feature_range):
    '''scales column accoring to given feature_range, e.g. (0,1)'''
    m = MinMaxScaler(feature_range=feature_range)
    X = columndata.values.reshape(-1, 1)
    m.fit(X)
    with open(f'{PATH_MODELS}support/scaler{feature_range}{columnlabel}.model', 'wb') as handle:
        pickle.dump(m, handle, protocol=pickle.HIGHEST_PROTOCOL)
    out = m.transform(X)
    return out

def scaling(X, feature_range):
    '''scales whole dataframe'''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_columns = list(X.select_dtypes(include=numerics))
    for col in numeric_columns:
        X[col] = scale_column(X[col], col, feature_range)
    print(f"AFTER SCALING {feature_range}: {type(X)} - {X.shape}\n")
    return X


############################################################################### 
# DIMENSIONALITY REDUCTION

def principal_component_analysis(X, n_components):
    '''applies PCA on dataframe, works best with scaled data to (-1,1) '''
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
    '''saves preprocessed Xtrain to disc'''
    X.to_csv(PATH_XTRAIN_PREPROCESSED, index = False, sep = SEPARATOR)
    print(f"SAVED Xtrain TO DISC")
    
def save_preprocessed_ytrain(y):
    '''saves preprocessed ytrain to disc'''
    y.to_csv(PATH_YTRAIN_PREPROCESSED, index = False, sep = SEPARATOR)    
    print(f"SAVED ytrain TO DISC")

############################################################################### 
# MAIN FUNCTIONS

def main(trainingset_size):
    '''preprocesses and save X,y for model training'''
    X = load_chunk(PATH_XTRAIN, chunksize=trainingset_size)
    X, y = drop_y_from_X(X) #--> decouple y from X at this point, only AFTER row deletions (!)
    X = feature_engineer_train(X)
    X = factorization_encoding(X)
    X = delete_sparse_columns(X, p=0.75)
    X = impute_numerical_NaNs(X, "mean") # “median”, “most_frequent”, "mean"
    X = scaling(X, (0,1))  
#    X = select_high_variance_threshold(X, threshold=0.001)
    X = clip_outliers(X, 0.02, 0.98)
    X = select_tree_based(X, y, 10)
#    X = select_k_best_features(X, y, 50, f_classif) #26 works best for random forest, 60 for nn
    X = scaling(X, (-1,1)) # scaling -1,1 for best NN performance
    save_preprocessed_Xtrain(X)
    save_preprocessed_ytrain(y)