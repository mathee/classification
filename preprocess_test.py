import pandas as pd
from config import Y_COLUMN, PATH_MODELS, PATH_XTEST
from feature_engineering import engineer_test as feature_engineer_test
import pickle

###############################################################################

def load_chunk(path, chunksize, id_column_label):
    iter_csv = pd.read_csv(path, iterator=True, chunksize=chunksize) #usecols = X_COLUMNS
    df = next(iter_csv)
    id_column = df[id_column_label]
    print(f"AFTER LOADING: {type(df)} - {df.shape}\n")
    return df, id_column


def apply_column_structure_of_train(X):
    with open(f'{PATH_MODELS}support/trainingdata.structure', 'rb') as handle:
        trainingdata_structure = pickle.load(handle) # dataframe
    a1, a2 = X.align(trainingdata_structure, join='right', axis=1)
    print(f"ALIGNED COLUMN STRUCTURE: {type(a1)} - {a1.shape}\n")
    return a1

def apply_factorization_encoding(X):
    object_columns = list(X.select_dtypes(include=['object']))
    for col in object_columns:
        with open(f'{PATH_MODELS}support/factorization_encoder_{col}.dict', 'rb') as handle:
            mapping = pickle.load(handle)
        X[col]= X[col].map(mapping).fillna(-1).astype(int)
    print(f"APPLIED FACTORIZATION ENCODING: {type(X)} - {X.shape}\n")
    return X

def apply_imputation(X):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_columns = list(X.select_dtypes(include=numerics))
    for col in numeric_columns:
        with open(f'{PATH_MODELS}support/imputer_{col}.model', 'rb') as handle:
            imp = pickle.load(handle)
        X[col] = imp.transform(X[col].values.reshape(-1, 1))
    print(f"APPLIED IMPUTATION: {type(X)} - {X.shape}\n")
    return X

def apply_scaling_0_1(X):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_columns = list(X.select_dtypes(include=numerics))
    for columnlabel in numeric_columns:
        with open(f'{PATH_MODELS}support/scaler_0_1_{columnlabel}.model', 'rb') as handle:
            scaler = pickle.load(handle)
        X[columnlabel] = scaler.transform(X[columnlabel].values.reshape(-1, 1))
    print(f"APPLIED SCALING 0-1: {type(X)} - {X.shape}\n")
    return X

############################################################################### 
# NON-MODEL-SPECIFIC FEATURE ENGINEERING FOR TRAINING DATA

def preprocessing_Xtest(path):
    X, id_column = load_chunk(path, chunksize=20000, id_column_label="MachineIdentifier")
    X = apply_column_structure_of_train(X)
    X = apply_factorization_encoding(X)
    X = apply_imputation(X)
    X = apply_scaling_0_1(X)
    return X, id_column

def preprocessing_ytest(path):
    iter_csv = pd.read_csv(path, usecols=Y_COLUMN, iterator=True, chunksize=20000)
    y = next(iter_csv)
    return y

def main():
    '''prepares and returns X, y'''
    X, id_column  = preprocessing_Xtest(PATH_XTEST)
    X = feature_engineer_test(X)
    #y = preprocessing_ytest(y_path)
    #y = y.values.reshape(-1,)
    return X, id_column#, y

#preprocessing_Xtest("data/test.csv")
    