"""PREPROCESS TESTDATA
Applying preprocessing steps on testdata so that it has the same shape/features
as the preprocessed trainingdata that the model has been trained on."""

import pickle
import pandas as pd
from config import (CHUNKSIZE_TEST, ID_COLUMN_LABEL, PATH_MODELS,
                    PATH_SUBMISSION_FILE_PREP, PATH_XTEST,
                    PATH_XTEST_PREPROCESSED, PATH_YTEST, SEPARATOR, Y_COLUMN)
from feature_engineering import engineer_test as feature_engineer_test

###############################################################################

def load_iterator(path=PATH_XTEST, chunksize=CHUNKSIZE_TEST):
    '''loads the iterator which is used to generate chunks of the testset
    to make predictions on'''
    iter_csv = pd.read_csv(path, iterator=True, chunksize=chunksize)
    return iter_csv

def load_chunk(iterator):
    '''creates dataframe form iterator'''
    df = next(iterator)
    id_column = df[ID_COLUMN_LABEL]
    print(f"AFTER LOADING: {type(df)} - {df.shape}")
    return df, id_column

def apply_column_structure_of_train(X):
    '''applies column structure of trainingdata on testdata, e.g. in case one-
    hot-encodings on train result in different col numbers as on test, then columns
    that appear only in test will be ignored'''
    with open(f'{PATH_MODELS}support/trainingdata.structure', 'rb') as handle:
        trainingdata_structure = pickle.load(handle) # dataframe
    a1, a2 = X.align(trainingdata_structure, join='right', axis=1)
    print(f"ALIGNED COLUMN STRUCTURE: {type(a1)} - {a1.shape}")
    return a1

def apply_one_hot_encoding(X):
    '''applies one_hot-encoding on testdata'''
    object_columns = list(X.select_dtypes(include=['object']))
    X = pd.get_dummies(X, prefix=object_columns)
    print(f"ONE-HOT-ENCODING: {type(X)} - {X.shape}")
    return X

def apply_factorization_encoding(X):
    '''applies the same encoding that has been made on trainingdata on testdata.
    if a string it not known from trainingset it becomes a "-1" '''
    object_columns = list(X.select_dtypes(include=['object']))
    for col in object_columns:
        with open(f'{PATH_MODELS}support/factorization_encoder_{col}.dict', 'rb') as handle:
            mapping = pickle.load(handle)
        X[col]= X[col].map(mapping).fillna(-1).astype(int)
    print(f"APPLIED FACTORIZATION ENCODING: {type(X)} - {X.shape}")
    return X

def apply_frequency_encoding(X):
    '''applies the same encoding that has been made on trainingdata on testdata.
    if a string it not known from trainingset it becomes a "-1" '''
    object_columns = list(X.select_dtypes(include=['object']))
    for col in object_columns:
        with open(f'{PATH_MODELS}support/frequency_encoder_{col}.dict', 'rb') as handle:
            mapping = pickle.load(handle)
        X[col]= X[col].map(mapping).fillna(-1).astype(int)
    print(f"APPLIED FREQUENCY ENCODING: {type(X)} - {X.shape}")
    return X

def apply_imputation(X):
    '''applies the same imputation that has been made on trainingdata on testdata'''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_columns = list(X.select_dtypes(include=numerics))
    for col in numeric_columns:
        with open(f'{PATH_MODELS}support/imputer_{col}.model', 'rb') as handle:
            imp = pickle.load(handle)
        X[col] = imp.transform(X[col].values.reshape(-1, 1))
    print(f"APPLIED IMPUTATION: {type(X)} - {X.shape}")
    return X

def apply_scaling(X, feature_range):
    '''applies the same scaling that has been made on trainingdata on testdata'''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_columns = list(X.select_dtypes(include=numerics))
    for columnlabel in numeric_columns:
        with open(f'{PATH_MODELS}support/scaler{feature_range}{columnlabel}.model', 'rb') as handle:
            scaler = pickle.load(handle)
        X[columnlabel] = scaler.transform(X[columnlabel].values.reshape(-1, 1))
    print(f"APPLIED SCALING {feature_range}: {type(X)} - {X.shape}")
    return X

def apply_PCA(X):
    '''applies the same PCA that has been made on trainingdata on testdata'''
    with open(f'{PATH_MODELS}support/PCA.model', 'rb') as handle:
        pca = pickle.load(handle)
    X = pd.DataFrame(pca.transform(X))
    print(f"APPLIED PCA: {type(X)} - {X.shape}\n")
    return X

############################################################################### 
# SAVING

def save_preprocessed_Xtest(X, chunk):
    X.to_csv(f"{PATH_XTEST_PREPROCESSED}_{chunk}.csv", index = False, sep = SEPARATOR)
    print(f"SAVED Xtest CHUNK {chunk} TO DISC\n")    
    
def save_submission_frame(id_column, chunk):
    '''optional, prepare file for competition submission, e.g. kaggle'''
    submission = pd.DataFrame(id_column, columns=[ID_COLUMN_LABEL])
    submission.to_csv(f"{PATH_SUBMISSION_FILE_PREP}_{chunk}.csv", index = False, sep = SEPARATOR)    
    print(f"PREPARED SUBMISSION FILE CHUNK {chunk}\n")
    
############################################################################### 
# NON-MODEL-SPECIFIC FEATURE ENGINEERING FOR TRAINING DATA

def preprocessing_Xtest(path=PATH_XTEST):
    X, id_column = load_chunk(path)
#    X = feature_engineer_test(X)
    X = apply_column_structure_of_train(X)
    X = apply_factorization_encoding(X)
    X = apply_imputation(X)
    X = apply_scaling(X, (0,1))
    X = apply_scaling(X, (-1,1)) # does need to reflect the order of preprocessing train data
    return X, id_column

def preprocessing_ytest(path=PATH_YTEST):
    iter_csv = pd.read_csv(path, usecols=Y_COLUMN, iterator=True, chunksize=CHUNKSIZE_TEST)
    y = next(iter_csv)
    return y

def main():
    '''prepares and returns X, y'''
    iterator = load_iterator()
    chunk = 0
    while True:
        print(f"\nCHUNK_{chunk}__________\n")
        X, id_column_for_submission  = preprocessing_Xtest(iterator)
        save_preprocessed_Xtest(X, chunk)
        save_submission_frame(id_column_for_submission, chunk)
        chunk += 1
