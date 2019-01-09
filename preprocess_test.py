import pandas as pd
from config import Y_COLUMN, PATH_MODELS, PATH_XTEST, PATH_YTEST, ID_COLUMN_LABEL, CHUNKSIZE_TEST, PATH_XTEST_PREPROCESSED, PATH_YTEST_PREPROCESSED, SEPARATOR, PATH_SUBMISSION_FILE
from feature_engineering import engineer_test as feature_engineer_test
import pickle

###############################################################################

def load_chunk(path):
    iter_csv = pd.read_csv(path, iterator=True, chunksize=CHUNKSIZE_TEST) #usecols = X_COLUMNS
    df = next(iter_csv)
    id_column = df[ID_COLUMN_LABEL]
    print(f"AFTER LOADING: {type(df)} - {df.shape}\n")
    return df, id_column

def apply_column_structure_of_train(X):
    with open(f'{PATH_MODELS}support/trainingdata.structure', 'rb') as handle:
        trainingdata_structure = pickle.load(handle) # dataframe
    a1, a2 = X.align(trainingdata_structure, join='right', axis=1)
    print(f"ALIGNED COLUMN STRUCTURE: {type(a1)} - {a1.shape}\n")
    return a1

def apply_one_hot_encoding(X):
    object_columns = list(X.select_dtypes(include=['object']))
    X = pd.get_dummies(X, prefix=object_columns)
    print(f"ONE-HOT-ENCODING: {type(X)} - {X.shape}\n")
    return X

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

def apply_scaling(X, feature_range):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_columns = list(X.select_dtypes(include=numerics))
    for columnlabel in numeric_columns:
        with open(f'{PATH_MODELS}support/scaler{feature_range}{columnlabel}.model', 'rb') as handle:
            scaler = pickle.load(handle)
        X[columnlabel] = scaler.transform(X[columnlabel].values.reshape(-1, 1))
    print(f"APPLIED SCALING {feature_range}: {type(X)} - {X.shape}\n")
    return X


############################################################################### 
# SAVING
    
def save_preprocessed_Xtest(X):
    X.to_csv(PATH_XTEST_PREPROCESSED, index = False, sep = SEPARATOR)
    print(f"SAVED Xtest TO DISC")
    
def save_preprocessed_ytest(y):
    y.to_csv(PATH_YTEST_PREPROCESSED, index = False, sep = SEPARATOR)    
    print(f"SAVED ytest TO DISC")
    
def save_submission_frame(id_column):
    '''optional, prepare file for competition submission, e.g. kaggle'''
    submission = pd.DataFrame(id_column, columns=[ID_COLUMN_LABEL])
    submission.to_csv(PATH_SUBMISSION_FILE, index = False, sep = SEPARATOR)    
    print(f"PREPARED SUBMISSION FILE")
    
    
############################################################################### 
# NON-MODEL-SPECIFIC FEATURE ENGINEERING FOR TRAINING DATA

def preprocessing_Xtest(path=PATH_XTEST):
    X, id_column = load_chunk(path)
#    X = apply_one_hot_encoding(X)
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
    X, id_column_for_submission  = preprocessing_Xtest()
    X = feature_engineer_test(X)
    save_preprocessed_Xtest(X)
#    y = preprocessing_ytest()
#    y = y.values.reshape(-1,)
#    save_preprocessed_ytest(y)
    save_submission_frame(id_column_for_submission)

#preprocessing_Xtest("data/test.csv")
    