import pandas as pd
from config import Y_COLUMN, PATH_MODELS, CHUNKS_TEST, PATH_XTEST, PATH_YTEST, ID_COLUMN_LABEL, CHUNKSIZE_TEST, PATH_XTEST_PREPROCESSED, SEPARATOR, PATH_SUBMISSION_FILE_PREP
from feature_engineering import engineer_test as feature_engineer_test
import pickle

###############################################################################

def load_iterator(path=PATH_XTEST, chunksize=CHUNKSIZE_TEST):
    iter_csv = pd.read_csv(path, iterator=True, chunksize=chunksize)
    return iter_csv

def load_chunk(iterator):
    df = next(iterator)
    id_column = df[ID_COLUMN_LABEL]
    print(f"AFTER LOADING: {type(df)} - {df.shape}")
    return df, id_column

def apply_column_structure_of_train(X):
    with open(f'{PATH_MODELS}support/trainingdata.structure', 'rb') as handle:
        trainingdata_structure = pickle.load(handle) # dataframe
    a1, a2 = X.align(trainingdata_structure, join='right', axis=1)
    print(f"ALIGNED COLUMN STRUCTURE: {type(a1)} - {a1.shape}")
    return a1

def apply_one_hot_encoding(X):
    object_columns = list(X.select_dtypes(include=['object']))
    X = pd.get_dummies(X, prefix=object_columns)
    print(f"ONE-HOT-ENCODING: {type(X)} - {X.shape}")
    return X

def apply_factorization_encoding(X):
    object_columns = list(X.select_dtypes(include=['object']))
    for col in object_columns:
        with open(f'{PATH_MODELS}support/factorization_encoder_{col}.dict', 'rb') as handle:
            mapping = pickle.load(handle)
        X[col]= X[col].map(mapping).fillna(-1).astype(int)
    print(f"APPLIED FACTORIZATION ENCODING: {type(X)} - {X.shape}")
    return X

def apply_imputation(X):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_columns = list(X.select_dtypes(include=numerics))
    for col in numeric_columns:
        with open(f'{PATH_MODELS}support/imputer_{col}.model', 'rb') as handle:
            imp = pickle.load(handle)
        X[col] = imp.transform(X[col].values.reshape(-1, 1))
    print(f"APPLIED IMPUTATION: {type(X)} - {X.shape}")
    return X

def apply_scaling(X, feature_range):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_columns = list(X.select_dtypes(include=numerics))
    for columnlabel in numeric_columns:
        with open(f'{PATH_MODELS}support/scaler{feature_range}{columnlabel}.model', 'rb') as handle:
            scaler = pickle.load(handle)
        X[columnlabel] = scaler.transform(X[columnlabel].values.reshape(-1, 1))
    print(f"APPLIED SCALING {feature_range}: {type(X)} - {X.shape}")
    return X

def apply_PCA(X):
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
    for chunk in range(CHUNKS_TEST):
        print(f"\nCHUNK_{chunk}__________\n")
        X, id_column_for_submission  = preprocessing_Xtest(iterator)
        X = feature_engineer_test(X)
        save_preprocessed_Xtest(X, chunk)
    #    y = preprocessing_ytest()
    #    y = y.values.reshape(-1,)
    #    save_preprocessed_ytest(y)
        save_submission_frame(id_column_for_submission, chunk)
    
