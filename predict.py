"""this script contains predictions functions, i.e. applying trained models on
a set of preprocessed testdata"""

from config import SUBMISSION_TYPE, PATH_MODELS, CHUNKS_TEST, Y_COLUMN, PATH_XTEST_PREPROCESSED, SEPARATOR, PATH_SUBMISSION_FILE, PATH_SUBMISSION_FILE_PREP
from sklearn.externals.joblib import load
import pandas as pd
from keras.models import load_model
from keras import backend as K

###############################################################################
# DATA LOADING / SAVING

def load_preprocessed_Xtest(chunk):
    X = pd.read_csv(f"{PATH_XTEST_PREPROCESSED}_{chunk}.csv", sep = SEPARATOR)
    print(f"LOADED Xtest FROM DISC")
    return X

def load_submission_file(chunk):
    submission = pd.read_csv(f"{PATH_SUBMISSION_FILE_PREP}_{chunk}.csv",  sep = SEPARATOR)    
    print(f"LOADED EMPTY SUBMISSION FILE FROM DISC")
    return submission

def save_submission_file(ypred, modelname, chunk):
    '''optional, dave file for competition submission, e.g. kaggle'''
    submission_file = load_submission_file(chunk)
    submission_file[Y_COLUMN[0]] = ypred
    submission_file.to_csv(f"{PATH_SUBMISSION_FILE}_{modelname}_{chunk}.csv", index = False, sep = ",")    
    print(f"SAVED SUBMISSION FILE {chunk}")


###############################################################################
# PREDICT
    
def postprocess_ypred(ypred, submission_type):
    if submission_type == "float":
        return ypred.astype(float)
    elif submission_type == "int":
        return ypred.round(0).astype(int)
    else:
        return ypred

def make_predictions_ML(Xtest, modelname):
    path = f"{PATH_MODELS}{modelname}.model"
    m = load(path)
    y = m.predict(Xtest)
    y = postprocess_ypred(y, SUBMISSION_TYPE) # for binary problems, turns probabilites into 1 or 0
    print(f"MADE PREDICTIONS WITH {modelname}")
    #OR m.predict_proba(Xtest)[:,1] to receive probabilites, if outcome should not be binary
    return y
    
def make_predictions_NN(Xtest, nn_name):
    K.clear_session()
    path = f"{PATH_MODELS}{nn_name}.model"
    m = load_model(path)
    y = m.predict(Xtest)
    y = postprocess_ypred(y, SUBMISSION_TYPE) # for binary problems, turns probabilites into 1 or 0
#    y = y.argmax(axis=-1) #make probabilities distinct classes
    print(f"MADE PREDICTIONS WITH {nn_name}")
    return y

###############################################################################
# MAIN
    
def ML_predict(modelname, chunk_start, chunk_end):
    for chunk in range(chunk_start, chunk_end+1):
        print(f"\nPREDICT CHUNK_{chunk}__________\n")
        Xtest = load_preprocessed_Xtest(chunk)
        ypred = make_predictions_ML(Xtest, modelname)
        save_submission_file(ypred, modelname, chunk)

def NN_predict(modelname, chunk_start, chunk_end):
    for chunk in range(chunk_start, chunk_end+1): 
        print(f"\nPREDICT CHUNK_{chunk}__________\n")
        Xtest = load_preprocessed_Xtest(chunk)
        ypred = make_predictions_NN(Xtest, modelname)
        save_submission_file(ypred, modelname, chunk)

def combine_submission_chunks(modelname):
    dfs = []
    for chunk in range(CHUNKS_TEST):
        dfs.append(pd.read_csv(f"{PATH_SUBMISSION_FILE}_{modelname}_{chunk}.csv",  sep = ","))
    combined = pd.concat(dfs, ignore_index=False)
    combined.to_csv(f"{PATH_SUBMISSION_FILE}_{modelname}_combined.csv", index = False, sep = ",")   
    print(f"CREATED COMBINED SUBMISSION FILE OUT OF {CHUNKS_TEST} CHUNKS")
        
