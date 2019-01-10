"""
+ m.predict,
+ optional scoring on test
+ optional make submission
"""
from config import PATH_MODELS, PATH_XTEST, PATH_YTEST, Y_COLUMN, PATH_XTEST_PREPROCESSED, PATH_YTEST_PREPROCESSED, SEPARATOR, PATH_SUBMISSION_FILE, PATH_SUBMISSION_FILE_PREP
from sklearn.externals.joblib import load
import pandas as pd
from keras.models import load_model
from keras import backend as K

###############################################################################
# DATA LOADING / SAVING

def load_preprocessed_Xtest():
    X = pd.read_csv(PATH_XTEST_PREPROCESSED, sep = SEPARATOR)
    print(f"LOADED Xtest FROM DISC\n")
    return X
    
def load_preprocessed_ytest():
    y = pd.read_csv(PATH_YTEST_PREPROCESSED,  sep = SEPARATOR)    
    print(f"LOADED ytest FROM DISC\n")
    return y

def load_submission_file():
    submission = pd.read_csv(PATH_SUBMISSION_FILE_PREP,  sep = SEPARATOR)    
    print(f"LOADED EMPTY SUBMISSION FILE FROM DISC\n")
    return submission

def save_submission_file(ypred, modelname):
    '''optional, dave file for competition submission, e.g. kaggle'''
    submission_file = load_submission_file()
    submission_file[Y_COLUMN[0]] = ypred
    submission_file.to_csv(f"{PATH_SUBMISSION_FILE}_{modelname}.csv", index = False, sep = ",")    
    print(f"SAVED SUBMISSION FILE\n")

###############################################################################
# PREDICT

def predict_ML(Xtest, modelname):
    path = f"{PATH_MODELS}{modelname}.model"
    m = load(path)
    y = m.predict(Xtest)
    y = y.round(0).astype(int) # for binary problems, turns probabilites into 1 or 0
    print(f"MADE PREDICTIONS WITH {modelname}\n")
    #OR m.predict_proba(Xtest)[:,1] to receive probabilites, if outcome should not be binary
    return y
    
def predict_NN(Xtest, nn_name):
    K.clear_session()
    path = f"{PATH_MODELS}{nn_name}.model"
    m = load_model(path)
    y = m.predict(Xtest)
    y = y.round(0).astype(int) # for binary problems, turns probabilites into 1 or 0
#    y = y.argmax(axis=-1) #make probabilities distinct classes
    print(f"MADE PREDICTIONS\n")
    return y

###############################################################################
# MAIN

def test_ML_model(modelname):
    Xtest = load_preprocessed_Xtest()
#    ytest = load_preprocessed_ytest()
    ypred = predict_ML(Xtest, modelname)
    save_submission_file(ypred, modelname)

def test_neural_net(modelname = "NEURAL_NET"):
    Xtest = load_preprocessed_Xtest()
#    ytest = load_preprocessed_ytest()
    ypred = predict_NN(Xtest, modelname)
    save_submission_file(ypred, modelname)
    