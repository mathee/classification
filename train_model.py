"""Model training"""

from config import PATH_XTRAIN, PATH_YTRAIN, PATH_MODELS, PATH_XTRAIN_PREPROCESSED, PATH_YTRAIN_PREPROCESSED, SEPARATOR
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.externals.joblib import dump
import pandas as pd


SCORING = "accuracy"


###############################################################################
# LOADING DATA

def load_preprocessed_Xtrain():
    X = pd.read_csv(PATH_XTRAIN_PREPROCESSED, sep = SEPARATOR)
    print(f"LOADED X FROM DISC")
    return X
    
def load_preprocessed_ytrain():
    y = pd.read_csv(PATH_YTRAIN_PREPROCESSED,  sep = SEPARATOR)    
    print(f"LOADED y FROM DISC")
    return y

###############################################################################
#  MODEL SPECIFIC TWEAKING/PREPROCESSING

def reshape_y(ytrain):
    '''ytrain comes as dataframe, needs little reshape before it can be used
    in scikit-learn models'''
    return ytrain.values.reshape(-1,)

###############################################################################
#  GRIDSEARCH FUNCTIONS
def gridsearch(Xtrain, ytrain, m, parameter_grid, cv=5):
    grid = GridSearchCV(m, 
                        param_grid=parameter_grid,
                        scoring=SCORING,
                        cv=cv
                        )
    print("performing gridsearch...")
    grid.fit(Xtrain, ytrain)
    print("done")
    return grid


def save_best_model(grid, modelname):
    info = f'''MODEL: {modelname}
BEST PARAMS: {grid.best_params_}
SCORE: {grid.best_score_}\n
GRIDSEARCH: {grid}\n
        '''
    with open(f"{PATH_MODELS}{modelname}_readme.txt", "w") as text_file:
        text_file.write(info)
    m = grid.best_estimator_
    dump(m, f'{PATH_MODELS}{modelname}.joblib')
    # load later via: clf = load('filename.joblib')  # from sklearn.externals.joblib import load
    return print(info)

###############################################################################
# TRAIN MODELS

def train_logistic_regression(Xtrain, ytrain):
    modelname = "LOGISTIC_REGRESSION"
    m = LogisticRegression()    
    
    ytrain = reshape_y(ytrain)
    
    # DEFINE SEARCHSPACE FOR GRIDSEARCH
    c_params = []
    for i in range(14, 20, 1):
        c_params.append(i / 100.0)
    penalty_types = ["l1", "l2"]
    params = {"penalty":penalty_types,
              "C":c_params}    
    # GRIDSEARCH & SAVE TO DISC
    grid = gridsearch(Xtrain, ytrain, m, params, cv=5)
    save_best_model(grid, modelname)


def train_random_forest(Xtrain, ytrain):
    modelname = "RANDOM_FOREST"
    m = RandomForestClassifier(random_state=42)    
    
    ytrain = reshape_y(ytrain)
    
    trees = []
    for i in range(15, 18):
        trees.append(i)
    depths = []
    for i in range(9, 12):
        depths.append(i)
    params = {"n_estimators":trees,
              "max_depth" : depths}    
    grid = gridsearch(Xtrain, ytrain, m, params, cv=5)
    save_best_model(grid, modelname)

    
def train_SVC(Xtrain, ytrain):
    # dont forget scaling!
    modelname = "SUPPORT_VECTOR_MACHINE"
    m = SVC()    
    
    ytrain = reshape_y(ytrain)
    
    #SVM
    c_params_svm = []
    for i in range(8, 12, 1): # e.g. 60 --> c 0.60
        c_params_svm.append(i / 100.0)
    kernels = ["rbf"]
    #kernel.append("rbf")
    gammas = []
    for i in range(6, 12, 1): # e.g. 60 --> c 6.0
        gammas.append(i)
    #gamma.append("auto")
    params = {"C":c_params_svm,
              "kernel": kernels,
              "gamma": gammas}    
    grid = gridsearch(Xtrain, ytrain, m, params, cv=5)
    save_best_model(grid, modelname)
    

def main():
    Xtrain = load_preprocessed_Xtrain()
    ytrain = load_preprocessed_ytrain()
    #train_logistic_regression(Xtrain, ytrain)
    train_random_forest(Xtrain, ytrain)
    #train_SVC(Xtrain, ytrain)

