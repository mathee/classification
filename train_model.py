"""Model training"""

from config import PATH_MODELS, PATH_XTRAIN_PREPROCESSED, PATH_YTRAIN_PREPROCESSED, SEPARATOR, SCORING
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.externals.joblib import dump
import pandas as pd
from evaluate import evaluate_gridsearch

###############################################################################
# LOADING DATA
def load_preprocessed_Xtrain():
    X = pd.read_csv(PATH_XTRAIN_PREPROCESSED, sep = SEPARATOR)
    print(f"LOADED Xtrain FROM DISC")
    return X
    
def load_preprocessed_ytrain():
    y = pd.read_csv(PATH_YTRAIN_PREPROCESSED,  sep = SEPARATOR)    
    print(f"LOADED ytrain FROM DISC")
    return y

###############################################################################
#  MODEL SPECIFIC TWEAKING/PREPROCESSING
def reshape_y(ytrain):
    '''ytrain comes as dataframe, needs little reshape before it can be used
    in scikit-learn models'''
    return ytrain.values.reshape(-1,)

def convert_y_to_float(ytrain):
    ytrain = ytrain.astype('float64')
    return ytrain

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
    m = grid.best_estimator_
    dump(m, f'{PATH_MODELS}{modelname}.model')
    # load later via: clf = load('filename.model')  # from sklearn.externals.joblib import load

###############################################################################
# CLASSIFICATION/REGRESSION GRIDSEARCH

def get_parameters(modelname):
    '''define here the parameter search spaces for specific models'''
    if modelname == "RANDOM_FOREST":
        n_estimators = [i for i in range(16, 18)]
        depths = [i for i in range(11, 13)]
        parameters = {"n_estimators":n_estimators,
                  "max_depth" : depths} 
        return parameters
    elif modelname == "LINEAR_REGRESSION": # etc.
        return {}
    else: 
        return {}

def train_grid(Xtrain, ytrain, modelname, model, parameters):
    '''logistic regression, random forest, support vector machines,
    linear regression, ridge regression, lasso regression, elasticnet'''
    ytrain = reshape_y(ytrain)
    grid = gridsearch(Xtrain, ytrain, model, parameters, cv=5)
    save_best_model(grid, modelname)
    evaluate_gridsearch(Xtrain, ytrain, grid, modelname)
    
###############################################################################
# MAIN FUNCTION
def main():
    Xtrain = load_preprocessed_Xtrain()
    ytrain = load_preprocessed_ytrain()
    
    train_grid(Xtrain, ytrain, 
               "RANDOM_FOREST", 
               RandomForestClassifier(random_state=42), 
               get_parameters("RANDOM_FOREST"))
    
    train_grid(Xtrain, ytrain, 
               "LOGISTIC_REGRESSION", 
               LogisticRegression(), 
               get_parameters("LOGISTIC_REGRESSION"))
    


