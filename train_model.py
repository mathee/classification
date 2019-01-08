"""Model training"""

from config import PATH_XTRAIN, PATH_YTRAIN, PATH_MODELS, PATH_XTRAIN_PREPROCESSED, PATH_YTRAIN_PREPROCESSED, SEPARATOR, SCORING
from sklearn.model_selection import GridSearchCV
from sklearn.externals.joblib import dump
import pandas as pd

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
# REGRESSION MODELS

def train_linear_regression(Xtrain, ytrain):
    from sklearn.linear_model import LinearRegression
    '''for non-binary regression problems''' 
    modelname = "LINEAR_REGRESSION"
    m = LinearRegression()
    ytrain = reshape_y(ytrain)
    params = {}
    grid = gridsearch(Xtrain, ytrain, m, params, cv=5)
    save_best_model(grid, modelname)
    
def train_lasso_regression(Xtrain, ytrain):
    from sklearn.linear_model import Lasso
    '''for non-binary regression problems''' 
    modelname = "LASSO_REGRESSION"
    m = Lasso()
    ytrain = reshape_y(ytrain)
    params = {"alpha":[0.4, 0.5, 0.6]}
    grid = gridsearch(Xtrain, ytrain, m, params, cv=5)
    save_best_model(grid, modelname)
    
def train_ridge_regression(Xtrain, ytrain):
    from sklearn.linear_model import Ridge
    '''for non-binary regression problems''' 
    modelname = "RIDGE_REGRESSION"
    m = Ridge()
    ytrain = reshape_y(ytrain)
    params = {"alpha":[0.01, 0.02]}
    grid = gridsearch(Xtrain, ytrain, m, params, cv=5)
    save_best_model(grid, modelname)
    
def train_elasticnet(Xtrain, ytrain):
    from sklearn.linear_model import ElasticNet
    '''for non-binary regression problems''' 
    modelname = "ELASTICNET_REGRESSION"
    m = ElasticNet()
    ytrain = reshape_y(ytrain)
    params = {"alpha" : [0.5], "l1_ratio" : [0.5]}
    grid = gridsearch(Xtrain, ytrain, m, params, cv=5)
    save_best_model(grid, modelname)

###############################################################################
# CLASSIFICATION MODELS

def train_logistic_regression(Xtrain, ytrain):
    from sklearn.linear_model import LogisticRegression
    modelname = "LOGISTIC_REGRESSION"
    m = LogisticRegression()    
    ytrain = reshape_y(ytrain)
    c_params = []
    for i in range(14, 20, 1):
        c_params.append(i / 100.0)
    penalty_types = ["l1", "l2"]
    params = {"penalty":penalty_types,
              "C":c_params}    
    grid = gridsearch(Xtrain, ytrain, m, params, cv=5)
    save_best_model(grid, modelname)


def train_random_forest(Xtrain, ytrain):
    from sklearn.ensemble import RandomForestClassifier
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
    from sklearn.svm import SVC
    modelname = "SUPPORT_VECTOR_MACHINE"
    m = SVC()    
    ytrain = reshape_y(ytrain)
    c_params_svm = []
    for i in range(8, 12, 1): # e.g. 60 --> c 0.60
        c_params_svm.append(i / 100.0)
    kernels = ["rbf"]
    gammas = []
    for i in range(6, 12, 1): # e.g. 60 --> c 6.0
        gammas.append(i)
    gammas.append("auto")
    params = {"C":c_params_svm,
              "kernel": kernels,
              "gamma": gammas}    
    grid = gridsearch(Xtrain, ytrain, m, params, cv=5)
    save_best_model(grid, modelname)
    
###############################################################################
# MAIN FUNCTION

def main():
    Xtrain = load_preprocessed_Xtrain()
    ytrain = load_preprocessed_ytrain()
    train_random_forest(Xtrain, ytrain)


