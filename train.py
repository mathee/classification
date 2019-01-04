"""Model training"""

from preprocessing import main_train as preprocessing
from config import PATH_XTRAIN, PATH_YTRAIN, PATH_MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.externals.joblib import dump


SCORING = "accuracy"

###############################################################################
# LOADING AND APPLYING NON-SPECIFIC FEATURE ENGINEERING
Xtrain, ytrain = preprocessing(PATH_XTRAIN, PATH_YTRAIN)

###############################################################################
# TRAINING FUNCTIONS

def gridsearch(m, parameter_grid):
    grid = GridSearchCV(m, 
                        param_grid=parameter_grid,
                        scoring=SCORING,
                        cv=5
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
    #m = grid.best_estimator_
    return print(info)


###############################################################################
# TRAIN MODELS

def train_logistic_regression():
    modelname = "LOGISTIC_REGRESSION_1"
    m = LogisticRegression()
    
    # DEFINE SEARCHSPACE FOR GRIDSEARCH
    c_params = []
    for i in range(9, 12, 1):
        c_params.append(i / 100.0)
    penalty_types = ["l1", "l2"]
    params = {"penalty":penalty_types,
              "C":c_params}
    
    # GRIDSEARCH & SAVE TO DISC
    grid = gridsearch(m, params)
    save_best_model(grid, modelname)


def train_random_forest():
    modelname = "RANDOM_FOREST_1_SIMPLE"
    m = RandomForestClassifier(random_state=42)
    
    trees = []
    for i in range(5, 15):
        trees.append(i)
    depths = []
    for i in range(7, 14):
        depths.append(i)
    params = {"n_estimators":trees,
              "max_depth" : depths}
    
    grid = gridsearch(m, params)
    save_best_model(grid, modelname)
    
    
def train_SVC():
    modelname = "SUPPORT_VECTOR_MACHINE"
    m = SVC()
    
    #SVM
    c_params_svm = []
    for i in range(30, 32, 1): # e.g. 60 --> c 0.60
        c_params_svm.append(i / 100.0)
    kernels = ["rbf"]
    #kernel.append("rbf")
    gammas = []
    for i in range(10, 12, 1): # e.g. 60 --> c 6.0
        gammas.append(i)
    #gamma.append("auto")
    params = {"C":c_params_svm,
              "kernel": kernels,
              "gamma": gammas}
    
    
    grid = gridsearch(m, params)
    save_best_model(grid, modelname)
    
    
#print(Xtrain.shape, ytrain.shape)    
#train_random_forest()
#train_SVC()
#train_logistic_regression()