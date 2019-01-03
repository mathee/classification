"""Model training"""

from engineering import main as prepareXy
from config import PATH_XTRAIN, PATH_YTRAIN, PATH_MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.externals.joblib import dump

SCORING = "accuracy"

###############################################################################
# SHARED FUNCTIONS

def do_gridsearch(m, parameter_grid):
    grid = GridSearchCV(m, 
                        param_grid=parameter_grid,
                        scoring=SCORING,
                        cv=5
                        )
    print("starting gridsearch...")
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
# LOADING AND NON-SPECIFIC FEATURE ENGINEERING
Xtrain, ytrain = prepareXy(PATH_XTRAIN, PATH_YTRAIN)

###############################################################################
# TRAIN LOGISTIC REGRESSION 

modelname = "LOGISTIC_REGRESSION_1"
m_logreg_init = LogisticRegression()
# DEFINE SEARCHSPACE FOR GRIDSEARCH
c_params = []
for i in range(5, 30, 1):
    c_params.append(i / 100.0)
penalty_types = ["l1", "l2"]
params_logreg = {"penalty":penalty_types,
          "C":c_params}
# GRIDSEARCH & SAVE TO DISC
grid_logreg = do_gridsearch(m_logreg_init, params_logreg)
save_best_model(grid_logreg, modelname)

