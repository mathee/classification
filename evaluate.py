"""this script contains evaluation functions for training / testing model
performance"""

from config import PATH_MODELS
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd

# add ROC matrix

###############################################################################
# EVALUATE GRIDSEARCH TRAINING PERFORMANCE
def save_gridsearch_scores(X, y, grid, modelname):
    m = grid.best_estimator_
    y_pred = m.predict(X)
    y_true = y
    conf_matrix = confusion_matrix(y_true, y_pred)
    info = f'''MODEL: {modelname}
BEST PARAMS: {grid.best_params_}
GRID_SCORE: {grid.best_score_}
BEST_ESTIMATOR_ACCURACY: {m.score(X,y)}
BEST_ESTIMATOR_PRECISION: {precision_score(y_true, y_pred)}
BEST_ESTIMATOR_RECALL: {recall_score(y_true, y_pred)}
CONFUSION_MATRIX: \n{conf_matrix}\n
GRIDSEARCH: {grid}\n
        '''
    with open(f"{PATH_MODELS}{modelname}_readme.txt", "w") as text_file:
        text_file.write(info)
    print(info)    
    
def save_gridsearch_results(grid, modelname):
    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results.to_csv(f"{PATH_MODELS}{modelname}_gridsearch.csv", index = False, sep = ";")
    
def evaluate_gridsearch(X, y, grid, modelname):
    save_gridsearch_scores(X, y, grid, modelname)
    save_gridsearch_results(grid, modelname)
    
###############################################################################
# EVALUATE NEURAL NET TRAINING PERFORMANCE