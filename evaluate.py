"""this script contains evaluation functions for training / testing model
performance
"""

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from config import PATH_MODELS

###############################################################################
# EVALUATE GRIDSEARCH TRAINING PERFORMANCE
def get_gridsearch_classification_report(X,y,grid,modelname):
    '''creates report including some performance metrics on the model
    works only for classification problems'''
    m = grid.best_estimator_
    y_pred = m.predict(X)
    y_true = y
    conf_matrix = confusion_matrix(y_true, y_pred)
    report = f'''MODEL: {modelname}
BEST PARAMS: {grid.best_params_}
GRID_SCORE: {grid.best_score_}
BEST_ESTIMATOR_ACCURACY: {m.score(X,y)}
BEST_ESTIMATOR_PRECISION: {precision_score(y_true, y_pred)}
BEST_ESTIMATOR_RECALL: {recall_score(y_true, y_pred)}
CONFUSION_MATRIX: \n{conf_matrix}\n
GRIDSEARCH: {grid}\n
        '''
    return report

def get_gridsearch_regression_report(grid,modelname):
    '''creates report for regression problem'''
    report = f'''MODEL: {modelname}
BEST PARAMS: {grid.best_params_}
GRID_SCORE: {grid.best_score_}
GRIDSEARCH: {grid}\n
        '''
    return report

def save_gridsearch_report(X, y, grid, modelname):
    try: #will give an error if problem is regression
        report = get_gridsearch_classification_report(X,y,grid,modelname)
    except:
        report = get_gridsearch_regression_report(grid,modelname) # does not exist yet
    with open(f"{PATH_MODELS}{modelname}_readme.txt", "w") as text_file:
        text_file.write(report)
    print(report)    
    
def save_gridsearch_results(grid, modelname):
    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results.to_csv(f"{PATH_MODELS}{modelname}_gridsearch.csv", index = False, sep = ";")
    
def evaluate_gridsearch(X, y, grid, modelname):
    save_gridsearch_report(X, y, grid, modelname)
    save_gridsearch_results(grid, modelname)
    
###############################################################################
# EVALUATE NEURAL NET TRAINING PERFORMANCE
    
def save_learning_history(history, modelname):
    out = pd.DataFrame(history.history)
    out.to_csv(f"{PATH_MODELS}{modelname}_learning_history.csv", sep = ";")
    
def save_model_summary(model, modelname):    
    with open(f"{PATH_MODELS}{modelname}_readme.txt", "w") as text_file:
        model.summary(print_fn=lambda x: text_file.write(x + '\n'))
    
def evaluate_nn(history, model, modelname):
    save_learning_history(history, modelname)
    save_model_summary(model, modelname)
