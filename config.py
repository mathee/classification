# columns to be loaded from original csv, for training and testing
''' pathes to files that contain the raw data for xtrain, ytrain, xtest, ytest'''
CHUNKSIZE_TRAIN = 50000
CHUNKSIZE_TEST = 50000

PATH_XTRAIN = "data/titanic_train.csv"
PATH_YTRAIN = "data/titanic_train.csv"
PATH_XTEST = "data/titanic_test.csv"
PATH_YTEST = "data/titanic_test.csv"
ID_COLUMN_LABEL = "PassengerId"
Y_COLUMN = ["Survived"]
"""
PATH_XTRAIN = "data/train.csv"
PATH_YTRAIN = "data/train.csv"
PATH_XTEST = "data/test.csv"
PATH_YTEST = "data/test.csv"
#ID_COLUMN_LABEL = "MachineIdentifier"
#Y_COLUMN = ["HasDetections"]
"""

PATH_XTRAIN_PREPROCESSED = "data/preprocessed/Xtrain_preprocessed.csv"
PATH_YTRAIN_PREPROCESSED = "data/preprocessed/ytrain_preprocessed.csv"
PATH_XTEST_PREPROCESSED = "data/preprocessed/Xtest_preprocessed.csv"
PATH_YTEST_PREPROCESSED = "data/preprocessed/ytest_preprocessed.csv"
PATH_SUBMISSION_FILE_PREP = "results/temp.csv"
PATH_SUBMISSION_FILE = "results/submission"#_{modelname}.csv
PATH_MODELS = "models/"

SEPARATOR = "|"
SCORING = "accuracy" # score to be maxed by ML model gridsearch