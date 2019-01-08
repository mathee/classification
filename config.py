# columns to be loaded from original csv, for training and testing
''' pathes to files that contain the raw data for xtrain, ytrain, xtest, ytest'''
CHUNKSIZE_TRAIN = 200000
CHUNKSIZE_TEST = 200000
#PATH_XTRAIN = "data/titanic_train.csv"
#PATH_YTRAIN = "data/titanic_train.csv"
PATH_XTRAIN = "data/train.csv"
PATH_YTRAIN = "data/train.csv"
PATH_XTEST = "data/test.csv"
PATH_YTEST = "data/test.csv"

PATH_XTRAIN_PREPROCESSED = "data/preprocessed/Xtrain_preprocessed.csv"
PATH_YTRAIN_PREPROCESSED = "data/preprocessed/ytrain_preprocessed.csv"
PATH_XTEST_PREPROCESSED = "data/preprocessed/Xtest_preprocessed.csv"
PATH_YTEST_PREPROCESSED = "data/preprocessed/ytest_preprocessed.csv"


Y_COLUMN = ["HasDetections"]
PATH_MODELS = "models/"

SEPARATOR = "|"
SCORING = "accuracy" # score to be maxed by gridsearch