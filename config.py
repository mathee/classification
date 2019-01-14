# columns to be loaded from original csv, for training and testing
''' pathes to files that contain the raw data for xtrain, ytrain, xtest, ytest'''
CHUNKSIZE_TRAIN = 5000000
CHUNKSIZE_TEST = 200 # iterator chunk sizes for loading testdata
CHUNKS_TEST = 3 # chunks of testdata to be preprocessed

SEPARATOR = "|"
###############################################################################
# CHOOSE FILE PATHES

PATH_XTRAIN = "data/titanic_train.csv"
PATH_YTRAIN = "data/titanic_train.csv"
PATH_XTEST = "data/titanic_test.csv"
PATH_YTEST = "data/titanic_test.csv"
ID_COLUMN_LABEL = "PassengerId"
Y_COLUMN = ["Survived"]
SUBMISSION_TYPE = "int"
"""
PATH_XTRAIN = "data/microsoft_train.csv"
PATH_YTRAIN = "data/microsoft_train.csv"
PATH_XTEST = "data/microsoft_test.csv"
PATH_YTEST = "data/microsoft_test.csv"
ID_COLUMN_LABEL = "MachineIdentifier"
Y_COLUMN = ["HasDetections"]
SUBMISSION_TYPE = "float"

PATH_XTRAIN = "data/housing_train.csv"
PATH_YTRAIN = "data/housing_train.csv"
PATH_XTEST = "data/housing_test.csv"
PATH_YTEST = "data/housing_test.csv"
ID_COLUMN_LABEL = "Id"
Y_COLUMN = ["SalePrice"]
SUBMISSION_TYPE = "float"

PATH_XTRAIN = "data/pubg_train.csv"
PATH_YTRAIN = "data/pubg_train.csv"
PATH_XTEST = "data/pubg_test.csv"
PATH_YTEST = "data/pubg_test.csv"
ID_COLUMN_LABEL = "Id"
Y_COLUMN = ["winPlacePerc"]
SUBMISSION_TYPE = "float"
"""
PATH_XTRAIN_PREPROCESSED = "data/preprocessed/Xtrain_preprocessed.csv"
PATH_YTRAIN_PREPROCESSED = "data/preprocessed/ytrain_preprocessed.csv"
PATH_XTEST_PREPROCESSED = "data/preprocessed/Xtest_preprocessed"#_{chunk}.csv
PATH_SUBMISSION_FILE_PREP = "results/temp"
PATH_SUBMISSION_FILE = "results/submission"#_{modelname}.csv
PATH_MODELS = "models/"
