''' pathes to files that contain the raw data for xtrain, ytrain, xtest, ytest'''
###############################################################################
# GENERAL SETTINGS
SEPARATOR = "|" # separator for saving preprocessed files, to avoid confusion with "," values in file
###############################################################################
# PATHES
PATH_XTRAIN = "data/wrangled/train.csv"
PATH_YTRAIN = "data/wrangled/train.csv"
PATH_XTEST = "data/wrangled/test.csv"
PATH_YTEST = "data/wrangled/test.csv"
ID_COLUMN_LABEL = "MachineIdentifier" # name/label of identifier column
Y_COLUMN = ["HasDetections"] # column to be predicted, save here as [list]
SUBMISSION_TYPE = "float" # datatype of predicted values, used in postprocess_ypred in predictions.py

PATH_XTRAIN_PREPROCESSED = "data/preprocessed/Xtrain_preprocessed.csv"
PATH_YTRAIN_PREPROCESSED = "data/preprocessed/ytrain_preprocessed.csv"
PATH_XTEST_PREPROCESSED = "data/preprocessed/Xtest_preprocessed"#_{chunk}.csv
PATH_SUBMISSION_FILE_PREP = "results/temp"
PATH_SUBMISSION_FILE = "results/submission"#_{modelname}.csv
PATH_MODELS = "models/"
PATH_REPORTS = "reports/"
