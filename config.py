# columns to be loaded from original csv, for training and testing

''' pathes to files that contain the raw data for xtrain, ytrain, xtest, ytest'''
PATH_XTRAIN = "data/train.csv"
PATH_YTRAIN = "data/train.csv"
PATH_XTEST = "data/test.csv"
PATH_YTEST = "data/test.csv"


X_COLUMNS = ["AVProductStatesIdentifier",
             "AVProductsInstalled"
            ]
Y_COLUMN = ["HasDetections"]



PATH_MODELS = "models/"