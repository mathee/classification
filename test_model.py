"""
+ m.predict,
+ optional scoring on test
+ optional make submission
"""
from config import PATH_MODELS, PATH_XTEST, PATH_YTEST
from sklearn.externals.joblib import load
import pandas as pd

"""
Xtest, id_column = preprocessing(PATH_XTEST, PATH_YTEST)

def predict(model):
    path = f"{PATH_MODELS}{model}.joblib"
    m = load(path)
    y = m.predict(Xtest)
    return y
    

y = predict("RANDOM_FOREST")
submission = pd.DataFrame()
submission["MachineIdentifier"] = id_column
submission["HasDetections"] = y

print(submission.head(5))
print(submission.shape)
"""