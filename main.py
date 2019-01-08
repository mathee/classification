"""this script combines alle major steps of the ML pipeline and serves as the main
controller"""

from wrangle import wrangle_trainingdata as wrangle_trainingdata
from wrangle import wrangle_testdata as wrangle_testdata
from preprocess_train import main as preprocess_trainingdata
from preprocess_test import main as preprocess_testdata
from train_model import main as train_model
#from test_model import main as test_model

def training_process():
#    wrangle_trainingdata()
    preprocess_trainingdata()
#    train_model()
    
#def test_process():
#    wrangle_testdata()
#    preprocess_testdata()
#    test_model()
    
def main():
    training_process()
    #test_process()

main()