"""this script combines alle major steps of the ML pipeline and serves as the main
controller"""

from wrangle import wrangle_trainingdata as wrangle_trainingdata
from wrangle import wrangle_testdata as wrangle_testdata
from preprocess_train import main as preprocess_trainingdata
from preprocess_test import main as preprocess_testingdata
from train_model import main as train_model
from train_neural_net import initialize_training, continue_training
from test_model import test_ML_model, test_neural_net


###############################################################################
# DATA PREPARATION
def wrangle_data():
    wrangle_trainingdata()
    wrangle_testdata()
    
def preprocess_traindata():
    preprocess_trainingdata()
    
###############################################################################
# TRAIN ML MODEL
def train_ML_model():
    train_model()

###############################################################################
# TRAIN NEURAL NETWORK 
def initialize_nn():
    initialize_training()
    
def continue_training_nn(epochs):
    continue_training(epochs)

###############################################################################
# APPLY MODEL ON UNSEEN DATA (PREDICT ON TEST) 
def preprocess_testdata():
    preprocess_testingdata()    

def apply_ML_model(modelname):
    test_ML_model(modelname)
    
def apply_nn():
    test_neural_net()