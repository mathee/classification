"""this script combines alle major steps of the ML pipeline and serves as the main
controller"""

from wrangle import wrangle_trainingdata as wrangle_trainingdata
from wrangle import wrangle_testdata as wrangle_testdata
from preprocess_train import main as preprocess_trainingdata
#from preprocess_test import main as preprocess_testdata
from train_model import main as train_model
from train_neural_net import initialize_training, continue_training
#from test_model import main as test_model

###############################################################################
# DATA PREPARATION
def wrangle_data():
    wrangle_trainingdata()
    wrangle_testdata()
    
def preprocess_data():
    preprocess_trainingdata()
#    preprocess_testdata()

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
