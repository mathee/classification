"""this script combines alle major steps of the ML pipeline and serves as the main
controller"""

from wrangle import wrangle_trainingdata as wrangle_trainingdata
from wrangle import wrangle_testdata as wrangle_testdata
from preprocess_train import main as preprocess_trainingdata
from preprocess_test import main as preprocess_testingdata
from train_ML import main as train_model
from train_NN import initialize_training, continue_training
from predict import ML_predict, NN_predict, combine_submission_chunks


###############################################################################
# DATA PREPARATION
def wrangle_data():
    wrangle_trainingdata()
    wrangle_testdata()
    
def preprocess_traindata():
    preprocess_trainingdata()
    
###############################################################################
# TRAIN ML MODEL
def train_ML_model(modelname):
    train_model(modelname)

###############################################################################
# TRAIN NEURAL NETWORK 
def initialize_nn(modelname):
    initialize_training(modelname)
    
def continue_training_nn(modelname, epochs):
    continue_training(modelname, epochs)

###############################################################################
# APPLY MODEL ON UNSEEN DATA (PREDICT ON TEST) 
def preprocess_testdata():
    preprocess_testingdata()    

def apply_ML(modelname, chunk_start, chunk_end):
    ML_predict(modelname, chunk_start, chunk_end)
    
def apply_NN(modelname, chunk_start, chunk_end):
    NN_predict(modelname, chunk_start, chunk_end)
    
##############################################################################
# COMBINE SUBMISSION FILE CHUNKS
def create_submission_file(modelname):
    combine_submission_chunks(modelname)