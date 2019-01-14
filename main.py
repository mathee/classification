"""This script combines alle major steps of the ML pipeline and serves as the main
controller. Check config.py for settings"""

from predict import ML_predict, NN_predict, combine_submission_chunks
from preprocess_test import main as preprocess_testingdata
from preprocess_train import main as preprocess_trainingdata
from train_ML import main as train_model
from train_NN import continue_training, initialize_training

###############################################################################
# DATA PREPARATION
"""
def wrangle_data():
    '''combine/wrangle data from different sources into single *.csv files'''
    wrangle_trainingdata()
    wrangle_testdata()
"""

def preprocess_traindata(): # set trainingset size in config
    '''preprocess trainingdata as defined in preprocess_train.py'''
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
def preprocess_testdata(): # set chunksize of testdata in config
    # ! BE SURE PROCESS MATCHES PREPROCESSING OF TRAININGDATA
    preprocess_testingdata()    

def predict_chunks_ML(modelname, chunk_first, chunk_last):
    ML_predict(modelname, chunk_first, chunk_last)
    
def predict_chunks_NN(modelname, chunk_first, chunk_last):
    NN_predict(modelname, chunk_first, chunk_last)
    
##############################################################################
# COMBINE SUBMISSION FILE CHUNKS
def create_submission_file(modelname, chunk_first, chunk_last):
    combine_submission_chunks(modelname, chunk_first, chunk_last)
